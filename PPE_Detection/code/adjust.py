from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import time
from pi5neo import Pi5Neo
import threading

# --- Global variables for threading ---
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()
# ------------------------------------

# --- pi5Neo LED Setup ---
NUM_LEDS = 35         # <-- From your summer code
SPI_SPEED_KHZ = 800
SPI_DEVICE = "/dev/spidev0.0"

try:
    neo = Pi5Neo(SPI_DEVICE, NUM_LEDS, SPI_SPEED_KHZ)
    print(f"pi5Neo strip initialized on {SPI_DEVICE} with {NUM_LEDS} LEDs.")
except Exception as e:
    print(f"Failed to initialize pi5Neo: {e}")
    exit()

# --- Define Colors ---
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_OFF = (0, 0, 0)
# -------------------------

# --- Frame Grabber Thread Function ---
def frame_grabber(cam_proc, FRAME_SIZE, WIDTH, HEIGHT):
    global latest_frame, frame_lock, stop_event
    print("Frame grabber thread started.")
    while not stop_event.is_set():
        try:
            raw = cam_proc.stdout.read(FRAME_SIZE)
            if not raw:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            if yuv.size != FRAME_SIZE:
                continue
            yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            with frame_lock:
                latest_frame = frame
        except Exception:
            if not stop_event.is_set():
                print("Error in frame grabber")
            break
    print("Frame grabber thread stopped.")
# -------------------------------------

# Load trained model
MODEL_PATH = "/home/pi/PPE_Detection/code/best.pt"
model = YOLO(MODEL_PATH)

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 320
STABLE_THRESHOLD = 3  # From your summer code

# --- Per-class thresholds (from your summer code) ---
# ! IMPORTANT: This assumes your model has a class named "eyeglasses"
class_thresholds = {
    'goggles': 0.7,
    'eyeglasses': 0.25,
    'none': 0.3
}

# Detection memory (for on-screen text only)
last_detected_class = None
stable_class = None
stable_counter = 0

# --- UPDATED Camera Command ---
# Uses rpicam-vid (modern), --nopreview (hides extra window), and 30fps
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0",
    "--framerate", "30",
    "--nopreview"
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# --- Start the frame grabber thread ---
grabber_thread = threading.Thread(
    target=frame_grabber,
    args=(cam_proc, FRAME_SIZE, WIDTH, HEIGHT)
)
grabber_thread.start()

try:
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame_to_process = latest_frame.copy()

        # --- PREDICTION (from summer code) ---
        # Predict with low threshold to get all potential boxes
        results = model.predict(
            frame_to_process,
            imgsz=YOLO_IMGSZ,
            conf=0.01,  # Predict everything
            device="cpu",
            verbose=False,
            half=True   # Use FP16 for speed
        )
        
        # --- MANUAL FILTER (from summer code) ---
        filtered_boxes = []
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                
                # Check if this class is one we track and if it meets its threshold
                if label in class_thresholds and conf >= class_thresholds[label]:
                    filtered_boxes.append(box)
        
        # Re-assign the filtered boxes to the results object for plotting
        results[0].boxes = filtered_boxes
        # ----------------------------------------

        # --- INSTANT LED LOGIC (from summer code) ---
        # Get all labels from the *filtered* boxes
        labels = [model.names[int(box.cls)] for box in filtered_boxes]
        
        new_color = COLOR_OFF # Default to OFF
        
        # "Safety First" logic: If ANY "bad" items are seen, turn red
        if any(lbl in ["none", "eyeglasses"] for lbl in labels):
            new_color = COLOR_RED
        elif "goggles" in labels:
            new_color = COLOR_GREEN
            
        # Update LEDs every single frame
        neo.fill_strip(*new_color)
        neo.update_strip()
        # ----------------------------------------

        # --- TEXT LABEL SMOOTHING (from summer code) ---
        # This logic is now *only* for the on-screen text
        detected_class = labels[0] if labels else None # Get top detection for text
        
        if detected_class == last_detected_class and detected_class is not None:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_THRESHOLD:
            stable_class = detected_class
        elif stable_counter == 0 and detected_class is None:
             stable_class = None
             
        last_detected_class = detected_class
        # ----------------------------------------

        # Annotate and show
        annotated = results[0].plot()
        label_text = stable_class if stable_class else "No Detection"
        cv2.putText(annotated, f"Detected: {label_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOv8 (Summer Logic + pi5Neo)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    print("\nShutting down...")
    stop_event.set()
    grabber_thread.join()
    cam_proc.kill()
    cv2.destroyAllWindows()
    neo.clear_strip()  # Set all LEDs to (0,0,0)
    neo.update_strip() # Push the "off" command
    print("Done.")

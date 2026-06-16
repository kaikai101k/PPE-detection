from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import time
from pi5neo import Pi5Neo
import threading  # <-- NEW: Import threading

# --- Global variables for threading ---
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()
# ------------------------------------

# --- pi5Neo LED Setup ---
NUM_LEDS = 30
SPI_SPEED_KHZ = 800
SPI_DEVICE = "/dev/spidev0.0"

try:
    neo = Pi5Neo(SPI_DEVICE, NUM_LEDS, SPI_SPEED_KHZ)
    print(f"pi5Neo strip initialized on {SPI_DEVICE} with {NUM_LEDS} LEDs.")
except Exception as e:
    print(f"Failed to initialize pi5Neo: {e}")
    exit()

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_OFF = (0, 0, 0)

# --- Frame Grabber Thread Function ---
def frame_grabber(cam_proc, FRAME_SIZE, WIDTH, HEIGHT):
    """
    This function runs in a separate thread.
    It continuously reads frames from the camera process
    and updates the global 'latest_frame'.
    """
    global latest_frame, frame_lock, stop_event
    print("Frame grabber thread started.")
    while not stop_event.is_set():
        try:
            raw = cam_proc.stdout.read(FRAME_SIZE)
            if not raw:
                print("? Grabber: Frame grab failed.")
                break

            yuv = np.frombuffer(raw, dtype=np.uint8)
            if yuv.size != FRAME_SIZE:
                continue

            yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            
            # Use a lock to safely update the global frame
            with frame_lock:
                latest_frame = frame
        
        except Exception as e:
            if not stop_event.is_set():
                print(f"Error in frame grabber: {e}")
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
STABLE_THRESHOLD = 2

CLASS_THRESHOLDS = {
    "goggles": 0.7,
    "glasses": 0.6,
    "none": 0.6
}
GLOBAL_CONF_THRESHOLD = min(CLASS_THRESHOLDS.values())

# Detection memory
last_detected_class = None
stable_class = None
stable_counter = 0

# --- UPDATED Camera Command ---
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0",
    "--framerate", "30", # <-- Increased framerate
    "--nopreview"         # <-- Kills the extra preview window
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# --- Start the frame grabber thread ---
grabber_thread = threading.Thread(
    target=frame_grabber, 
    args=(cam_proc, FRAME_SIZE, WIDTH, HEIGHT)
)
grabber_thread.start()
# --------------------------------------

try:
    while True:
        # Wait for the first frame to be captured
        with frame_lock:
            if latest_frame is None:
                continue
            # Make a copy for processing
            frame_to_process = latest_frame.copy()

        # --- Run YOLOv8 prediction on the latest frame ---
        results = model.predict(
            frame_to_process,
            imgsz=YOLO_IMGSZ,
            conf=GLOBAL_CONF_THRESHOLD, 
            device="cpu",
            verbose=False,
            half=True  # <-- NEW: Use FP16 for potential speedup
        )
        # -----------------------------------------------
        
        boxes = results[0].boxes
        class_names = results[0].names 

        # Filter detections
        detected_class = None
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names[cls_id] 
                if class_name in CLASS_THRESHOLDS:
                    if conf >= CLASS_THRESHOLDS[class_name]:
                        detected_class = class_name
                        break 
        
        # Temporal smoothing
        if detected_class == last_detected_class and detected_class is not None:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_THRESHOLD:
            stable_class = detected_class
        elif stable_counter == 0 and detected_class is None:
             stable_class = None
        last_detected_class = detected_class

        # pi5Neo LED Control
        new_color = COLOR_OFF
        if stable_class == "goggles":
            new_color = COLOR_GREEN
        elif stable_class in ["glasses", "none"]:
            new_color = COLOR_RED
        
        neo.fill_strip(*new_color)
        neo.update_strip()

        # Annotate and display
        annotated = results[0].plot()
        label = stable_class if stable_class else "No Detection"
        cv2.putText(annotated, f"Detected: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # This is now the ONLY window that will appear
        cv2.imshow("YOLOv8 (Smoothed)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Cleanup ---
    print("\nShutting down...")
    stop_event.set()       # Signal the grabber thread to stop
    grabber_thread.join()  # Wait for the thread to finish
    cam_proc.kill()        # Kill the camera process
    cv2.destroyAllWindows()
    neo.clear_strip()
    neo.update_strip()
    print("Done.")

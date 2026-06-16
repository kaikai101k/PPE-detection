from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import time
from pi5neo import Pi5Neo  # <-- NEW: Using the pi5Neo library

# --- pi5Neo LED Setup ---
NUM_LEDS = 30         # <-- IMPORTANT: Change this to your LED count
SPI_SPEED_KHZ = 800
SPI_DEVICE = "/dev/spidev0.0"

# Initialize the pi5Neo strip
try:
    neo = Pi5Neo(SPI_DEVICE, NUM_LEDS, SPI_SPEED_KHZ)
    print(f"pi5Neo strip initialized on {SPI_DEVICE} with {NUM_LEDS} LEDs.")
except Exception as e:
    print(f"Failed to initialize pi5Neo: {e}")
    print("Make sure SPI is enabled (sudo raspi-config) and you have permissions.")
    print("You may need to run: sudo usermod -aG spi $USER (and then reboot)")
    exit()

# --- Define Colors ---
COLOR_GREEN = (0, 255, 0)  # Green for 'goggles'
COLOR_RED = (255, 0, 0)    # Red for 'glasses' or 'none'
COLOR_OFF = (0, 0, 0)      # Off for no detection
# -------------------------

# Load trained model
MODEL_PATH = "/home/pi/PPE_Detection/code/best.pt"
model = YOLO(MODEL_PATH)

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 320
STABLE_THRESHOLD = 2  # number of consistent frames required

# Per-Class Confidence Thresholds
# ! IMPORTANT: Class names must EXACTLY match your model's class names
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

# Start camera command
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0", "--framerate", "15"
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_id = 0

try:
    while True:
        raw = cam_proc.stdout.read(FRAME_SIZE)
        if not raw:
            print("? Frame grab failed.")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8)
        if yuv.size != FRAME_SIZE:
            continue

        yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        frame_id += 1
        if frame_id % 2 != 0:
            continue # Skip frame

        # Run YOLOv8 prediction
        results = model.predict(
            frame,
            imgsz=YOLO_IMGSZ,
            conf=GLOBAL_CONF_THRESHOLD, 
            device="cpu",
            verbose=False
        )
        
        boxes = results[0].boxes
        class_names = results[0].names 

        # Filter detections based on per-class thresholds
        detected_class = None
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names[cls_id] 

                if class_name in CLASS_THRESHOLDS:
                    if conf >= CLASS_THRESHOLDS[class_name]:
                        detected_class = class_name
                        break # Found highest-conf match passing its threshold
        
        # Temporal smoothing logic
        if detected_class == last_detected_class and detected_class is not None:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_THRESHOLD:
            stable_class = detected_class
        elif stable_counter == 0 and detected_class is None:
             stable_class = None # Reset stable class if no detection
             
        last_detected_class = detected_class

        # --- UPDATED: pi5Neo LED Control Logic ---
        new_color = COLOR_OFF # Default to OFF

        if stable_class == "goggles":
            new_color = COLOR_GREEN
        elif stable_class in ["glasses", "none"]:
            new_color = COLOR_RED
        
        neo.fill_strip(*new_color) # Set all LEDs to the determined color
        neo.update_strip()         # Send the color data to the strip
        # ----------------------------------------

        # Annotate and display the video feed
        annotated = results[0].plot()
        label = stable_class if stable_class else "No Detection"
        cv2.putText(annotated, f"Detected: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 (Smoothed)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Cleanup ---
    print("\nCleaning up and turning off LEDs...")
    cam_proc.kill()
    cv2.destroyAllWindows()
    neo.clear_strip()      # Set all LEDs to (0,0,0)
    neo.update_strip()     # Push the "off" command
    print("Done.")

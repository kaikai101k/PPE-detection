from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
import time
from pi5neo import Pi5Neo # <-- Updated library

# Load trained model
model = YOLO("/home/pi/PPE_Detection/code/best.pt")

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 320
STABLE_THRESHOLD = 3

# Per-class thresholds
class_thresholds = {
    'goggles': 0.7,
    'eyeglasses': 0.25,
    'none': 0.3
}

# --- UPDATED LED Setup (pi5Neo) ---
NUM_PIXELS = 35
SPI_SPEED_KHZ = 800
SPI_DEVICE = "/dev/spidev0.0" # Assumes data wire is on GPIO 10 (Pin 19)

try:
    neo = Pi5Neo(SPI_DEVICE, NUM_PIXELS, SPI_SPEED_KHZ)
    print(f"pi5Neo strip initialized on {SPI_DEVICE} with {NUM_PIXELS} LEDs.")
except Exception as e:
    print(f"Failed to initialize pi5Neo: {e}")
    print("CRITICAL: Have you enabled SPI (sudo raspi-config)?")
    print("CRITICAL: Is your data wire on GPIO 10 (Pin 19)?")
    exit()

last_alert = None
# ---------------------------------

# Detection memory
last_detected_class = None
stable_class = None
stable_counter = 0

# --- Start camera (Updated to rpicam-vid) ---
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0", "--framerate", "15"
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

frame_id = 0

try:
    while True:
        # All frame reading, processing, and skipping logic is identical
        raw = cam_proc.stdout.read(FRAME_SIZE)
        if not raw:
            print("❌ Frame grab failed.")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8)
        if yuv.size != FRAME_SIZE:
            continue

        yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        frame_id += 1
        if frame_id % 2 != 0:
            continue

        # All prediction and manual filtering logic is identical
        results = model.predict(frame, imgsz=YOLO_IMGSZ, conf=0.01, device="cpu", verbose=False)
        filtered_boxes = []

        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                if label in class_thresholds and conf >= class_thresholds[label]:
                    filtered_boxes.append(box)

        results[0].boxes = filtered_boxes

        # --- UPDATED LED Logic (pi5Neo commands) ---
        labels = [model.names[int(box.cls)] for box in filtered_boxes]
        if any(lbl in ["none", "eyeglasses"] for lbl in labels):
            current_alert = "red"
        elif "goggles" in labels:
            current_alert = "green"
        else:
            current_alert = "off"

        if current_alert != last_alert:
            if current_alert == "red":
                neo.fill_strip(255, 0, 0) # REPLACED: pixels.fill(...)
            elif current_alert == "green":
                neo.fill_strip(0, 255, 0) # REPLACED: pixels.fill(...)
            elif current_alert == "off":
                neo.fill_strip(0, 0, 0)   # REPLACED: pixels.fill(...)
            # Removed the blank line here that likely caused the copy-paste error
            neo.update_strip() # REPLACED: pixels.show()
            last_alert = current_alert
        # ------------------------------------------

        # All temporal smoothing, annotation, and display logic is identical
        detected_class = labels[0] if labels else None
        if detected_class == last_detected_class and detected_class is not None:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_THRESHOLD:
            stable_class = detected_class
        last_detected_class = detected_class

        annotated = results[0].plot()
        label = stable_class if stable_class else "No Detection"
        cv2.putText(annotated, f"Detected: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 (LED + Threshold Adj)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam_proc.kill()
    cv2.destroyAllWindows()
    
    # --- UPDATED Cleanup ---
    neo.clear_strip()  # REPLACED: pixels.fill((0, 0, 0))
    neo.update_strip() # REPLACED: pixels.show()
    # ---------------------

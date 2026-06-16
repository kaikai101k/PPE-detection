# Import necessary libraries
from ultralytics import YOLO
import cv2
import subprocess
import numpy as np
from rpi_ws281x import PixelStrip, Color
import time

## ---------------- SETUP ---------------- ##

# 1. LED Strip Configuration
LED_COUNT = 35          # Number of LED pixels on your strip
LED_PIN = 18            # GPIO pin 18 (physical pin 12)
LED_FREQ_HZ = 800000    # LED signal frequency
LED_DMA = 10            # DMA channel
LED_BRIGHTNESS = 128    # Brightness (0-255)
LED_INVERT = False      # Set to True if signal is inverted

# Create and initialize the PixelStrip object
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
strip.begin()

# 2. Load trained YOLO model
model = YOLO("/home/pi/PPE_Detection/code/best.pt")

# 3. Camera and Detection Constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 320
CONFIDENCE_THRESHOLD = 0.5

# 4. Start the camera using the 'rpicam-vid' command
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0", "--framerate", "15"
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
print("? Camera process started. Press 'q' in the video window to exit.")

## ---------------- HELPER FUNCTIONS FOR LEDS ---------------- ##

def set_color(strip, color):
    """Fills the entire strip with a single color."""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()

## ---------------- MAIN LOOP ---------------- ##

try:
    frame_id = 0
    while True:
        # --- Frame Capture and Processing ---
        raw = cam_proc.stdout.read(FRAME_SIZE)
        if not raw:
            print("? Frame grab failed.")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        # Optimization: Process every other frame to maintain performance
        frame_id += 1
        if frame_id % 2 != 0:
            continue

        # --- Object Detection ---
        results = model.predict(frame, imgsz=YOLO_IMGSZ, conf=CONFIDENCE_THRESHOLD, device="cpu", verbose=False)
        detected_labels = [model.names[int(box.cls)] for box in results[0].boxes]

        # --- LED LOGIC (UPDATED AS REQUESTED) ---
        if 'goggles' in detected_labels:
            # If safety goggles are detected, set color to GREEN
            set_color(strip, Color(0, 255, 0))
        elif 'none' in detected_labels or 'eyeglasses' in detected_labels:
            # If regular glasses or nothing is detected, set color to RED
            set_color(strip, Color(255, 0, 0))
        else:
            # If no relevant objects are detected, turn the strip OFF
            set_color(strip, Color(0, 0, 0))

        # --- Display Output ---
        annotated_frame = results[0].plot()
        cv2.imshow("PPE Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed. Exiting...")
            break

finally:
    # --- Cleanup ---
    print("Cleaning up resources...")
    cam_proc.kill()
    cv2.destroyAllWindows()
    set_color(strip, Color(0, 0, 0)) # Ensure LEDs are off on exit
    print("? LED strip turned off. Exit successful.")

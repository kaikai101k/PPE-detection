from picamera2 import Picamera2
import time
from datetime import datetime
import os

# Directory to save images
save_dir = "/home/pi/new_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize camera
picam2 = Picamera2()
picam2.start_preview()  # Optional: shows the live preview window

# Configure still capture
picam2.configure(picam2.create_still_configuration())
picam2.start()

print("Starting image capture...")

try:
    for i in range(30):  # Capture 50 images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"image_{timestamp}.jpg")
        picam2.capture_file(filename)
        print(f"[{i+1}/50] Captured: {filename}")
        if i < 29:
            time.sleep(2)  # Wait 2 seconds (skip wait after last image)
except KeyboardInterrupt:
    print("\nCapture interrupted.")

picam2.stop_preview()
picam2.close()
print("Done.")

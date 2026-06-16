from ultralytics import YOLO
import cv2
import subprocess
import numpy as np

# Load trained model
model = YOLO("/home/pi/PPE_Detection/code/best.pt")

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 320
CONFIDENCE_THRESHOLD = 0.6
STABLE_THRESHOLD = 2  # number of consistent frames required to confirm class

# Detection memory
last_detected_class = None
stable_class = None
stable_counter = 0

# Start camera command - THE ONLY CHANGE IS HERE
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
            continue

        results = model.predict(frame, imgsz=YOLO_IMGSZ, conf=CONFIDENCE_THRESHOLD, device="cpu", verbose=False)
        boxes = results[0].boxes

        detected_class = None
        if boxes is not None and len(boxes) > 0:
            # Only use the most confident detection
            top_box = boxes[0]
            cls_id = int(top_box.cls)
            conf = float(top_box.conf)
            if conf >= CONFIDENCE_THRESHOLD:
                detected_class = results[0].names[cls_id]

        # Temporal smoothing logic
        if detected_class == last_detected_class and detected_class is not None:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_THRESHOLD:
            stable_class = detected_class
        last_detected_class = detected_class

        # Annotate output
        annotated = results[0].plot()
        label = stable_class if stable_class else "No Detection"
        cv2.putText(annotated, f"Detected: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 (Smoothed)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam_proc.kill()
    cv2.destroyAllWindows()

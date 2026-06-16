from ultralytics import YOLO
import subprocess
import cv2
import numpy as np
 
# Load model
model = YOLO('/home/pi/PPE_Detection/new_model_test/publicsh17.pt')

# Optimization constants
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
YOLO_IMGSZ = 640  # ← MATCH YOUR TRAINING SIZE (huge difference!)
CONFIDENCE_THRESHOLD = 0.5  # ← Lower to match MacBook default
FRAME_SKIP = 2 # Keep this for speed, but now with better detection

# Start camera
cam_cmd = [
    "rpicam-vid", "--inline", "--width", str(WIDTH), "--height", str(HEIGHT),
    "--codec", "yuv420", "-o", "-", "-t", "0", "--framerate", "15"
]
cam_proc = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_id = 0

""" model.model.names = {
    0: 'Person',
    1: 'Ears', 
    2: 'Helmet',
    3: 'Face',          # Changed from Glasses
    4: 'Face-mask-medical',
    5: 'Face-guard',
    6: 'Safety-vest',
    7: 'Earmuffs',
    8: 'Glasses',
    9: 'Gloves',
    10: 'Foot',
    11: 'Hands',        # Changed from Shoes
    12: 'Head',         # Changed from Safety-vest
    13: 'Tools',
    14: 'Shoes',
    15: 'Medical-suit',
    16: 'Safety-suit'
} """

try:
    while True:
        raw = cam_proc.stdout.read(FRAME_SIZE)
        if not raw:
            break
        
        yuv = np.frombuffer(raw, dtype=np.uint8)
        if yuv.size != FRAME_SIZE:
            continue
        
        yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        
        frame_id += 1
        
        if frame_id % FRAME_SKIP != 0:
            continue
        
        # Run inference with proper settings
        results = model.predict(
            frame, 
            imgsz=YOLO_IMGSZ,  # Now 640 - matches training
            conf=CONFIDENCE_THRESHOLD,  # Now 0.25 - matches MacBook
            iou=0.45,  # Add explicit IoU threshold
            device="cpu",
            verbose=False,
            half=False,
        )
        
        annotated = results[0].plot()
        cv2.imshow('YOLO Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam_proc.kill()
    cv2.destroyAllWindows()
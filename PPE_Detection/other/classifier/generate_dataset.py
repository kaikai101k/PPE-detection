# C:\eye_ppe_project\scripts\extract_crops_for_classifier.py

import os
import cv2
from ultralytics import YOLO

INPUT_ROOT = r"C:\yolo3"
OUTPUT_ROOT = r"C:\eye_ppe_project\dataset"
YOLO_MODEL_PATH = r"C:\eye_ppe_project\yolo_weights\best.pt"

model = YOLO(YOLO_MODEL_PATH)

# Class folders: eyeglasses, goggles, none
classes = ["eyeglasses", "goggles", "none"]

for label in classes:
    input_dir = os.path.join(INPUT_ROOT, label)
    output_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(output_dir, exist_ok=True)

    for i, fname in enumerate(os.listdir(input_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(input_dir, fname)
        image = cv2.imread(image_path)
        results = model(image)[0]

        if len(results.boxes) == 0:
            print(f"[WARN] No detection in: {fname}")
            continue

        for j, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_path = os.path.join(output_dir, f"{label}_{i}_{j}.jpg")
            cv2.imwrite(out_path, crop)
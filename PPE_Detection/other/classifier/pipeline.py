# pipeline.py

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from collections import deque
from torchvision import transforms, models
from ultralytics import YOLO
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# === Paths ===
YOLO_MODEL_PATH = "/home/pi/Machine Vision/ppe_project/yolo_weights/best.pt"
CLASSIFIER_PATH = "/home/pi/Machine Vision/ppe_project/classifier/eye_classifier.pt"

print("[INFO] Loading models...")

# === Load YOLO ===
yolo_model = YOLO(YOLO_MODEL_PATH)
print("[INFO] YOLO model loaded.")

# === Load Classifier ===
classifier = models.resnet18(weights=None)
classifier.fc = torch.nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location='cpu'))
classifier.eval()
print("[INFO] Classifier model loaded.")

# === Labels and Transforms ===
class_labels = ["eyeglasses", "goggles", "none"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

recent_preds = deque(maxlen=5)

def smoothed_prediction(pred):
    recent_preds.append(pred)
    return max(set(recent_preds), key=recent_preds.count)

def classify_crop(crop_img):
    if crop_img.size == 0:
        return "unsure"
    img = transform(Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        out = classifier(img)
        probs = F.softmax(out, dim=1).squeeze()
    top2 = torch.topk(probs, 2)
    margin = top2.values[0] - top2.values[1]
    print(f"[DEBUG] Probs → none: {probs[2]:.2f}, eyeglasses: {probs[0]:.2f}, goggles: {probs[1]:.2f}")
    if top2.values[0] < 0.6 or margin < 0.15:
        return "unsure"
    return class_labels[top2.indices[0]]

# === Start Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()
else:
    print("✅ Webcam initialized.")

print("[INFO] Starting inference loop...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    results = yolo_model(frame)[0]
    print(f"[INFO] {len(results.boxes)} detection(s) found.")

    unsafe_detected = False

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        crop = frame[y1:y2, x1:x2]
        raw_pred = classify_crop(crop)
        final_pred = smoothed_prediction(raw_pred)
        color = (0, 255, 0) if final_pred == "goggles" else (0, 0, 255)
        if final_pred in ["none", "eyeglasses"]:
            unsafe_detected = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, final_pred, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if unsafe_detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
        cv2.putText(frame, "⚠ SAFETY ALERT: Wear Proper Goggles!",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("🛡 Eye PPE Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Inference ended.")

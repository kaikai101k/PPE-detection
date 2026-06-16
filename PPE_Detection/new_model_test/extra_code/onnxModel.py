import onnxruntime as ort
import numpy as np
import cv2

# Path to model
model_path = '/home/pi/PPE_Detection/new_model_test/model.onnx'  # UPDATE THIS

print("Loading ONNX model...")
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Print model info
print("\n=== Model Inputs ===")
for inp in session.get_inputs():
    print(f"Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

print("\n=== Model Outputs ===")
for out in session.get_outputs():
    print(f"Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

# Common PPE classes (we'll adjust once we see detections)
class_names = [
    'person', 'head', 'helmet', 'face', 'vest', 
    'gloves', 'goggles', 'mask', 'no-helmet', 'no-vest'
]

input_name = session.get_inputs()[0].name
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.4

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStarting camera... Press 'q' to quit")
print("Look at console output to see detection info\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        if frame_count % 3 == 0:  # Process every 3rd frame
            # Preprocess
            input_tensor = preprocess(frame)
            
            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            
            # Print output info on first run
            if frame_count == 3:
                print(f"\n=== First Inference Output ===")
                print(f"Number of outputs: {len(outputs)}")
                for i, out in enumerate(outputs):
                    print(f"Output {i} shape: {out.shape}")
                    print(f"Output {i} sample values: {out.flatten()[:10]}")
                print("\nNow processing detections...\n")
            
            # Try to parse detections (adjust based on output)
            preds = outputs[0]
            
            # Handle different formats
            if len(preds.shape) == 3:
                preds = preds[0]
            
            # Transpose if needed
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T
            
            # Basic filtering
            if preds.shape[1] >= 5:
                boxes = preds[:, :4]
                confs = preds[:, 4]
                
                for i, conf in enumerate(confs):
                    if conf > CONFIDENCE_THRESHOLD:
                        x, y, w_box, h_box = boxes[i]
                        
                        # Convert to pixel coords
                        x1 = int((x - w_box/2) * w)
                        y1 = int((y - h_box/2) * h)
                        x2 = int((x + w_box/2) * w)
                        y2 = int((y + h_box/2) * h)
                        
                        # Get class if available
                        if preds.shape[1] > 5:
                            class_scores = preds[i, 5:]
                            class_id = int(np.argmax(class_scores))
                            class_conf = class_scores[class_id]
                            
                            if class_id < len(class_names):
                                label = f"{class_names[class_id]}: {conf*class_conf:.2f}"
                            else:
                                label = f"Class {class_id}: {conf*class_conf:.2f}"
                        else:
                            label = f"Detection: {conf:.2f}"
                        
                        # Draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Print detection info
                        if frame_count % 30 == 0:  # Print every 10 processed frames
                            print(f"Detection: {label} at ({x1},{y1})-({x2},{y2})")
        
        cv2.imshow('PPE Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
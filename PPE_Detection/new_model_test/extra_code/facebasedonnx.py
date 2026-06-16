#!/usr/bin/env python3
"""
Safety Glasses Detection using YOLOv8 ONNX model on Raspberry Pi 5
Detects faces and glasses to determine if people are wearing safety glasses
"""

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
from collections import deque

# Model configuration
MODEL_ONNX = "/home/pi/PPE_Detection/new_model_test/publicsh17.onnx"
CUSTOM_NAMES = {
    0: 'Person',
    1: 'Head',
    2: 'Face',
    3: 'Glasses',
    4: 'Face-mask-medical',
    5: 'Face-guard',
    6: 'Ear',
    7: 'Earmuffs',
    8: 'Hands',
    9: 'Gloves',
    10: 'Foot',
    11: 'Shoes',
    12: 'Safety-vest',
    13: 'Tools',
    14: 'Helmet',
    15: 'Medical-suit',
    16: 'Safety-suit'
}

# Detection parameters
FACE_CLASS_ID = 2
GLASSES_CLASS_ID = 3
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
INPUT_SIZE = 512

# Frame window for temporal detection (in frames)
FRAME_WINDOW = 1

class SafetyGlassesDetector:
    def __init__(self):
        # Initialize ONNX Runtime
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            MODEL_ONNX,
            providers=['CPUExecutionProvider']  # Use CPU for Pi
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Initialize PiCamera2
        print("Initializing camera...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # Detection history for temporal analysis
        self.face_history = deque(maxlen=FRAME_WINDOW)
        self.glasses_history = deque(maxlen=FRAME_WINDOW)
        
    def preprocess_image(self, image):
        """Preprocess image for YOLO ONNX model"""
        # Resize to model input size
        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        
        # Convert BGR to RGB if needed (PiCamera2 already gives RGB)
        # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to NCHW format
        input_image = resized.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image
    
    def postprocess_predictions(self, outputs, original_shape):
        """Process YOLO output to get bounding boxes"""
        predictions = outputs[0]  # Get the output tensor
        
        # YOLOv8 output format: [1, 84, 8400] or similar
        # Where 84 = 4 (box) + 80 (classes) for COCO, adjust for your model
        # Your model has 17 classes, so it should be 4 + 17 = 21
        
        if len(predictions.shape) == 3:
            predictions = np.squeeze(predictions, axis=0)  # Remove batch dimension
        
        # Transpose if needed to get [num_predictions, num_features]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        # Extract boxes and class scores
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # Class scores
        
        # Get max scores and class IDs
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidences > CONF_THRESHOLD
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert from xywh to xyxy format
        if len(boxes) > 0:
            # Scale boxes back to original image size
            scale_x = original_shape[1] / INPUT_SIZE
            scale_y = original_shape[0] / INPUT_SIZE
            
            # Convert center format to corner format
            x_center = boxes[:, 0] * scale_x
            y_center = boxes[:, 1] * scale_y
            width = boxes[:, 2] * scale_x
            height = boxes[:, 3] * scale_y
            
            boxes[:, 0] = x_center - width / 2  # x1
            boxes[:, 1] = y_center - height / 2  # y1
            boxes[:, 2] = x_center + width / 2   # x2
            boxes[:, 3] = y_center + height / 2  # y2
        
        return boxes, confidences, class_ids
    
    def apply_nms(self, boxes, confidences, class_ids):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return [], [], []
        
        # Convert boxes to integer coordinates for OpenCV NMS
        boxes_int = boxes.astype(np.int32)
        boxes_cv = []
        for box in boxes_int:
            x1, y1, x2, y2 = box
            boxes_cv.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x,y,w,h for OpenCV
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_cv, 
            confidences.tolist(), 
            CONF_THRESHOLD, 
            IOU_THRESHOLD
        )
        
        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], confidences[indices], class_ids[indices]
        
        return [], [], []
    
    def detect_objects(self, image):
        """Run detection on a single frame"""
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Postprocess predictions
        boxes, confidences, class_ids = self.postprocess_predictions(outputs, image.shape)
        
        # Apply NMS
        boxes, confidences, class_ids = self.apply_nms(boxes, confidences, class_ids)
        
        return boxes, confidences, class_ids, inference_time
    
    def analyze_safety_compliance(self, class_ids):
        """Analyze if faces have corresponding glasses"""
        face_count = np.sum(class_ids == FACE_CLASS_ID)
        glasses_count = np.sum(class_ids == GLASSES_CLASS_ID)
        
        # Update history
        self.face_history.append(face_count)
        self.glasses_history.append(glasses_count)
        
        # Check temporal window
        avg_faces = np.mean(self.face_history) if self.face_history else 0
        avg_glasses = np.mean(self.glasses_history) if self.glasses_history else 0
        
        # Determine compliance
        if avg_faces > 0:
            if avg_glasses >= avg_faces * 0.8:  # 80% threshold for compliance
                status = "COMPLIANT - Safety glasses detected"
                color = (0, 255, 0)  # Green
            else:
                status = "WARNING - Safety glasses may be missing"
                color = (0, 0, 255)  # Red
        else:
            status = "No faces detected"
            color = (128, 128, 128)  # Gray
        
        return face_count, glasses_count, status, color
    
    def run(self):
        """Main detection loop"""
        print("\n=== Safety Glasses Detection Started ===")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Run detection
                boxes, confidences, class_ids, inference_time = self.detect_objects(frame)
                
                # Analyze safety compliance
                face_count, glasses_count, status, color = self.analyze_safety_compliance(class_ids)
                
                # Print console output
                if frame_count % 10 == 0:  # Print every 10 frames to avoid spam
                    print(f"Frame {frame_count:4d} | Faces: {face_count} | Glasses: {glasses_count} | "
                          f"Inference: {inference_time:.1f}ms | Status: {status}")
                
                # Add status text to frame
                cv2.putText(frame, f"Faces: {face_count} | Glasses: {glasses_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, status, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"FPS: {1000/inference_time:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Safety Glasses Detection", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n\nDetection stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    detector = SafetyGlassesDetector()
    detector.run()

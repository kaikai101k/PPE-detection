from ultralytics import YOLO

# Path to your model
MODEL_PT = "/home/pi/PPE_Detection/new_model_test/best_spring202511s.pt"


# Load the model
model = YOLO(MODEL_PT)

# Export to ONNX with specific image size
# imgsz=(512, 512) sets the input dimensions
# format='onnx' specifies ONNX format
model.export(
    format='onnx',
    imgsz=512,
    opset=17 # Single value for square images
)

print(f"Model exported successfully!")
print(f"ONNX model saved as: {MODEL_PT.replace('.pt', '.onnx')}")

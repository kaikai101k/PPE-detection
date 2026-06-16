import cv2
import numpy as np
import onnxruntime as ort
import torch
import time
import os
import sys
from collections import deque, defaultdict
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression
from time import sleep
from pi5neo import Pi5Neo

# Suppress ONNX Runtime warnings about GPU (Raspberry Pi doesn't have one)
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Only show errors


# --- CONFIG FOR LED STRIP---
NUM_LEDS = 30           # adjust to your strip length
SPI_DEVICE = "/dev/spidev0.0"
SPI_SPEED_KHZ = 800     # default is fine
neo = Pi5Neo(SPI_DEVICE, NUM_LEDS, SPI_SPEED_KHZ)
print("Initialized Pi5Neo LED strip")


# Try importing picamera2 for Pi Camera support
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# Print versions for debugging
print("\n" + "="*60)
print("PPE Detection System - Starting Up")
print("="*60)
print(f"Python: {sys.version.split()[0]}")
print(f"OpenCV: {cv2.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"PiCamera2: {'Available ✓' if PICAMERA2_AVAILABLE else 'Not installed (USB camera mode)'}")
print("="*60 + "\n")

# ------------------------------
# DEVICE CONFIGURATION
# ------------------------------
# ⚠️ IMPORTANT: Set to True when running on Raspberry Pi! ⚠️
RASPBERRY_PI_MODE = True  # Change to True for Raspberry Pi

# ========== CONSOLE OUTPUT MODES ==========
# Choose your preferred mode based on use case:

# Mode 1: HEADLESS + VERBOSE (Recommended for Pi)
#   - No display window (saves compute)
#   - Prints detailed status every N frames
#   - Shows all detection stats
HEADLESS_MODE = False
VERBOSE_OUTPUT = True
PRINT_EVERY_N_FRAMES = 30

# Mode 2: HEADLESS + VIOLATIONS ONLY (Minimal output)
#   - No display window
#   - Only prints when violations detected
#   - Best for logging/alerts only
# HEADLESS_MODE = True
# VERBOSE_OUTPUT = False
# PRINT_EVERY_N_FRAMES = 30

# Mode 3: DISPLAY MODE (For desktop/testing)
#   - Shows visual window with bounding boxes
#   - Good for debugging and demos
# HEADLESS_MODE = False
# VERBOSE_OUTPUT = True
# PRINT_EVERY_N_FRAMES = 30

# Performance settings
if RASPBERRY_PI_MODE:
    # Raspberry Pi optimizations
    img_size = 512  # Smaller resolution for faster inference (vs 640)
    conf_thres = 0.3  # Slightly higher confidence threshold
    iou_thres = 0.5
    CAMERA_WIDTH = 640  # Lower camera resolution
    CAMERA_HEIGHT = 480
    SKIP_FRAMES = 0  # Process every frame (set to 1 to skip every other frame)
else:
    # Desktop/MacBook settings
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    SKIP_FRAMES = 0

# ONNX Runtime optimization for Raspberry Pi
onnx_providers = ["CPUExecutionProvider"]
if RASPBERRY_PI_MODE:
    # Enable threading for better CPU utilization on Pi
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Use 4 cores on Pi 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
else:
    sess_options = ort.SessionOptions()

# ------------------------------
# MODEL SETTINGS
# ------------------------------
# ⚠️ UPDATE THIS PATH FOR YOUR SYSTEM! ⚠️
onnx_model_path = "/home/pi/PPE_Detection/new_model_test/best_spring202511s.onnx"  # Update for Pi path
# Example Raspberry Pi path: "/home/pi/models/best_spring202511s.onnx"

# Check if model file exists before proceeding
if not os.path.exists(onnx_model_path):
    print("\n" + "="*60)
    print("❌ ERROR: Model file not found!")
    print("="*60)
    print(f"Looking for: {onnx_model_path}")
    print("\nPlease update the 'onnx_model_path' variable with the correct path.")
    print("On Raspberry Pi, it might be something like:")
    print("  /home/pi/models/best_spring202511s.onnx")
    print("="*60 + "\n")
    exit(1)

class_names = [
    "Person", "Ear", "Shoes", "Face", "Face-mask-medical", "Face-guard", 
    "Safety-vest", "Earmuffs", "Glasses", "Gloves", "Foot", "Hands", 
    "Head", "Tools", "Helmet", "Medical-suit", "Safety-suit"
]

FACE_CLASS = 3
GLASSES_CLASS = 8

# ------------------------------
# FPS-ADAPTIVE THRESHOLDS
# ------------------------------
class AdaptiveThresholds:
    """Automatically adjust thresholds based on actual FPS"""
    
    def __init__(self):
        self.measured_fps = 30
        self.fps_history = deque(maxlen=30)
        
        # Time-based thresholds (in seconds)
        self.glasses_detection_time = 0.3  # 0.3 seconds to confirm glasses
        self.no_glasses_time = 0.75  # 0.75 seconds without glasses = violation
        self.history_time = 1.5  # Keep 1.5 seconds of history
        
    def update_fps(self, fps):
        """Update measured FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) >= 10:  # Use average of last 10 measurements
            self.measured_fps = sum(list(self.fps_history)[-10:]) / 10
    
    def get_glasses_threshold(self):
        """Get glasses detection threshold scaled for current FPS"""
        # At 6 FPS: 0.3s * 6 = 1.8 ≈ 2 frames
        # At 10 FPS: 0.3s * 10 = 3 frames
        # At 30 FPS: 0.3s * 30 = 9 frames
        scaled = max(1, int(self.glasses_detection_time * self.measured_fps))
        return scaled
    
    def get_no_glasses_threshold(self):
        """Get no-glasses threshold scaled for current FPS"""
        # At 6 FPS: 0.75s * 6 = 4.5 ≈ 5 frames
        # At 10 FPS: 0.75s * 10 = 7.5 ≈ 8 frames
        # At 30 FPS: 0.75s * 30 = 22.5 ≈ 23 frames
        scaled = max(3, int(self.no_glasses_time * self.measured_fps))
        return scaled
    
    def get_history_length(self):
        """Get history buffer length scaled for current FPS"""
        # Always keep history_time seconds of history
        # At 6 FPS: 1.5s * 6 = 9 frames
        # At 30 FPS: 1.5s * 30 = 45 frames
        scaled = max(10, int(self.history_time * self.measured_fps))
        return scaled
    
    def get_time_settings(self):
        """Return the time-based settings for display"""
        return {
            "glasses_time": self.glasses_detection_time,
            "no_glasses_time": self.no_glasses_time,
            "history_time": self.history_time
        }

# ------------------------------
# LOAD ONNX MODEL
# ------------------------------
print("Loading ONNX model...")
try:
    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=sess_options,
        providers=onnx_providers
    )
    input_name = session.get_inputs()[0].name
    print(f"✓ Model loaded successfully")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {session.get_inputs()[0].shape}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at: {onnx_model_path}")
    print("Please update the onnx_model_path variable with the correct path")
    exit(1)
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# ------------------------------
# CAMERA WRAPPER CLASS
# ------------------------------
class Camera:
    """Unified camera interface for both Pi Camera and USB cameras"""
    
    def __init__(self, width=640, height=480, use_picamera=True):
        self.camera = None
        self.camera_type = None
        self.width = width
        self.height = height
        
        # Try Pi Camera first if requested and available
        if use_picamera and PICAMERA2_AVAILABLE:
            try:
                print("🔍 Attempting to open Pi Camera...")
                self.camera = Picamera2()
                
                # Configure for video capture (better performance than still)
                config = self.camera.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"},
                    buffer_count=2  # Reduce buffer for lower latency
                )
                self.camera.configure(config)
                self.camera.start()
                
                # Wait for camera to warm up
                time.sleep(1)
                
                self.camera_type = "picamera"
                print(f"✓ Pi Camera opened successfully at {width}x{height}")
                return
            except Exception as e:
                print(f"⚠ Pi Camera failed: {e}")
                if self.camera:
                    try:
                        self.camera.stop()
                    except:
                        pass
                self.camera = None
        
        # Fall back to USB camera
        print("🔍 Attempting to open USB camera...")
        for idx in [0, 1, -1]:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.camera = cap
                        self.camera_type = "usb"
                        print(f"✓ USB camera opened at index {idx} ({width}x{height})")
                        return
                    cap.release()
            except Exception as e:
                print(f"  Index {idx} failed: {e}")
                continue
        
        raise RuntimeError("❌ No camera found! Check connections.")
    
    def read(self):
        """Read a frame from the camera (returns ret, frame like cv2.VideoCapture)"""
        if self.camera_type == "picamera":
            try:
                frame = self.camera.capture_array()
                # Pi Camera returns RGB, OpenCV expects BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception as e:
                print(f"Pi Camera read error: {e}")
                return False, None
        elif self.camera_type == "usb":
            return self.camera.read()
        return False, None
    
    def release(self):
        """Release the camera"""
        if self.camera_type == "picamera":
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
        elif self.camera_type == "usb":
            self.camera.release()
        print("Camera released")

# ------------------------------
# FPS COUNTER
# ------------------------------
class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

# ------------------------------
# TRACKING CLASS
# ------------------------------
class FaceTracker:
    """Simple IoU-based tracker for faces across frames"""
    
    def __init__(self, max_lost_frames=15, tracking_iou_threshold=0.3):
        self.next_id = 0
        self.tracked_faces = {}
        self.max_lost_frames = max_lost_frames
        self.tracking_iou_threshold = tracking_iou_threshold
    
    def update(self, current_detections):
        if len(current_detections) == 0:
            for track_id in list(self.tracked_faces.keys()):
                self.tracked_faces[track_id]["lost_frames"] += 1
                if self.tracked_faces[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracked_faces[track_id]
            return []
        
        tracked_results = []
        matched_track_ids = set()
        
        for detection in current_detections:
            det_box = detection[:4]
            best_iou = 0
            best_track_id = None
            
            for track_id, track_data in self.tracked_faces.items():
                if track_id in matched_track_ids:
                    continue
                    
                track_box = track_data["box"]
                iou = self._calculate_iou(det_box, track_box)
                
                if iou > self.tracking_iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                self.tracked_faces[best_track_id]["box"] = det_box
                self.tracked_faces[best_track_id]["lost_frames"] = 0
                tracked_results.append((best_track_id, det_box))
                matched_track_ids.add(best_track_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracked_faces[new_id] = {"box": det_box, "lost_frames": 0}
                tracked_results.append((new_id, det_box))
        
        for track_id in list(self.tracked_faces.keys()):
            if track_id not in matched_track_ids:
                self.tracked_faces[track_id]["lost_frames"] += 1
                if self.tracked_faces[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracked_faces[track_id]
        
        return tracked_results
    
    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# ------------------------------
# COMPLIANCE TRACKER
# ------------------------------
class ComplianceTracker:
    """Track compliance with adaptive thresholds"""
    
    def __init__(self, adaptive_thresholds):
        self.face_histories = defaultdict(lambda: deque(maxlen=100))  # Max size, will be trimmed
        self.adaptive_thresholds = adaptive_thresholds
    
    def update(self, track_id, wearing_glasses):
        history_length = self.adaptive_thresholds.get_history_length()
        # Trim history to current length
        while len(self.face_histories[track_id]) > history_length:
            self.face_histories[track_id].popleft()
        self.face_histories[track_id].append(wearing_glasses)
    
    def get_compliance_status(self, track_id):
        if track_id not in self.face_histories or len(self.face_histories[track_id]) == 0:
            return False, "unknown"
        
        history = self.face_histories[track_id]
        glasses_count = sum(history)
        no_glasses_count = len(history) - glasses_count
        
        glasses_threshold = self.adaptive_thresholds.get_glasses_threshold()
        no_glasses_threshold = self.adaptive_thresholds.get_no_glasses_threshold()
        
        # LENIENT: If glasses detected in threshold frames -> COMPLIANT
        if glasses_count >= glasses_threshold:
            confidence = "high" if glasses_count >= glasses_threshold * 3 else "medium"
            return True, confidence
        
        # STRICT: Only non-compliant with enough evidence
        if len(history) >= no_glasses_threshold and no_glasses_count >= no_glasses_threshold:
            return False, "high"
        
        return True, "low"
    
    def cleanup_old_tracks(self, active_track_ids):
        all_track_ids = list(self.face_histories.keys())
        for track_id in all_track_ids:
            if track_id not in active_track_ids:
                del self.face_histories[track_id]

# ------------------------------
# SPATIAL FUNCTIONS
# ------------------------------
def get_box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def is_wearing_glasses(face_box, glasses_box, iou_threshold=0.1):
    face_x1, face_y1, face_x2, face_y2 = face_box
    glasses_center = get_box_center(glasses_box)
    
    if not point_in_box(glasses_center, face_box):
        return False
    
    face_height = face_y2 - face_y1
    glasses_cy = glasses_center[1]
    upper_face_limit = face_y1 + (face_height * 0.7)
    
    if glasses_cy > upper_face_limit:
        return False
    
    iou = calculate_iou(face_box, glasses_box)
    if iou < iou_threshold:
        return False
    
    return True

# ------------------------------
# PREPROCESSING
# ------------------------------
def preprocess(frame):
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------------
# CONSOLE OUTPUT FUNCTIONS
# ------------------------------
def print_detection_summary(frame_counter, inference_time, fps, tracked_faces, compliance_tracker, adaptive_thresholds):
    """Print detection results to console"""
    from datetime import datetime
    
    # Calculate statistics
    total_faces = len(tracked_faces)
    compliant_count = 0
    checking_count = 0
    non_compliant_faces = []
    
    for track_id, face_box in tracked_faces:
        is_compliant, confidence = compliance_tracker.get_compliance_status(track_id)
        if is_compliant:
            if confidence in ["high", "medium"]:
                compliant_count += 1
            else:
                checking_count += 1
        else:
            non_compliant_faces.append((track_id, confidence))
    
    non_compliant_count = len(non_compliant_faces)
    
    # Print timestamp and frame info
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Frame {frame_counter:05d} | Inference: {inference_time*1000:.0f}ms | FPS: {fps:.1f}")
    
    # Print detection statistics
    print(f"  Faces: {total_faces} | ✓ Compliant: {compliant_count} | ? Checking: {checking_count} | ✗ Non-compliant: {non_compliant_count}")
    
    # Print adaptive thresholds with time info
    time_settings = adaptive_thresholds.get_time_settings()
    print(f"  Thresholds: Glass={time_settings['glasses_time']}s ({adaptive_thresholds.get_glasses_threshold()}f) | " \
          f"NoGlass={time_settings['no_glasses_time']}s ({adaptive_thresholds.get_no_glasses_threshold()}f)")
    
    # Print non-compliant faces details
    if non_compliant_count > 0:
        print(f"  ⚠️  NON-COMPLIANT IDs: {[f'ID{track_id}' for track_id, _ in non_compliant_faces]}")

def print_violation_alert(track_id, frame_counter):
    """Print alert when a face becomes non-compliant"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n{'='*60}")
    print(f"🚨 VIOLATION DETECTED - Frame {frame_counter}")
    print(f"   Time: {timestamp}")
    print(f"   Face ID: {track_id}")
    print(f"   Status: NOT WEARING SAFETY GLASSES")
    print(f"{'='*60}")
    

# ------------------------------
# VISUALIZATION (OPTIONAL - ONLY IF NOT HEADLESS)
# ------------------------------
def draw_detections(frame, tracked_faces, glasses_list, compliance_tracker):
    for track_id, face_box in tracked_faces:
        x1, y1, x2, y2 = map(int, face_box)
        
        is_compliant, confidence = compliance_tracker.get_compliance_status(track_id)
        
        if is_compliant:
            if confidence == "high":
                color = (0, 255, 0)
                status = "✓ COMPLIANT"
            elif confidence == "medium":
                color = (0, 200, 100)
                status = "✓ Likely"
            else:
                color = (0, 150, 150)
                status = "? Check"
        else:
            color = (0, 0, 255)
            status = "✗ NO PPE"
        
        thickness = 3 if confidence == "high" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"ID:{track_id} {status}"
        font_scale = 0.5 if RASPBERRY_PI_MODE else 0.6
        cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    # Draw glasses (subtle)
    for glasses in glasses_list:
        x1, y1, x2, y2 = map(int, glasses[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

def draw_summary(frame, fps_counter, adaptive_thresholds, total_faces, compliant_count, checking_count):
    non_compliant = total_faces - compliant_count - checking_count
    current_fps = fps_counter.fps
    
    # Line 1: Stats
    stats = f"Faces:{total_faces} OK:{compliant_count} Check:{checking_count} FAIL:{non_compliant}"
    # Line 2: Performance
    perf = f"FPS:{current_fps:.1f} | Glass:{adaptive_thresholds.get_glasses_threshold()}f NoGlass:{adaptive_thresholds.get_no_glasses_threshold()}f"
    
    font_scale = 0.5 if RASPBERRY_PI_MODE else 0.6
    
    cv2.rectangle(frame, (5, 5), (750, 55), (0, 0, 0), -1)
    cv2.putText(frame, stats, (10, 22), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    cv2.putText(frame, perf, (10, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 200, 255), 1)

# ------------------------------
# MAIN LOOP
# ------------------------------
print("\n" + "="*60)
print(f"🎥 Face-Glasses Detection {'[RASPBERRY PI MODE]' if RASPBERRY_PI_MODE else '[DESKTOP MODE]'}")
print("="*60)

# Initialize camera
try:
    use_picamera = RASPBERRY_PI_MODE and PICAMERA2_AVAILABLE
    cap = Camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, use_picamera=use_picamera)
    print(f"✓ Camera type: {cap.camera_type}")
except Exception as e:
    print(f"\n❌ Camera initialization failed: {e}")
    print("\nTroubleshooting tips:")
    if RASPBERRY_PI_MODE:
        print("For Pi Camera:")
        print("  1. Enable camera: sudo raspi-config → Interface Options → Camera")
        print("  2. Check connection: libcamera-hello --list-cameras")
        print("  3. Install picamera2: sudo apt install -y python3-picamera2")
    print("For USB camera:")
    print("  1. Check if connected: ls /dev/video*")
    print("  2. Try: v4l2-ctl --list-devices")
    exit(1)

# Initialize
adaptive_thresholds = AdaptiveThresholds()
face_tracker = FaceTracker()
compliance_tracker = ComplianceTracker(adaptive_thresholds)
fps_counter = FPSCounter()

print(f"📊 Initial Settings:")
print(f"   - Mode: {'HEADLESS (Console Only)' if HEADLESS_MODE else 'Display Mode (GUI)'}")
print(f"   - Output: {'Verbose' if VERBOSE_OUTPUT else 'Violations Only'}")
if VERBOSE_OUTPUT:
    print(f"   - Print interval: Every {PRINT_EVERY_N_FRAMES} frames")
print(f"   - Image size: {img_size}x{img_size}")
print(f"   - Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
time_settings = adaptive_thresholds.get_time_settings()
print(f"   - Glasses detection: {time_settings['glasses_time']}s (~{adaptive_thresholds.get_glasses_threshold()} frames)")
print(f"   - Violation threshold: {time_settings['no_glasses_time']}s (~{adaptive_thresholds.get_no_glasses_threshold()} frames)")
print(f"   - History buffer: {time_settings['history_time']}s (~{adaptive_thresholds.get_history_length()} frames)")
print(f"   - Skip frames: {SKIP_FRAMES}")
print("   - Frame thresholds will auto-adjust based on actual FPS")
print("\nPress Ctrl+C to stop\n")

frame_counter = 0
inference_count = 0
last_compliance_status = {}  # Track previous compliance status for change detection

print("\n🚀 Starting detection loop...")
if HEADLESS_MODE:
    print("   📺 Headless mode: No display window (optimized for performance)")
    print("   Press Ctrl+C to stop")
else:
    print("   📺 Display mode: Window will open")
    print("   Press 'q' to quit, 'd' for debug info")
print()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Warning: Failed to read frame from camera")
            break
        
        frame_counter += 1
        
        # Skip frames if configured
        if SKIP_FRAMES > 0 and frame_counter % (SKIP_FRAMES + 1) != 0:
            continue
        
        # Show progress on first few frames
        if frame_counter <= 3:
            print(f"Processing frame {frame_counter}...")
        
        # Measure FPS
        current_fps = fps_counter.update()
        adaptive_thresholds.update_fps(current_fps)

        original_shape = frame.shape[:2]
        
        # Preprocessing and inference
        try:
            inference_start = time.time()
            input_tensor = preprocess(frame)
            outputs = session.run(None, {input_name: input_tensor})[0]
            outputs = torch.from_numpy(outputs)
            results = non_max_suppression(outputs, conf_thres, iou_thres)
            inference_time = time.time() - inference_start
            
            inference_count += 1
            if inference_count == 1:
                print(f"✓ First inference successful! ({inference_time*1000:.0f}ms)")
                print(f"  Expected FPS: ~{1.0/inference_time:.1f}\n")
        
        except Exception as e:
            print(f"❌ Inference error: {e}")
            import traceback
            traceback.print_exc()
            break

        det = results[0]
        
        if len(det):
            det[:, :4] = scale_boxes((img_size, img_size), det[:, :4], original_shape)
            
            faces = []
            glasses_list = []
            
            for *xyxy, conf, cls in det:
                cls_idx = int(cls)
                if cls_idx == FACE_CLASS:
                    faces.append([*xyxy, conf])
                elif cls_idx == GLASSES_CLASS:
                    glasses_list.append([*xyxy, conf])
            
            tracked_faces = face_tracker.update(faces)
            
            for track_id, face_box in tracked_faces:
                wearing_glasses_now = False
                
                for glasses in glasses_list:
                    glasses_box = glasses[:4]
                    if is_wearing_glasses(face_box, glasses_box):
                        wearing_glasses_now = True
                        break
                
                compliance_tracker.update(track_id, wearing_glasses_now)
                
                # Check for compliance status changes (for violation alerts)
                is_compliant, confidence = compliance_tracker.get_compliance_status(track_id)
                prev_status = last_compliance_status.get(track_id, (True, "unknown"))
                
                # Alert on new violations (was compliant, now non-compliant)
                if not is_compliant and prev_status[0] and confidence == "high":
                    print_violation_alert(track_id, frame_counter)
                
                if not is_compliant and confidence == "high":
                    neo.fill_strip(255,0,0)
                    neo.update_strip
                else:
                    neo.fill_strip(0,0,0)
                    neo.update_strip
                
                last_compliance_status[track_id] = (is_compliant, confidence)
            
            active_ids = {track_id for track_id, _ in tracked_faces}
            compliance_tracker.cleanup_old_tracks(active_ids)
            
            # Remove old IDs from compliance tracking
            for track_id in list(last_compliance_status.keys()):
                if track_id not in active_ids:
                    del last_compliance_status[track_id]
            
            # Print status at intervals or if verbose
            should_print = (
                VERBOSE_OUTPUT and (frame_counter % PRINT_EVERY_N_FRAMES == 0 or len(tracked_faces) > 0)
            ) or (not VERBOSE_OUTPUT and len(tracked_faces) > 0)
            
            if should_print:
                print_detection_summary(
                    frame_counter, inference_time, current_fps, 
                    tracked_faces, compliance_tracker, adaptive_thresholds
                )
            
            # Optional display (only if not headless)
            if not HEADLESS_MODE:
                compliant_count = 0
                checking_count = 0
                for track_id, _ in tracked_faces:
                    is_compliant, confidence = compliance_tracker.get_compliance_status(track_id)
                    if is_compliant:
                        if confidence in ["high", "medium"]:
                            compliant_count += 1
                        else:
                            checking_count += 1
                
                draw_detections(frame, tracked_faces, glasses_list, compliance_tracker)
                draw_summary(frame, fps_counter, adaptive_thresholds, len(tracked_faces), compliant_count, checking_count)
                
                cv2.imshow("PPE Compliance Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 User pressed 'q' - exiting...")
                    break
                elif key == ord('d'):
                    # Toggle debug info
                    print(f"\nDebug Info - Frame {frame_counter}:")
                    print(f"  FPS: {current_fps:.1f}")
                    print(f"  Inference time: {inference_time*1000:.0f}ms")
                    print(f"  Faces detected: {len(faces)}")
                    print(f"  Glasses detected: {len(glasses_list)}")
        else:
            # No detections - still update display if not headless
            if not HEADLESS_MODE:
                draw_summary(frame, fps_counter, adaptive_thresholds, 0, 0, 0)
                cv2.imshow("PPE Compliance Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

except cv2.error as e:
    if not HEADLESS_MODE:
        print(f"\n❌ OpenCV Display Error: {e}")
        print("\nPossible causes:")
        print("1. No display available (running headless?)")
        print("2. X11 forwarding not enabled")
        print("3. Need to run: export DISPLAY=:0")
        print("\nSet HEADLESS_MODE = True for headless operation")

except KeyboardInterrupt:
    print("\n\n⚠ Interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    if not HEADLESS_MODE:
        cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"Total frames processed: {frame_counter}")
    print(f"Total inferences: {inference_count}")
    print(f"Average FPS: {fps_counter.fps:.1f}")
    if inference_count > 0:
        print(f"Final expected FPS: ~{1.0/inference_time:.1f}")
    print("="*60)
    print("✅ Cleanup complete. Detection stopped.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Set env (before heavy imports) ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import time
import threading
import queue
import subprocess
import pathlib
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import onnxruntime as ort
except Exception:
    ort = None

# =======================
# Config
# =======================
# Paths
MODEL_PT = "/home/pi/PPE_Detection/new_model_test/publicsh17.pt"

# Inference (tune for speed)
IMGSZ = 416            
CONF = 0.45
IOU = 0.45
MAX_DET = 20

MODEL_ONNX = f"/home/pi/PPE_Detection/new_model_test/publicsh17_dyn_{IMGSZ}.onnx"

# Camera & pipeline
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
FRAMERATE = 30

# Classes - now including Head for better detection
CLASSES = [0, 1, 3]    # Person (0), Head (1), Glasses (3)
SHOW = True            
FRAME_SKIP_BASE = 3    
TARGET_INFER_MS = 120  

# Detection thresholds
HEAD_OVERLAP_THRESHOLD = 0.7   # How much glasses should overlap with head region
PERSON_UPPER_RATIO = 0.35      # Upper portion of person bbox to look for glasses
MIN_GLASSES_SIZE = 20          # Minimum glasses bbox width/height
MAX_GLASSES_SIZE = 200         # Maximum glasses bbox width/height

# Temporal smoothing
DETECTION_HISTORY_SIZE = 5     # Number of frames to average for stability
WEARING_THRESHOLD = 0.6        # Percentage of recent frames needed to confirm wearing

# Logging
LOG_SUMMARY_EVERY_INFER = True
LOG_DETAILS = False
LOG_VIOLATIONS = True           # Log safety violations

# Class name map
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

# =======================
# Data Classes
# =======================
@dataclass
class Detection:
    """Represents a single detection"""
    xyxy: Tuple[int, int, int, int]
    cls: int
    conf: float
    
    @property
    def x1(self): return self.xyxy[0]
    @property
    def y1(self): return self.xyxy[1]
    @property
    def x2(self): return self.xyxy[2]
    @property
    def y2(self): return self.xyxy[3]
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def center(self): return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    @property
    def area(self): return self.width * self.height

@dataclass
class PersonStatus:
    """Tracks a person's safety equipment status"""
    person_det: Detection
    head_det: Optional[Detection] = None
    glasses_det: Optional[Detection] = None
    wearing_glasses: bool = False
    confidence: float = 0.0
    
# =======================
# Spatial Logic Functions
# =======================
def calc_iou(box1: Detection, box2: Detection) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = box1.area + box2.area - intersection
    
    return intersection / union if union > 0 else 0.0

def calc_overlap_ratio(small_box: Detection, large_box: Detection) -> float:
    """Calculate how much of small_box overlaps with large_box"""
    x1 = max(small_box.x1, large_box.x1)
    y1 = max(small_box.y1, large_box.y1)
    x2 = min(small_box.x2, large_box.x2)
    y2 = min(small_box.y2, large_box.y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    return intersection / small_box.area if small_box.area > 0 else 0.0

def is_glasses_on_head(glasses: Detection, head: Detection) -> Tuple[bool, float]:
    """Check if glasses are positioned on a head"""
    overlap = calc_overlap_ratio(glasses, head)
    
    # Glasses should be in upper half of head
    head_upper_y = head.y1 + (head.height * 0.5)
    glasses_center_y = glasses.center[1]
    
    if glasses_center_y > head_upper_y:
        overlap *= 0.5  # Penalty if glasses too low
    
    # Size sanity check
    if glasses.width < MIN_GLASSES_SIZE or glasses.width > MAX_GLASSES_SIZE:
        return False, 0.0
    if glasses.height < MIN_GLASSES_SIZE or glasses.height > MAX_GLASSES_SIZE:
        return False, 0.0
    
    # Glasses shouldn't be too big relative to head
    if glasses.area > head.area * 0.6:
        return False, 0.0
    
    is_wearing = overlap > HEAD_OVERLAP_THRESHOLD
    return is_wearing, overlap

def is_glasses_on_person(glasses: Detection, person: Detection) -> Tuple[bool, float]:
    """Check if glasses are in the upper region of a person (when no head detected)"""
    # Define upper body region (top 35% of person bbox)
    upper_y_limit = person.y1 + int(person.height * PERSON_UPPER_RATIO)
    
    # Check if glasses center is in upper region
    glasses_cx, glasses_cy = glasses.center
    
    if not (person.x1 <= glasses_cx <= person.x2):
        return False, 0.0
    
    if not (person.y1 <= glasses_cy <= upper_y_limit):
        return False, 0.0
    
    # Calculate confidence based on position
    overlap = calc_overlap_ratio(glasses, person)
    relative_y = (glasses_cy - person.y1) / person.height
    
    # Higher confidence if glasses are higher up
    position_score = 1.0 - (relative_y / PERSON_UPPER_RATIO)
    confidence = overlap * position_score
    
    return confidence > 0.3, confidence

def analyze_detections(persons: List[Detection], heads: List[Detection], 
                       glasses: List[Detection]) -> List[PersonStatus]:
    """
    Analyze spatial relationships between persons, heads, and glasses.
    Returns a list of PersonStatus objects with PPE compliance info.
    """
    person_statuses = []
    
    # Track which glasses/heads have been assigned
    assigned_glasses = set()
    assigned_heads = set()
    
    for person in persons:
        status = PersonStatus(person_det=person)
        
        # Find head belonging to this person
        best_head = None
        best_head_overlap = 0
        
        for i, head in enumerate(heads):
            if i in assigned_heads:
                continue
            
            # Head should be in upper part of person
            upper_y_limit = person.y1 + int(person.height * 0.4)
            if head.center[1] > upper_y_limit:
                continue
                
            overlap = calc_overlap_ratio(head, person)
            if overlap > best_head_overlap and overlap > 0.5:
                best_head = head
                best_head_overlap = overlap
                best_head_idx = i
        
        if best_head:
            status.head_det = best_head
            assigned_heads.add(best_head_idx)
            
            # Look for glasses on this head
            best_glasses = None
            best_glasses_conf = 0
            
            for j, glass in enumerate(glasses):
                if j in assigned_glasses:
                    continue
                    
                is_wearing, conf = is_glasses_on_head(glass, best_head)
                if is_wearing and conf > best_glasses_conf:
                    best_glasses = glass
                    best_glasses_conf = conf
                    best_glasses_idx = j
            
            if best_glasses:
                status.glasses_det = best_glasses
                status.wearing_glasses = True
                status.confidence = best_glasses_conf
                assigned_glasses.add(best_glasses_idx)
        else:
            # No head detected, check if glasses on person upper body
            best_glasses = None
            best_glasses_conf = 0
            
            for j, glass in enumerate(glasses):
                if j in assigned_glasses:
                    continue
                    
                is_wearing, conf = is_glasses_on_person(glass, person)
                if is_wearing and conf > best_glasses_conf:
                    best_glasses = glass
                    best_glasses_conf = conf
                    best_glasses_idx = j
            
            if best_glasses:
                status.glasses_det = best_glasses
                status.wearing_glasses = True
                status.confidence = best_glasses_conf
                assigned_glasses.add(best_glasses_idx)
        
        person_statuses.append(status)
    
    return person_statuses

# =======================
# Temporal Smoothing
# =======================
class TemporalSmoothing:
    """Maintains detection history for stability"""
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.person_histories = {}  # Track by approximate position
        
    def update(self, person_statuses: List[PersonStatus]) -> List[PersonStatus]:
        """Update histories and return smoothed results"""
        current_positions = {}
        
        for status in person_statuses:
            # Use person center as rough ID
            center = status.person_det.center
            
            # Find closest match in history
            best_match_key = None
            best_distance = float('inf')
            
            for key in self.person_histories:
                dist = np.sqrt((center[0] - key[0])**2 + (center[1] - key[1])**2)
                if dist < 100 and dist < best_distance:  # Within 100 pixels
                    best_distance = dist
                    best_match_key = key
            
            if best_match_key:
                # Update existing history
                history = self.person_histories[best_match_key]
                history.append(status.wearing_glasses)
                if len(history) > self.history_size:
                    history.popleft()
                
                # Update position key
                del self.person_histories[best_match_key]
                self.person_histories[center] = history
                
                # Calculate smoothed wearing status
                wearing_ratio = sum(history) / len(history)
                status.wearing_glasses = wearing_ratio >= WEARING_THRESHOLD
                status.confidence = wearing_ratio
            else:
                # New person
                self.person_histories[center] = deque([status.wearing_glasses])
                
            current_positions[center] = True
        
        # Clean up old entries
        keys_to_remove = [k for k in self.person_histories if k not in current_positions]
        for key in keys_to_remove:
            del self.person_histories[key]
        
        return person_statuses

# =======================
# Helpers
# =======================
def ensure_onnx_dynamic(pt_path: str, onnx_path: str, imgsz: int):
    """Ensure ONNX model exists"""
    onnx_p = pathlib.Path(onnx_path)
    if onnx_p.exists():
        return
    pt_p = pathlib.Path(pt_path)
    if not pt_p.exists():
        print(f"[ERROR] .pt model not found at: {pt_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[EXPORT] Creating DYNAMIC ONNX from {pt_path} (imgsz={imgsz}) ...")
    YOLO(str(pt_p)).export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True)
    produced = pt_p.with_suffix('.onnx')
    try:
        if produced.resolve() != onnx_p.resolve():
            produced.replace(onnx_p)
    except Exception:
        pass
    print(f"[EXPORT] ONNX (dynamic) saved to: {onnx_path}")

def to_detections(r_boxes, cls_filter=None) -> List[Detection]:
    """Convert YOLO boxes to Detection objects"""
    out = []
    if r_boxes is None or len(r_boxes) == 0:
        return out
    for b in r_boxes:
        cls = int(b.cls[0]) if b.cls is not None else -1
        if cls_filter is not None and cls not in cls_filter:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        out.append(Detection(xyxy=(x1, y1, x2, y2), cls=cls, conf=conf))
    return out

def draw_status(img, person_statuses: List[PersonStatus]):
    """Draw detection results with safety status"""
    for status in person_statuses:
        p = status.person_det
        
        # Color based on compliance
        if status.wearing_glasses:
            color = (0, 255, 0)  # Green - compliant
            label = "SAFE"
        else:
            color = (0, 0, 255)  # Red - violation
            label = "NO GLASSES!"
        
        # Draw person box
        cv2.rectangle(img, (p.x1, p.y1), (p.x2, p.y2), color, 2)
        
        # Draw status label
        label_full = f"{label} ({status.confidence:.2f})"
        cv2.putText(img, label_full, (p.x1, max(0, p.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Draw head if detected
        if status.head_det:
            h = status.head_det
            cv2.rectangle(img, (h.x1, h.y1), (h.x2, h.y2), (255, 255, 0), 1)
        
        # Draw glasses if detected
        if status.glasses_det:
            g = status.glasses_det
            glasses_color = (0, 255, 255) if status.wearing_glasses else (128, 128, 128)
            cv2.rectangle(img, (g.x1, g.y1), (g.x2, g.y2), glasses_color, 2)

def draw_summary(img, total_persons: int, compliant: int, violations: int):
    """Draw summary statistics on frame"""
    # Background for text
    cv2.rectangle(img, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (250, 80), (255, 255, 255), 1)
    
    # Summary text
    cv2.putText(img, f"Total People: {total_persons}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Compliant: {compliant}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"Violations: {violations}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# =======================
# Main
# =======================
def main():
    # OpenCV perf knobs
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 1) Ensure ONNX exists
    ensure_onnx_dynamic(MODEL_PT, MODEL_ONNX, IMGSZ)

    # 2) Load ONNX model
    model = YOLO(MODEL_ONNX)
    try:
        model.model.names = CUSTOM_NAMES
    except Exception:
        pass

    # ORT optimization
    if ort is not None and hasattr(model, "predictor") and hasattr(model.predictor, "session"):
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CPUExecutionProvider"]
            model.predictor.session = ort.InferenceSession(MODEL_ONNX, sess_options=so, providers=providers)
            print("[OPT] ONNXRuntime session configured for CPUExecutionProvider with 4 threads.")
        except Exception as e:
            print(f"[OPT] ORT session tweak skipped: {e}", file=sys.stderr)

    # 3) Start camera
    cam_cmd = [
        "rpicam-vid", "--inline",
        "--width", str(WIDTH), "--height", str(HEIGHT),
        "--codec", "yuv420",
        "-o", "-", "-t", "0",
        "--framerate", str(FRAMERATE)
    ]
    try:
        cam = subprocess.Popen(cam_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[ERROR] rpicam-vid not found. Install libcamera apps on Raspberry Pi OS.", file=sys.stderr)
        sys.exit(1)

    # 4) Reader thread
    q_frames = queue.Queue(maxsize=3)
    stop_flag = {"stop": False}

    def reader():
        while not stop_flag["stop"]:
            raw = cam.stdout.read(FRAME_SIZE)
            if not raw:
                break
            yuv = np.frombuffer(raw, dtype=np.uint8)
            if yuv.size != FRAME_SIZE:
                continue
            yuv = yuv.reshape((int(HEIGHT * 1.5), WIDTH))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            try:
                q_frames.put_nowait(frame)
            except queue.Full:
                try:
                    _ = q_frames.get_nowait()
                except queue.Empty:
                    pass
                q_frames.put_nowait(frame)

    reader_t = threading.Thread(target=reader, daemon=True)
    reader_t.start()

    # 5) Warmup
    warmup = 2
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for _ in range(warmup):
        _ = model.predict(dummy, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                          device="cpu", verbose=False, classes=CLASSES)

    # 6) Initialize temporal smoothing
    temporal_smoother = TemporalSmoothing(DETECTION_HISTORY_SIZE)

    # 7) Main inference loop
    frame_id = 0
    adaptive_skip = FRAME_SKIP_BASE
    last_infer_end = time.time()
    
    # Statistics
    total_violations = 0
    total_compliant = 0

    try:
        while True:
            try:
                frame = q_frames.get(timeout=1.0)
            except queue.Empty:
                print("[INFO] No frames available; exiting.")
                break

            frame_id += 1

            # Skip frames for performance
            if frame_id % adaptive_skip != 0:
                if SHOW:
                    disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                    cv2.imshow("PPE Safety Detection", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            t0 = time.time()
            
            # Run inference
            results = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                device="cpu",
                verbose=False,
                classes=CLASSES,
                stream=False
            )
            
            t1 = time.time()

            # Parse detections
            r = results[0]
            all_dets = to_detections(r.boxes)
            
            persons = [d for d in all_dets if d.cls == 0]
            heads = [d for d in all_dets if d.cls == 1]
            glasses = [d for d in all_dets if d.cls == 3]

            # Analyze spatial relationships
            person_statuses = analyze_detections(persons, heads, glasses)
            
            # Apply temporal smoothing
            person_statuses = temporal_smoother.update(person_statuses)
            
            # Count compliance
            compliant = sum(1 for s in person_statuses if s.wearing_glasses)
            violations = len(person_statuses) - compliant
            
            # Update stats
            total_compliant += compliant
            total_violations += violations

            # Logging
            infer_ms = (t1 - t0) * 1000.0
            
            if LOG_SUMMARY_EVERY_INFER and len(person_statuses) > 0:
                print(f"[DETECT] People: {len(person_statuses)} | "
                      f"Compliant: {compliant} | Violations: {violations} | "
                      f"Infer: {infer_ms:.1f}ms", flush=True)
            
            if LOG_VIOLATIONS and violations > 0:
                for status in person_statuses:
                    if not status.wearing_glasses:
                        print(f"[VIOLATION] Person at ({status.person_det.center}) "
                              f"not wearing safety glasses!", flush=True)

            # Draw results
            if SHOW:
                draw_status(frame, person_statuses)
                draw_summary(frame, len(person_statuses), compliant, violations)
                
                disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                cv2.imshow("PPE Safety Detection", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Adaptive frame skipping
            if infer_ms > TARGET_INFER_MS and adaptive_skip < 6:
                adaptive_skip += 1
            elif infer_ms < TARGET_INFER_MS * 0.6 and adaptive_skip > 1:
                adaptive_skip -= 1

    finally:
        # Cleanup
        stop_flag["stop"] = True
        print(f"\n[SUMMARY] Total Compliant: {total_compliant}, "
              f"Total Violations: {total_violations}")
        try:
            cam.kill()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
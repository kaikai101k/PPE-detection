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
from typing import List, Dict, Tuple
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

# Inference
IMGSZ = 416            
CONF = 0.45
IOU = 0.45
MAX_DET = 20

MODEL_ONNX = f"/home/pi/PPE_Detection/new_model_test/publicsh17_dyn_{IMGSZ}.onnx"

# Camera
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)
FRAMERATE = 30

# Classes - simplified: just person and glasses
CLASSES = [0, 3]       # Person (0), Glasses (3)
SHOW = True            
FRAME_SKIP_BASE = 3    
TARGET_INFER_MS = 120  

# Simple zone detection - just one parameter!
HEAD_ZONE_RATIO = 0.3  # Check top 30% of person bbox for glasses

# Logging
LOG_SUMMARY_EVERY_INFER = True
LOG_VIOLATIONS = True

# Class names
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
# Simple Detection Logic
# =======================
def check_glasses_in_head_zone(person_box, glasses_boxes):
    """
    Super simple: Check if any glasses center point is in the upper zone of person bbox.
    Returns (is_wearing, confidence_based_on_position)
    """
    if not glasses_boxes:
        return False, 0.0
    
    x1, y1, x2, y2 = person_box
    
    # Define head zone (top portion of person bbox)
    head_zone_bottom = y1 + int((y2 - y1) * HEAD_ZONE_RATIO)
    
    # Check each glasses detection
    best_confidence = 0.0
    found_in_zone = False
    
    for glass_box in glasses_boxes:
        gx1, gy1, gx2, gy2 = glass_box
        
        # Get glasses center point
        glass_center_x = (gx1 + gx2) // 2
        glass_center_y = (gy1 + gy2) // 2
        
        # Simple check: Is glasses center in the person's head zone?
        if (x1 <= glass_center_x <= x2) and (y1 <= glass_center_y <= head_zone_bottom):
            found_in_zone = True
            
            # Simple confidence: higher = closer to top of person
            relative_height = 1.0 - ((glass_center_y - y1) / (head_zone_bottom - y1))
            best_confidence = max(best_confidence, relative_height)
    
    return found_in_zone, best_confidence

def analyze_frame_simple(persons, glasses):
    """
    Simple frame analysis without complex overlap calculations.
    Returns list of (person_box, is_wearing_glasses, confidence)
    """
    results = []
    used_glasses = set()
    
    for person_box in persons:
        # Get all glasses not yet assigned
        available_glasses = [g for i, g in enumerate(glasses) if i not in used_glasses]
        
        # Check if any glasses in head zone
        is_wearing, confidence = check_glasses_in_head_zone(person_box, available_glasses)
        
        # Mark glasses as used if found
        if is_wearing:
            for i, glass_box in enumerate(glasses):
                if i not in used_glasses:
                    gx1, gy1, gx2, gy2 = glass_box
                    glass_center_x = (gx1 + gx2) // 2
                    glass_center_y = (gy1 + gy2) // 2
                    
                    x1, y1, x2, y2 = person_box
                    head_zone_bottom = y1 + int((y2 - y1) * HEAD_ZONE_RATIO)
                    
                    if (x1 <= glass_center_x <= x2) and (y1 <= glass_center_y <= head_zone_bottom):
                        used_glasses.add(i)
                        break
        
        results.append((person_box, is_wearing, confidence))
    
    return results

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

def extract_boxes(r_boxes, target_class):
    """Extract bounding boxes for specific class"""
    boxes = []
    if r_boxes is None or len(r_boxes) == 0:
        return boxes
    
    for b in r_boxes:
        cls = int(b.cls[0]) if b.cls is not None else -1
        if cls == target_class:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    
    return boxes

def draw_results_simple(img, results):
    """Simple visualization with color coding"""
    for person_box, is_wearing, confidence in results:
        x1, y1, x2, y2 = person_box
        
        # Color based on compliance
        if is_wearing:
            color = (0, 255, 0)  # Green - wearing glasses
            label = f"SAFE ({confidence:.1%})"
        else:
            color = (0, 0, 255)  # Red - not wearing glasses
            label = "NO GLASSES"
        
        # Draw box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Optionally draw head zone indicator
        if not is_wearing:
            head_zone_bottom = y1 + int((y2 - y1) * HEAD_ZONE_RATIO)
            cv2.line(img, (x1, head_zone_bottom), (x2, head_zone_bottom), (0, 128, 255), 1)

def draw_summary(img, total_persons: int, compliant: int, violations: int):
    """Draw summary statistics"""
    cv2.rectangle(img, (10, 10), (200, 70), (0, 0, 0), -1)
    cv2.putText(img, f"People: {total_persons}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Safe: {compliant}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"Violations: {violations}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# =======================
# Main
# =======================
def main():
    # OpenCV optimization
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)
    except Exception:
        pass

    # 1) Ensure ONNX exists
    ensure_onnx_dynamic(MODEL_PT, MODEL_ONNX, IMGSZ)

    # 2) Load model
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
            print("[OPT] ONNXRuntime session configured.")
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
        print("[ERROR] rpicam-vid not found.", file=sys.stderr)
        sys.exit(1)

    # 4) Frame reader thread
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
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for _ in range(2):
        _ = model.predict(dummy, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                          device="cpu", verbose=False, classes=CLASSES)

    # 6) Main loop
    frame_id = 0
    adaptive_skip = FRAME_SKIP_BASE
    
    # Statistics
    total_violations = 0
    total_compliant = 0
    
    # Simple temporal smoothing (optional)
    recent_violations = deque(maxlen=10)  # Track last 10 checks

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
                    cv2.imshow("PPE Safety Monitor", disp)
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

            # Extract detections
            r = results[0]
            persons = extract_boxes(r.boxes, 0)  # Person class
            glasses = extract_boxes(r.boxes, 3)  # Glasses class

            # Simple analysis
            analysis_results = analyze_frame_simple(persons, glasses)
            
            # Count compliance
            compliant = sum(1 for _, wearing, _ in analysis_results if wearing)
            violations = len(analysis_results) - compliant
            
            # Update stats
            total_compliant += compliant
            total_violations += violations
            recent_violations.append(violations)
            
            # Logging
            infer_ms = (t1 - t0) * 1000.0
            
            if LOG_SUMMARY_EVERY_INFER and len(analysis_results) > 0:
                print(f"[DETECT] People: {len(analysis_results)} | "
                      f"Safe: {compliant} | Violations: {violations} | "
                      f"Time: {infer_ms:.1f}ms", flush=True)
            
            if LOG_VIOLATIONS and violations > 0:
                print(f"[⚠️ ALERT] {violations} person(s) not wearing safety glasses!", flush=True)
                
            # Alert if persistent violations (optional)
            if len(recent_violations) == 10 and sum(recent_violations) > 5:
                print("[‼️ PERSISTENT VIOLATION] Multiple people without safety glasses!", flush=True)

            # Draw results
            if SHOW:
                draw_results_simple(frame, analysis_results)
                draw_summary(frame, len(analysis_results), compliant, violations)
                
                disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                cv2.imshow("PPE Safety Monitor", disp)
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
        print(f"\n[FINAL] Session Stats - Safe: {total_compliant}, Violations: {total_violations}")
        if total_compliant + total_violations > 0:
            compliance_rate = total_compliant / (total_compliant + total_violations) * 100
            print(f"[FINAL] Compliance Rate: {compliance_rate:.1f}%")
        try:
            cam.kill()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Set env (before heavy imports) ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
# Quiet ORT GPU probe warnings on Pi; harmless but noisy
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import time
import threading
import queue
import subprocess
import pathlib
import sys
from typing import List, Dict, Tuple

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
IMGSZ = 416            # try 416 or 448 for speed; must be multiple of 32
CONF = 0.45
IOU = 0.45
MAX_DET = 20

# We’ll export a DYNAMIC ONNX per size so it won’t clash with old fixed 512 model
MODEL_ONNX = f"/home/pi/PPE_Detection/new_model_test/publicsh17_dyn_{IMGSZ}.onnx"

# Camera & pipeline (choose stride-friendly height like 384 or 480; 480 is OK)
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)  # YUV420 (I420) frame size
FRAMERATE = 30

# Classes (only what we need for count-based policy)
CLASSES = [0, 3]       # Person (0), Glasses (3)  ← drop Face for speed since it's not firing
SHOW = True            # False for max FPS
FRAME_SKIP_BASE = 3    # increase if still slow (3–5 typical on Pi)
TARGET_INFER_MS = 120  # aim for ~8 FPS on infer frames

# Logging
LOG_SUMMARY_EVERY_INFER = True
LOG_DETAILS = False

# Class name map (0-based)

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
# Helpers
# =======================
def ensure_onnx_dynamic(pt_path: str, onnx_path: str, imgsz: int):
    """
    Ensure a DYNAMIC-shape ONNX exists that matches our desired imgsz bucket.
    If missing, export with dynamic=True so we can vary imgsz later without re-export.
    """
    onnx_p = pathlib.Path(onnx_path)
    if onnx_p.exists():
        return
    pt_p = pathlib.Path(pt_path)
    if not pt_p.exists():
        print(f"[ERROR] .pt model not found at: {pt_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[EXPORT] Creating DYNAMIC ONNX from {pt_path} (imgsz={imgsz}) ...")
    # dynamic=True -> flexible spatial dims (must still be multiple of stride, e.g., 32)
    YOLO(str(pt_p)).export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True)
    # Ultralytics writes to same stem; if user provided a custom onnx_path, move/rename
    produced = pt_p.with_suffix('.onnx')
    try:
        if produced.resolve() != onnx_p.resolve():
            produced.replace(onnx_p)
    except Exception:
        pass
    print(f"[EXPORT] ONNX (dynamic) saved to: {onnx_path}")

def to_box_dicts(r_boxes) -> List[Dict]:
    out = []
    if r_boxes is None or len(r_boxes) == 0:
        return out
    for b in r_boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0]) if b.cls is not None else -1
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        out.append({"xyxy": (x1, y1, x2, y2), "cls": cls, "conf": conf})
    return out

def draw_boxes_fast(img, persons, glasses):
    """
    Minimal overlays for speed.
      - Persons: light blue thin box
      - Glasses: cyan thin box
    """
    for p in persons:
        x1, y1, x2, y2 = p["xyxy"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(img, f"Person {p['conf']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
    for g in glasses:
        x1, y1, x2, y2 = g["xyxy"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 255, 255), 1)
        cv2.putText(img, f"Glasses {g['conf']:.2f}", (x1, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1, cv2.LINE_AA)

# =======================
# Main
# =======================
def main():
    # OpenCV perf knobs on Pi
    try:
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)  # avoid CPU thread thrash with ORT
    except Exception:
        pass

    # 1) Ensure DYNAMIC ONNX exists for our chosen size
    ensure_onnx_dynamic(MODEL_PT, MODEL_ONNX, IMGSZ)

    # 2) Load ONNX with Ultralytics (onnxruntime backend)
    model = YOLO(MODEL_ONNX)
    try:
        model.model.names = CUSTOM_NAMES
    except Exception:
        pass

    # Optional: tighten ORT session for Pi CPU
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

    # 3) Start rpicam (YUV420 for cheap conversion)
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

    # 4) Reader thread with small queue (drop oldest when full)
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

    # 5) Warmup (ensures kernels/graph are ready at our chosen imgsz)
    warmup = 2
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for _ in range(warmup):
        _ = model.predict(dummy, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                          device="cpu", verbose=False, classes=CLASSES)

    # 6) Inference loop with adaptive skipping
    frame_id = 0
    adaptive_skip = FRAME_SKIP_BASE
    last_infer_end = time.time()

    try:
        while True:
            try:
                frame = q_frames.get(timeout=1.0)
            except queue.Empty:
                print("[INFO] No frames available; exiting.")
                break

            frame_id += 1

            # Cheap preview on skipped frames
            if frame_id % adaptive_skip != 0:
                if SHOW:
                    disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                    cv2.imshow("YOLO (ONNX) Detection", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            t0 = time.time()
            results = model.predict(
                frame,
                imgsz=IMGSZ,              # must be multiple of 32, <= training size; works w/ dynamic ONNX
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                device="cpu",
                verbose=False,
                classes=CLASSES,
                stream=False
            )
            t1 = time.time()

            r = results[0]
            dets = to_box_dicts(r.boxes)
            persons = [d for d in dets if d["cls"] == 0]
            glasses = [d for d in dets if d["cls"] == 3]

            persons_count = len(persons)
            glasses_count = len(glasses)
            missing = max(0, persons_count - glasses_count)

            if LOG_DETAILS:
                for p in persons:
                    print(f"[PERSON] bbox={p['xyxy']} conf={p['conf']:.2f}", flush=True)
                for g in glasses:
                    print(f"[GLASSES] bbox={g['xyxy']} conf={g['conf']:.2f}", flush=True)

            infer_ms = (t1 - t0) * 1000.0
            if LOG_SUMMARY_EVERY_INFER:
                if persons_count > 0:
                    print(f"[DETECT] persons={persons_count}, glasses={glasses_count}, "
                          f"missing_glasses={missing} | infer={infer_ms:.1f} ms | skip={adaptive_skip}",
                          flush=True)
                else:
                    print(f"[DETECT] (no persons) glasses={glasses_count} | "
                          f"infer={infer_ms:.1f} ms | skip={adaptive_skip}", flush=True)

            # Draw (optional)
            if SHOW:
                draw_boxes_fast(frame, persons, glasses)
                disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                cv2.imshow("YOLO (ONNX) Detection", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Adaptive skip to chase target latency
            if infer_ms > TARGET_INFER_MS and adaptive_skip < 6:
                adaptive_skip += 1
            elif infer_ms < TARGET_INFER_MS * 0.6 and adaptive_skip > 1:
                adaptive_skip -= 1
            last_infer_end = time.time()

    finally:
        # Cleanup
        stop_flag["stop"] = True
        try:
            cam.kill()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

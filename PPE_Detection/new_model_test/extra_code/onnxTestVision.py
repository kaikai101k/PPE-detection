#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Set env (before heavy imports) ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import time
import threading
import queue
import subprocess
import pathlib
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# =======================
# Config
# =======================
# Paths
MODEL_PT   = "/home/pi/PPE_Detection/new_model_test/publicsh17.pt"
MODEL_ONNX = "/home/pi/PPE_Detection/new_model_test/publicsh17.onnx"  # created on first run if missing

# Camera & pipeline
WIDTH, HEIGHT = 640, 480
FRAME_SIZE = int(WIDTH * HEIGHT * 1.5)  # YUV420 (I420) frame size
FRAMERATE = 30

# Inference
IMGSZ = 512            # try 512; if you need more FPS, test 480 or 416
CONF = 0.45
IOU = 0.45
MAX_DET = 20           # low since we only care about 2 classes
CLASSES = [2, 3]       # Face (2), Glasses (3) — zero-based indices
SHOW = True            # set False for max FPS
FRAME_SKIP_BASE = 2    # base skip; adaptive logic will adjust around this

# Terminal logging for glasses
LOG_GLASSES_EVERY_FRAME = True  # print once per frame if glasses detected

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
def ensure_onnx_exists(pt_path: str, onnx_path: str, imgsz: int = 512):
    """
    If the ONNX file doesn't exist, export it from the .pt using Ultralytics.
    No dataset is required for export.
    """
    onnx_p = pathlib.Path(onnx_path)
    if onnx_p.exists():
        return
    pt_p = pathlib.Path(pt_path)
    if not pt_p.exists():
        print(f"[ERROR] .pt model not found at: {pt_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[EXPORT] Creating ONNX from {pt_path} (imgsz={imgsz}) ...")
    YOLO(str(pt_p)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    print(f"[EXPORT] ONNX saved to: {onnx_path}")

def draw_boxes_fast(img, boxes, names_map):
    """
    Minimal, fast OpenCV drawing for boxes and labels.
    """
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = int(b.cls[0]) if b.cls is not None else -1
        conf = float(b.conf[0]) if b.conf is not None else 0.0
        color = (0, 255, 0) if cls == 3 else (255, 255, 0)  # Glasses vs Face
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{names_map.get(cls, str(cls))} {conf:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# =======================
# Main
# =======================
def main():
    # 1) Ensure ONNX model exists (export if needed)
    ensure_onnx_exists(MODEL_PT, MODEL_ONNX, IMGSZ)

    # 2) Load ONNX with Ultralytics (uses onnxruntime under the hood)
    model = YOLO(MODEL_ONNX)

    # Overwrite names to guarantee class alignment with your list
    try:
        model.model.names = CUSTOM_NAMES
    except Exception:
        pass
    names = getattr(model.model, "names", CUSTOM_NAMES)

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
                    _ = q_frames.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                q_frames.put_nowait(frame)

    reader_t = threading.Thread(target=reader, daemon=True)
    reader_t.start()

    # 5) Inference loop with adaptive skipping
    frame_id = 0
    adaptive_skip = FRAME_SKIP_BASE
    last_infer_time = time.time()

    try:
        while True:
            try:
                frame = q_frames.get(timeout=1.0)
            except queue.Empty:
                print("[INFO] No frames available; exiting.")
                break

            frame_id += 1
            # Optional fast preview even on skipped frames
            if frame_id % adaptive_skip != 0:
                if SHOW:
                    disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                    cv2.imshow("YOLO (ONNX) Detection", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # ---- Inference (ONNX Runtime via Ultralytics) ----
            results = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                device="cpu",
                verbose=False,
                classes=CLASSES  # only Face & Glasses
            )
            r = results[0]

            # ---- Log to terminal if GLASSES detected ----
            glasses_detected_this_frame = False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls = int(b.cls[0]) if b.cls is not None else -1
                    if cls == 3:  # Glasses
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        conf = float(b.conf[0]) if b.conf is not None else 0.0
                        print(f"[GLASSES] conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})", flush=True)
                        glasses_detected_this_frame = True

            # (Optional) If you prefer a single line per frame when any glasses are seen:
            if LOG_GLASSES_EVERY_FRAME and glasses_detected_this_frame:
                pass  # already printed each detection; keep or aggregate if you want

            # ---- Draw (optional) ----
            if SHOW:
                if r.boxes is not None and len(r.boxes) > 0:
                    draw_boxes_fast(frame, r.boxes, names)
                disp = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                cv2.imshow("YOLO (ONNX) Detection", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ---- Adaptive skip (aim to keep detect-FPS comfortable) ----
            now = time.time()
            infer_ms = (now - last_infer_time) * 1000.0
            last_infer_time = now
            # If detections are taking too long, increase skipping; if fast, lower it
            if infer_ms > 90 and adaptive_skip < 6:
                adaptive_skip += 1
            elif infer_ms < 60 and adaptive_skip > 1:
                adaptive_skip -= 1

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

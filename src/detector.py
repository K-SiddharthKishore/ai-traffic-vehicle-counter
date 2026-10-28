"""
detector.py — YOLO model loader and vehicle detection wrapper.
Handles model caching, device selection, and per-frame inference.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs for the four vehicle categories we count
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# All classes we want to display (vehicles + optional pedestrian display)
DISPLAY_CLASSES = set(VEHICLE_CLASSES.keys()) | {0}  # 0 = person

# Colours for bounding boxes  (BGR for OpenCV)
CLASS_COLORS = {
    2: (0, 200, 0),      # Car — green
    3: (0, 140, 255),    # Motorcycle — orange
    5: (255, 50, 50),    # Bus — blue
    7: (0, 0, 220),      # Truck — red
    0: (180, 180, 180),  # Person — grey
}
DEFAULT_COLOR = (200, 200, 200)


def load_model(model_name: str = "yolo11s.pt"):
    """
    Load a pretrained YOLO model, downloading weights if needed.
    Tries yolo11s first (accurate), falls back to yolo11n then yolov8n.
    Returns the loaded model or raises RuntimeError.
    """
    from ultralytics import YOLO

    candidates = [model_name]
    if model_name == "yolo11s.pt":
        candidates += ["yolo11n.pt", "yolov8n.pt"]
    elif model_name != "yolov8n.pt":
        candidates.append("yolov8n.pt")

    last_error = None
    for name in candidates:
        try:
            logger.info("Loading model: %s", name)
            model = YOLO(name)
            logger.info("Model loaded: %s", name)
            return model
        except Exception as exc:
            logger.warning("Could not load %s: %s", name, exc)
            last_error = exc

    raise RuntimeError(
        f"Could not load any YOLO model. Last error: {last_error}\n"
        "Make sure you have an internet connection for the first run "
        "so weights can be downloaded automatically."
    )


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def detect_frame(model, frame: np.ndarray, conf: float = 0.3, imgsz: int = 640,
                 device: str = "cpu"):
    """
    Run YOLO tracking on a single frame.
    Returns a list of dicts, each describing one detection:
        {track_id, class_id, class_name, bbox: (x1,y1,x2,y2), conf, cx, cy}
    """
    results = model.track(
        frame,
        persist=True,
        conf=conf,
        iou=0.45,
        imgsz=imgsz,
        device=device,
        tracker="bytetrack.yaml",
        classes=list(DISPLAY_CLASSES),
        verbose=False,
    )

    detections = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in DISPLAY_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        conf_val = float(box.conf[0])

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0])

        detections.append({
            "track_id": track_id,
            "class_id": cls_id,
            "class_name": VEHICLE_CLASSES.get(cls_id, "Person"),
            "bbox": (x1, y1, x2, y2),
            "conf": conf_val,
            "cx": cx,
            "cy": cy,
        })

    return detections

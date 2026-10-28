"""
video_processor.py — Frame-by-frame processing pipeline.

Reads the input video, runs detection + tracking on sampled frames,
draws annotations, writes the output video, and returns final results.
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .detector import (
    CLASS_COLORS, DEFAULT_COLOR, VEHICLE_CLASSES, detect_frame, get_device
)
from .tracker import LineCounter

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Optional[dict]:
    """Return basic video metadata or None if the file cannot be opened."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


def process_video(
    model,
    video_path: str,
    output_path: str,
    line_position: float,
    orientation: str = "horizontal",
    frame_skip: int = 1,
    conf_threshold: float = 0.3,
    imgsz: int = 640,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """
    Main processing function.

    Parameters
    ----------
    model       : loaded YOLO model
    video_path  : path to the source video
    output_path : where to write the annotated output video
    line_position: pixel coordinate of counting line
    orientation : 'horizontal' or 'vertical'
    frame_skip  : process every Nth frame (1 = every frame)
    conf_threshold: minimum detection confidence
    imgsz       : YOLO input image size
    device      : 'cpu', 'cuda', 'mps' — auto-detected if None
    progress_callback: called each frame with (frame_idx, total, counts_dict)

    Returns a dict with final counts and metadata.
    """
    if device is None:
        device = get_device()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer — use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video: {output_path}")

    counter = LineCounter(
        line_position=line_position,
        orientation=orientation,
        frame_width=width,
        frame_height=height,
    )

    start_time = time.time()
    frame_idx = 0
    last_detections = []  # reuse annotations on skipped frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Run detection + tracking
                try:
                    detections = detect_frame(
                        model, frame, conf=conf_threshold,
                        imgsz=imgsz, device=device
                    )
                except Exception as exc:
                    logger.warning("Detection failed on frame %d: %s", frame_idx, exc)
                    detections = []

                last_detections = detections
                counter.update(detections, frame_idx)
            else:
                # Skipped frame: reuse previous detections for drawing
                detections = last_detections

            # Draw annotations
            annotated = _draw_frame(
                frame, detections, counter,
                line_position, orientation, width, height
            )
            out.write(annotated)

            if progress_callback:
                progress_callback(frame_idx, total_frames, counter.get_counts())

            frame_idx += 1

    finally:
        cap.release()
        out.release()

    elapsed = time.time() - start_time
    counts = counter.get_counts()

    return {
        "counts": counts,
        "total_vehicles": counter.total_count(),
        "crossing_log": counter.crossing_log,
        "processing_time": elapsed,
        "frames_processed": frame_idx,
        "fps_processed": frame_idx / elapsed if elapsed > 0 else 0,
        "video_fps": fps,
        "width": width,
        "height": height,
        "duration": total_frames / fps if fps > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_frame(
    frame: np.ndarray,
    detections: list,
    counter: LineCounter,
    line_position: float,
    orientation: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw bounding boxes, labels, counting line, and live count overlay."""
    out = frame.copy()

    # Draw counting line
    lstart = counter.line_start()
    lend = counter.line_end()
    cv2.line(out, lstart, lend, (0, 255, 255), 2)
    cv2.putText(
        out, "COUNTING LINE",
        (lstart[0] + 5, lstart[1] - 8 if orientation == "horizontal" else lstart[1] + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
    )

    # Draw bounding boxes
    for det in detections:
        cls_id = det["class_id"]
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        tid = det.get("track_id")
        label = f"{det['class_name']}"
        if tid is not None:
            label += f" #{tid}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Count overlay (top-left corner)
    counts = counter.get_counts()
    y_offset = 30
    overlay_lines = [f"Total: {counter.total_count()}"] + [
        f"{VEHICLE_CLASSES[cid]}: {cnt}"
        for cid, cnt in counts.items()
        if cid in VEHICLE_CLASSES
    ]
    for line_text in overlay_lines:
        cv2.putText(
            out, line_text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line_text, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA
        )
        y_offset += 28

    return out

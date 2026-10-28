"""
tracker.py — Line-crossing counter logic.

Maintains a record of which track IDs have already been counted so each
vehicle is counted exactly once even if it stays visible for many frames.
Also stores recent positions so we can detect genuine crossings (not just
vehicles that start on the line or sit still on it).
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class LineCounter:
    """
    Counts tracked vehicles crossing a virtual line.

    The line is defined as either:
      - horizontal: a fixed y value, crossing detected on y-axis movement
      - vertical  : a fixed x value, crossing detected on x-axis movement

    A vehicle is counted only once regardless of how many frames it stays
    near the line.  We look at the *previous* centre position vs the
    *current* centre position to decide if a genuine crossing happened.

    When ByteTrack is unavailable (track_id is None), a simple centroid-
    matching fallback assigns synthetic negative IDs so counting still works.
    """

    _CENTROID_MATCH_PX = 80  # max pixel distance to match a previous centroid

    def __init__(
        self,
        line_position: float,
        orientation: str = "horizontal",  # "horizontal" or "vertical"
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        self.line_position = line_position  # pixel coordinate of the line
        self.orientation = orientation
        self.frame_width = frame_width
        self.frame_height = frame_height

        # track_id → last known centre (cx, cy)
        self._prev_positions: Dict[int, Tuple[int, int]] = {}

        # track IDs that have already been counted
        self._counted_ids: set = set()

        # category counts: class_id → count
        self._counts: Dict[int, int] = defaultdict(int)

        # full crossing event log for timeline chart
        self.crossing_log: List[dict] = []

        # fallback centroid tracker (used when ByteTrack gives no track_id)
        self._next_fake_id: int = -1  # negative so never collides with real IDs
        self._fake_positions: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def _assign_fake_id(self, cx: int, cy: int) -> int:
        """Match centroid to an existing fake track or create a new one."""
        best_id, best_dist = None, self._CENTROID_MATCH_PX
        for fid, (fx, fy) in self._fake_positions.items():
            d = math.hypot(cx - fx, cy - fy)
            if d < best_dist:
                best_dist = d
                best_id = fid
        if best_id is None:
            best_id = self._next_fake_id
            self._next_fake_id -= 1
        self._fake_positions[best_id] = (cx, cy)
        return best_id

    def update(self, detections: list, frame_index: int) -> List[dict]:
        """
        Process detections for one frame.
        Returns a list of new crossing events (may be empty).
        """
        new_crossings = []

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                # ByteTrack unavailable — use centroid fallback
                tid = self._assign_fake_id(det["cx"], det["cy"])

            cx, cy = det["cx"], det["cy"]
            cls_id = det["class_id"]

            # skip pedestrians from counting (class 0)
            if cls_id == 0:
                self._prev_positions[tid] = (cx, cy)
                continue

            if tid not in self._counted_ids and tid in self._prev_positions:
                prev_cx, prev_cy = self._prev_positions[tid]
                if self._crossed(prev_cx, prev_cy, cx, cy):
                    self._counted_ids.add(tid)
                    self._counts[cls_id] += 1
                    event = {
                        "frame": frame_index,
                        "track_id": tid,
                        "class_id": cls_id,
                        "class_name": det["class_name"],
                    }
                    new_crossings.append(event)
                    self.crossing_log.append(event)

            self._prev_positions[tid] = (cx, cy)

        return new_crossings

    def get_counts(self) -> Dict[int, int]:
        """Return a copy of category counts keyed by class_id."""
        return dict(self._counts)

    def total_count(self) -> int:
        return sum(self._counts.values())

    def reset(self):
        self._prev_positions.clear()
        self._counted_ids.clear()
        self._counts.clear()
        self.crossing_log.clear()
        self._fake_positions.clear()
        self._next_fake_id = -1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crossed(self, px: int, py: int, cx: int, cy: int) -> bool:
        """Return True if the centre moved from one side of the line to the other."""
        lp = self.line_position
        if self.orientation == "horizontal":
            # crossing when py and cy are on opposite sides of lp
            return (py < lp <= cy) or (py > lp >= cy)
        else:
            return (px < lp <= cx) or (px > lp >= cx)

    # Line drawing helpers called by video_processor
    def line_start(self) -> Tuple[int, int]:
        if self.orientation == "horizontal":
            return (0, int(self.line_position))
        return (int(self.line_position), 0)

    def line_end(self) -> Tuple[int, int]:
        if self.orientation == "horizontal":
            return (self.frame_width, int(self.line_position))
        return (int(self.line_position), self.frame_height)

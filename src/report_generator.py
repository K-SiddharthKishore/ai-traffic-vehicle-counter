"""
report_generator.py — Create CSV and JSON summary reports from analysis results.
"""

import csv
import io
import json
from datetime import datetime
from typing import Dict, Optional

from .detector import VEHICLE_CLASSES


def make_csv(counts: Dict[int, int]) -> str:
    """Return a CSV string of category totals."""
    total = sum(counts.values())
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Category", "Count", "Percentage"])
    for cid, name in VEHICLE_CLASSES.items():
        cnt = counts.get(cid, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        writer.writerow([name, cnt, f"{pct:.1f}%"])
    writer.writerow(["Total", total, "100.0%"])
    return output.getvalue()


def make_json_report(
    filename: str,
    counts: Dict[int, int],
    video_info: dict,
    processing_time: float,
    frame_skip: int,
) -> str:
    """Return a JSON string summarising the analysis."""
    total = sum(counts.values())
    report = {
        "analysis_date": datetime.now().isoformat(timespec="seconds"),
        "original_filename": filename,
        "video_duration_seconds": round(video_info.get("duration", 0), 2),
        "video_resolution": f"{video_info.get('width', '?')}x{video_info.get('height', '?')}",
        "video_fps": round(video_info.get("fps", 0), 2),
        "frame_skip_interval": frame_skip,
        "processing_time_seconds": round(processing_time, 2),
        "total_vehicles_counted": total,
        "counts_by_category": {
            VEHICLE_CLASSES[cid]: counts.get(cid, 0)
            for cid in VEHICLE_CLASSES
        },
    }
    return json.dumps(report, indent=2)

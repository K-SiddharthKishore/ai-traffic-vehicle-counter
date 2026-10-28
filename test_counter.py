"""
test_counter.py — Unit tests for the vehicle counter logic.
Run with:  python test_counter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import traceback

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def run(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except AssertionError as exc:
        print(f"{FAIL}  {name}: {exc}")
        return False
    except Exception as exc:
        print(f"{FAIL}  {name}: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 1 — LineCounter: single vehicle crosses and is counted once
# ---------------------------------------------------------------------------
def test_single_crossing():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    det = {"track_id": 1, "class_id": 2, "class_name": "Car", "cx": 320, "cy": 230}
    lc.update([det], frame_index=0)  # above the line

    det["cy"] = 250
    events = lc.update([det], frame_index=1)  # below the line → should cross

    assert lc.total_count() == 1, f"Expected 1, got {lc.total_count()}"
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"


# ---------------------------------------------------------------------------
# Test 2 — Counted only once even if vehicle stays below the line
# ---------------------------------------------------------------------------
def test_no_double_count():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    det = {"track_id": 5, "class_id": 2, "class_name": "Car", "cx": 320, "cy": 230}
    lc.update([det], frame_index=0)

    det["cy"] = 250
    lc.update([det], frame_index=1)  # first crossing

    det["cy"] = 260
    lc.update([det], frame_index=2)  # stays below — no new count

    assert lc.total_count() == 1, f"Expected 1, got {lc.total_count()}"


# ---------------------------------------------------------------------------
# Test 3 — Vehicle that never crosses the line is NOT counted
# ---------------------------------------------------------------------------
def test_no_crossing_no_count():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    det = {"track_id": 7, "class_id": 2, "class_name": "Car", "cx": 320, "cy": 100}
    for frame in range(5):
        det["cx"] = 320 + frame * 10
        lc.update([det], frame_index=frame)

    assert lc.total_count() == 0, f"Expected 0, got {lc.total_count()}"


# ---------------------------------------------------------------------------
# Test 4 — Category counts are correct
# ---------------------------------------------------------------------------
def test_category_counts():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    vehicles = [
        {"track_id": 1, "class_id": 2, "class_name": "Car", "cx": 100, "cy": 230},
        {"track_id": 2, "class_id": 3, "class_name": "Motorcycle", "cx": 200, "cy": 230},
        {"track_id": 3, "class_id": 5, "class_name": "Bus", "cx": 300, "cy": 230},
    ]
    lc.update(vehicles, frame_index=0)

    for v in vehicles:
        v["cy"] = 250
    lc.update(vehicles, frame_index=1)

    counts = lc.get_counts()
    assert counts.get(2, 0) == 1, f"Cars: {counts.get(2,0)}"
    assert counts.get(3, 0) == 1, f"Motorcycles: {counts.get(3,0)}"
    assert counts.get(5, 0) == 1, f"Buses: {counts.get(5,0)}"
    assert lc.total_count() == 3


# ---------------------------------------------------------------------------
# Test 5 — Pedestrians are NOT counted
# ---------------------------------------------------------------------------
def test_pedestrian_not_counted():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    person = {"track_id": 99, "class_id": 0, "class_name": "Person", "cx": 320, "cy": 230}
    lc.update([person], frame_index=0)
    person["cy"] = 260
    lc.update([person], frame_index=1)

    assert lc.total_count() == 0, f"Expected 0 (person not counted), got {lc.total_count()}"


# ---------------------------------------------------------------------------
# Test 6 — Vertical line crossing
# ---------------------------------------------------------------------------
def test_vertical_crossing():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=320, orientation="vertical",
                     frame_width=640, frame_height=480)

    det = {"track_id": 10, "class_id": 7, "class_name": "Truck", "cx": 310, "cy": 240}
    lc.update([det], frame_index=0)
    det["cx"] = 330
    events = lc.update([det], frame_index=1)

    assert lc.total_count() == 1
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Test 7 — CSV report format
# ---------------------------------------------------------------------------
def test_csv_report():
    from src.report_generator import make_csv

    counts = {2: 5, 3: 2, 5: 1, 7: 3}
    csv_text = make_csv(counts)
    assert "Car" in csv_text
    assert "Motorcycle" in csv_text
    assert "Total" in csv_text
    lines = csv_text.strip().splitlines()
    assert len(lines) == 6, f"Expected 6 lines (header+4 cats+total), got {len(lines)}"


# ---------------------------------------------------------------------------
# Test 8 — JSON report is valid JSON
# ---------------------------------------------------------------------------
def test_json_report():
    import json
    from src.report_generator import make_json_report

    info = {"duration": 10.5, "width": 1280, "height": 720, "fps": 30.0}
    j = make_json_report("test.mp4", {2: 3, 3: 1}, info, 5.2, 2)
    data = json.loads(j)
    assert data["total_vehicles_counted"] == 4
    assert data["counts_by_category"]["Car"] == 3


# ---------------------------------------------------------------------------
# Test 9 — Empty video info from non-existent file
# ---------------------------------------------------------------------------
def test_invalid_video():
    from src.video_processor import get_video_info

    result = get_video_info("/nonexistent/path/video.mp4")
    assert result is None, f"Expected None for invalid path, got {result}"


# ---------------------------------------------------------------------------
# Test 10 — reset() clears all state
# ---------------------------------------------------------------------------
def test_reset():
    from src.tracker import LineCounter

    lc = LineCounter(line_position=240, orientation="horizontal",
                     frame_width=640, frame_height=480)

    det = {"track_id": 1, "class_id": 2, "class_name": "Car", "cx": 320, "cy": 230}
    lc.update([det], frame_index=0)
    det["cy"] = 250
    lc.update([det], frame_index=1)
    assert lc.total_count() == 1

    lc.reset()
    assert lc.total_count() == 0
    assert len(lc.crossing_log) == 0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("Single vehicle crosses line → counted once", test_single_crossing),
        ("Same vehicle stays past line → NOT double counted", test_no_double_count),
        ("Vehicle never crosses → count stays 0", test_no_crossing_no_count),
        ("Category counts (Car/Motorcycle/Bus)", test_category_counts),
        ("Pedestrian NOT included in vehicle count", test_pedestrian_not_counted),
        ("Vertical counting line crossing", test_vertical_crossing),
        ("CSV report format is correct", test_csv_report),
        ("JSON report is valid JSON with correct totals", test_json_report),
        ("Invalid video path returns None", test_invalid_video),
        ("reset() clears all state", test_reset),
    ]

    passed = 0
    for name, fn in tests:
        if run(name, fn):
            passed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {len(tests) - passed} test(s) failed — check output above.")
    print('='*50)

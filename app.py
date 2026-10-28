"""
AI Traffic Vehicle Counter — Streamlit Web Application
======================================================
Upload a traffic video → detect & track vehicles with YOLO →
count crossings over a virtual line → view results + download outputs.
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Vehicle Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, light, polished
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
        text-align: center; padding: 0.5rem 0;
    }
    .subtitle {
        font-size: 1rem; color: #4a4a6a; text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1rem 1.2rem; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,.12);
    }
    .metric-label { font-size: 0.8rem; opacity: 0.85; text-transform: uppercase; }
    .metric-value { font-size: 2rem; font-weight: 700; margin-top: 0.2rem; }
    .info-box {
        background: #f0f4ff; border-left: 4px solid #667eea;
        border-radius: 4px; padding: 0.7rem 1rem; margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .warn-box {
        background: #fff8e1; border-left: 4px solid #ffc107;
        border-radius: 4px; padding: 0.7rem 1rem; margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    div[data-testid="stProgress"] > div { height: 8px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🚦 AI Traffic Vehicle Counter</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a traffic video · detect & track vehicles with YOLO · '
    'count crossings over a virtual line</p>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Model loader (cached so it loads only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo_model():
    from src.detector import load_model
    return load_model("yolo11n.pt")


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.subheader("Counting Line")
    orientation = st.radio(
        "Line orientation",
        ["horizontal", "vertical"],
        index=0,
        help="Horizontal line counts vehicles moving up/down; vertical counts left/right.",
    )
    line_pct = st.slider(
        "Line position (% of frame)",
        min_value=10, max_value=90, value=50, step=5,
        help="Position of the virtual counting line as a percentage of frame height/width.",
    )

    st.subheader("Processing")
    frame_skip = st.select_slider(
        "Frame skip interval",
        options=[1, 2, 3, 5, 10],
        value=2,
        help="Process every Nth frame. Higher = faster but may miss fast vehicles.",
    )
    conf_threshold = st.slider(
        "Detection confidence threshold",
        min_value=0.1, max_value=0.9, value=0.30, step=0.05,
        help="Minimum YOLO confidence to accept a detection.",
    )
    imgsz = st.select_slider(
        "YOLO image size",
        options=[320, 416, 512, 640],
        value=640,
        help="Smaller = faster but less accurate.",
    )

    if frame_skip > 1:
        st.markdown(
            '<div class="warn-box">⚠️ Frame skipping can reduce counting '
            'accuracy for fast-moving vehicles.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Vehicle classes counted:**")
    st.markdown("🚗 Car (COCO class 2)")
    st.markdown("🏍️ Motorcycle (class 3)")
    st.markdown("🚌 Bus (class 5)")
    st.markdown("🚛 Truck (class 7)")
    st.markdown("*Pedestrians shown but NOT counted.*")

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
st.subheader("📁 Upload Traffic Video")

uploaded = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi"],
    help="MP4 is recommended. For best results, use a fixed-camera traffic video.",
)

if uploaded is None:
    st.markdown(
        '<div class="info-box">ℹ️ Upload a traffic video (MP4/MOV/AVI) using the panel above '
        'to begin. Results will appear here once analysis is complete.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Save uploaded file to a temp path
# ---------------------------------------------------------------------------
# Use session state to avoid re-saving on every Streamlit rerun
if "temp_input_path" not in st.session_state or \
        st.session_state.get("uploaded_name") != uploaded.name:
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.read())
    st.session_state["temp_input_path"] = input_path
    st.session_state["uploaded_name"] = uploaded.name
    st.session_state.pop("analysis_results", None)  # clear old results

input_path = st.session_state["temp_input_path"]

# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------
from src.video_processor import get_video_info

info = get_video_info(input_path)
if info is None:
    st.error("❌ Could not open the uploaded video. Please try a different file.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filename", uploaded.name)
col2.metric("Resolution", f"{info['width']}×{info['height']}")
col3.metric("Duration", f"{info['duration']:.1f}s")
col4.metric("Frame Rate", f"{info['fps']:.1f} fps")

# Compute line_position in pixels
line_position = (
    info["height"] * line_pct / 100
    if orientation == "horizontal"
    else info["width"] * line_pct / 100
)

# ---------------------------------------------------------------------------
# Start Analysis button
# ---------------------------------------------------------------------------
st.divider()
start_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

if start_btn:
    st.session_state.pop("analysis_results", None)

# ---------------------------------------------------------------------------
# Run analysis (only when button clicked; avoid re-running on rerun)
# ---------------------------------------------------------------------------
if start_btn and "analysis_results" not in st.session_state:
    try:
        model = load_yolo_model()
    except Exception as exc:
        st.error(f"❌ Failed to load YOLO model: {exc}")
        st.stop()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / f"annotated_{uploaded.name.rsplit('.', 1)[0]}.mp4")

    st.subheader("⏳ Processing…")
    progress_bar = st.progress(0)
    status_text = st.empty()
    col_a, col_b = st.columns(2)
    elapsed_ph = col_a.empty()
    fps_ph = col_b.empty()
    t0 = time.time()

    from src.detector import get_device
    device = get_device()
    status_text.markdown(f"**Device:** `{device}` | **Frame skip:** {frame_skip}")

    def on_progress(frame_idx, total, counts):
        pct = int(frame_idx / total * 100) if total > 0 else 0
        progress_bar.progress(min(pct, 100))
        elapsed = time.time() - t0
        elapsed_ph.metric("Elapsed", f"{elapsed:.1f}s")
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        fps_ph.metric("Processing FPS", f"{proc_fps:.1f}")

    try:
        from src.video_processor import process_video
        results = process_video(
            model=model,
            video_path=input_path,
            output_path=output_path,
            line_position=line_position,
            orientation=orientation,
            frame_skip=frame_skip,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
            progress_callback=on_progress,
        )
        results["output_path"] = output_path
        results["video_info"] = info
        results["frame_skip"] = frame_skip
        results["filename"] = uploaded.name
        st.session_state["analysis_results"] = results
        progress_bar.progress(100)
        status_text.success("✅ Analysis complete!")
    except Exception as exc:
        st.error(f"❌ Processing failed: {exc}")
        logger.exception("Processing error")
        st.stop()

# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.stop()

res = st.session_state["analysis_results"]
counts = res["counts"]
total = res["total_vehicles"]

from src.detector import VEHICLE_CLASSES

st.divider()
st.subheader("📊 Results Dashboard")

# --- Metric cards ---
metric_cols = st.columns(6)
metrics = [
    ("Total Vehicles", total, "🚦"),
    ("Cars", counts.get(2, 0), "🚗"),
    ("Motorcycles", counts.get(3, 0), "🏍️"),
    ("Buses", counts.get(5, 0), "🚌"),
    ("Trucks", counts.get(7, 0), "🚛"),
    ("Processing Time", f"{res['processing_time']:.1f}s", "⏱️"),
]
for col, (label, value, icon) in zip(metric_cols, metrics):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{icon} {label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# --- Chart + Table ---
chart_col, table_col = st.columns([1, 1])

with chart_col:
    st.markdown("#### Vehicle Category Distribution")
    if total > 0:
        import matplotlib.pyplot as plt

        labels = [VEHICLE_CLASSES[cid] for cid in VEHICLE_CLASSES]
        values = [counts.get(cid, 0) for cid in VEHICLE_CLASSES]
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=140,
            wedgeprops=dict(width=0.6),
        )
        for t in texts + autotexts:
            t.set_fontsize(10)
        ax.set_title("Counted Vehicles by Category", fontsize=12, fontweight="bold")
        fig.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No vehicles were counted crossing the line.")

with table_col:
    st.markdown("#### Category Breakdown")
    rows = []
    for cid, name in VEHICLE_CLASSES.items():
        cnt = counts.get(cid, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        rows.append({"Category": name, "Count": cnt, "Percentage": f"{pct:.1f}%"})
    rows.append({"Category": "**Total**", "Count": total, "Percentage": "100.0%"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### Processing Summary")
    summary_rows = [
        {"Metric": "Frames Processed", "Value": str(res.get("frames_processed", "?"))},
        {"Metric": "Processing FPS", "Value": f"{res.get('fps_processed', 0):.1f}"},
        {"Metric": "Video FPS", "Value": f"{res.get('video_fps', 0):.1f}"},
        {"Metric": "Frame Skip", "Value": str(res.get("frame_skip", 1))},
        {"Metric": "Resolution", "Value": f"{res.get('width','?')}×{res.get('height','?')}"},
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# --- Crossing timeline chart ---
crossing_log = res.get("crossing_log", [])
if crossing_log:
    st.markdown("#### Crossings Over Time")
    log_df = pd.DataFrame(crossing_log)
    log_df["vehicle"] = log_df["class_id"].map(VEHICLE_CLASSES)
    # bin by frame into ~20 buckets
    total_frames = res.get("frames_processed", 1)
    bins = max(1, total_frames // 30)
    log_df["time_bin"] = (log_df["frame"] // bins) * bins
    timeline = (
        log_df.groupby(["time_bin", "vehicle"])
        .size()
        .reset_index(name="crossings")
        .pivot(index="time_bin", columns="vehicle", values="crossings")
        .fillna(0)
    )
    st.bar_chart(timeline)

# --- Note about counting ---
st.markdown(
    '<div class="info-box">ℹ️ <b>How counting works:</b> Only tracked vehicles whose '
    'centre point crosses the virtual line are counted. A vehicle visible in many frames '
    'is counted exactly once. Vehicles never reaching the line are not counted.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Annotated video preview + download
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🎬 Annotated Output Video")

output_path = res.get("output_path", "")
if output_path and os.path.exists(output_path):
    file_size_mb = os.path.getsize(output_path) / 1_048_576
    st.markdown(f"Output file: `{os.path.basename(output_path)}` ({file_size_mb:.1f} MB)")
    try:
        with open(output_path, "rb") as vf:
            video_bytes = vf.read()
        st.video(video_bytes)
        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_bytes,
            file_name=os.path.basename(output_path),
            mime="video/mp4",
        )
    except Exception as exc:
        st.warning(f"Could not display video preview: {exc}. Download still available.")
        with open(output_path, "rb") as vf:
            st.download_button(
                label="⬇️ Download Annotated Video",
                data=vf,
                file_name=os.path.basename(output_path),
                mime="video/mp4",
            )
else:
    st.warning("Output video not found. The writer may have failed for this codec/format.")

# ---------------------------------------------------------------------------
# Export section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📥 Export Results")

from src.report_generator import make_csv, make_json_report

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    csv_data = make_csv(counts)
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_data,
        file_name=f"vehicle_counts_{uploaded.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )

with exp_col2:
    json_data = make_json_report(
        filename=res.get("filename", uploaded.name),
        counts=counts,
        video_info=res.get("video_info", info),
        processing_time=res.get("processing_time", 0),
        frame_skip=res.get("frame_skip", frame_skip),
    )
    st.download_button(
        label="⬇️ Download JSON Report",
        data=json_data,
        file_name=f"analysis_report_{uploaded.name.rsplit('.', 1)[0]}.json",
        mime="application/json",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<center><small>AI Traffic Vehicle Counter · YOLO + ByteTrack · "
    "Built with Streamlit & Ultralytics</small></center>",
    unsafe_allow_html=True,
)

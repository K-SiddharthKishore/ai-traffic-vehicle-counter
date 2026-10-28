"""
AI Traffic Vehicle Counter — Streamlit Web Application
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Traffic Vehicle Counter",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}
.hero h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 1rem 0; }
.stat-card {
    flex: 1; min-width: 120px;
    background: #1A1D27;
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(108,99,255,0.2); }
.stat-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.stat-value { font-size: 2rem; font-weight: 700; color: #6C63FF; line-height: 1; }
.stat-label { font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }

/* ── Step badges ── */
.step-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem; font-weight: 600; color: #A78BFA;
    margin-bottom: 0.5rem;
}

/* ── Info / warn boxes ── */
.info-box {
    background: rgba(59,130,246,0.1);
    border-left: 3px solid #3B82F6;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #93C5FD;
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(251,191,36,0.1);
    border-left: 3px solid #FBBF24;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #FDE68A;
    margin: 0.75rem 0;
}
.success-box {
    background: rgba(16,185,129,0.1);
    border-left: 3px solid #10B981;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #6EE7B7;
    margin: 0.75rem 0;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #E2E8F0;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108,99,255,0.4) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    background: rgba(108,99,255,0.04) !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(108,99,255,0.7) !important;
}

/* ── Progress bar ── */
div[data-testid="stProgress"] > div { height: 6px; border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #12141E !important; }
[data-testid="stSidebar"] .stMarkdown { color: #CBD5E1; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #3B82F6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    height: 3rem !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35) !important;
    transition: opacity .2s, transform .2s !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 2rem;
    color: #475569;
    font-size: 0.82rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.footer a { color: #6C63FF; text-decoration: none; }

/* ── Table styling ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #1A1D27;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚦 AI Traffic Vehicle Counter</h1>
  <p>Upload a traffic video &nbsp;·&nbsp; Detect &amp; track with YOLO + ByteTrack &nbsp;·&nbsp; Count crossings &amp; export results</p>
</div>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading YOLO model…")
def load_yolo_model():
    from src.detector import load_model
    return load_model("yolo11n.pt")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    st.markdown("---")

    st.markdown("**📏 Counting Line**")
    orientation = st.radio(
        "Line orientation",
        ["horizontal", "vertical"],
        index=0,
        help="Horizontal counts vehicles moving up/down; vertical counts left/right.",
    )
    line_pct = st.slider(
        "Position (% of frame)",
        min_value=10, max_value=90, value=50, step=5,
    )

    st.markdown("---")
    st.markdown("**🔧 Processing**")
    frame_skip = st.select_slider(
        "Frame skip interval",
        options=[1, 2, 3, 5, 10],
        value=5,
        help="Process every Nth frame. Higher = faster. Use 5-10 on cloud (CPU only).",
    )
    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.1, max_value=0.9, value=0.30, step=0.05,
    )
    imgsz = st.select_slider(
        "YOLO image size",
        options=[320, 416, 512, 640],
        value=320,
        help="320 is ~4× faster than 640 on CPU with minimal accuracy loss.",
    )

    st.markdown('<div class="warn-box">☁️ <b>Cloud CPU mode.</b> For a 30s video use frame skip 5 + size 320 (~2–3 min). Size 640 will take 10+ min.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🚗 Counted Vehicle Classes**")
    classes = [("🚗", "Car", "COCO class 2"), ("🏍️", "Motorcycle", "class 3"),
               ("🚌", "Bus", "class 5"), ("🚛", "Truck", "class 7")]
    for icon, name, cls in classes:
        st.markdown(f"{icon} **{name}** <span style='color:#64748B;font-size:0.8rem'>({cls})</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#64748B;font-size:0.82rem'>🚶 Pedestrians shown but NOT counted</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:0.75rem;text-align:center">YOLO + ByteTrack · Streamlit</div>', unsafe_allow_html=True)

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 &nbsp;·&nbsp; Upload Video</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop or click to upload a traffic video",
    type=["mp4", "mov", "avi"],
    help="MP4 recommended · max 200 MB · fixed-camera traffic works best",
)

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    ℹ️ Upload a traffic video (MP4 / MOV / AVI) above to begin.
    Results, annotated video, and downloadable reports will appear here.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-icon">🎯</div>
          <div class="stat-value" style="font-size:1.1rem;color:#A78BFA">YOLO</div>
          <div class="stat-label">Object Detection</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-icon">🔍</div>
          <div class="stat-value" style="font-size:1.1rem;color:#A78BFA">ByteTrack</div>
          <div class="stat-label">Multi-Object Tracking</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
          <div class="stat-icon">📊</div>
          <div class="stat-value" style="font-size:1.1rem;color:#A78BFA">CSV + JSON</div>
          <div class="stat-label">Export Reports</div>
        </div>""", unsafe_allow_html=True)

    st.stop()

# ── Save uploaded file ─────────────────────────────────────────────────────────
if "temp_input_path" not in st.session_state or \
        st.session_state.get("uploaded_name") != uploaded.name:
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.read())
    st.session_state["temp_input_path"] = input_path
    st.session_state["uploaded_name"] = uploaded.name
    st.session_state.pop("analysis_results", None)

input_path = st.session_state["temp_input_path"]

# ── Video metadata ─────────────────────────────────────────────────────────────
from src.video_processor import get_video_info

info = get_video_info(input_path)
if info is None:
    st.error("❌ Could not open the uploaded video. Please try a different file.")
    st.stop()

st.markdown('<div class="success-box">✅ Video loaded successfully — configure settings in the sidebar, then start analysis.</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📁 File", uploaded.name[:20] + ("…" if len(uploaded.name) > 20 else ""))
col2.metric("🖼️ Resolution", f"{info['width']}×{info['height']}")
col3.metric("⏱️ Duration", f"{info['duration']:.1f}s")
col4.metric("🎞️ Frame Rate", f"{info['fps']:.1f} fps")

line_position = (
    info["height"] * line_pct / 100 if orientation == "horizontal"
    else info["width"] * line_pct / 100
)

# ── Step 2: Analyze ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-badge">Step 2 &nbsp;·&nbsp; Run Analysis</div>', unsafe_allow_html=True)

start_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
if start_btn:
    st.session_state.pop("analysis_results", None)

if start_btn and "analysis_results" not in st.session_state:
    try:
        model = load_yolo_model()
    except Exception as exc:
        st.error(f"❌ Failed to load YOLO model: {exc}")
        st.stop()

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / f"annotated_{uploaded.name.rsplit('.', 1)[0]}.mp4")

    st.markdown('<div class="section-header">⏳ Processing Video…</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_col, elapsed_col, fps_col = st.columns(3)
    status_ph = status_col.empty()
    elapsed_ph = elapsed_col.empty()
    fps_ph = fps_col.empty()
    t0 = time.time()

    from src.detector import get_device
    device = get_device()
    status_ph.metric("Device", device.upper())

    def on_progress(frame_idx, total, counts):
        pct = int(frame_idx / total * 100) if total > 0 else 0
        progress_bar.progress(min(pct, 100))
        elapsed = time.time() - t0
        elapsed_ph.metric("Elapsed", f"{elapsed:.1f}s")
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        fps_ph.metric("Speed", f"{proc_fps:.1f} fps")

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
        status_ph.metric("Status", "Done ✅")
    except Exception as exc:
        st.error(f"❌ Processing failed: {exc}")
        logger.exception("Processing error")
        st.stop()

if "analysis_results" not in st.session_state:
    st.stop()

# ── Step 3: Results ────────────────────────────────────────────────────────────
res = st.session_state["analysis_results"]
counts = res["counts"]
total = res["total_vehicles"]

from src.detector import VEHICLE_CLASSES

st.markdown("---")
st.markdown('<div class="step-badge">Step 3 &nbsp;·&nbsp; Results</div>', unsafe_allow_html=True)

# Stat cards
icons = {2: "🚗", 3: "🏍️", 5: "🚌", 7: "🚛"}
cards_html = f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-icon">🚦</div>
    <div class="stat-value">{total}</div>
    <div class="stat-label">Total Vehicles</div>
  </div>
  {''.join(f'''
  <div class="stat-card">
    <div class="stat-icon">{icons[cid]}</div>
    <div class="stat-value">{counts.get(cid, 0)}</div>
    <div class="stat-label">{name}</div>
  </div>''' for cid, name in VEHICLE_CLASSES.items())}
  <div class="stat-card">
    <div class="stat-icon">⏱️</div>
    <div class="stat-value" style="font-size:1.5rem">{res['processing_time']:.1f}s</div>
    <div class="stat-label">Process Time</div>
  </div>
</div>
"""
st.markdown(cards_html, unsafe_allow_html=True)

# Chart + Table
chart_col, table_col = st.columns([1, 1])

with chart_col:
    st.markdown('<div class="section-header">📈 Distribution</div>', unsafe_allow_html=True)
    if total > 0:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        labels = [VEHICLE_CLASSES[cid] for cid in VEHICLE_CLASSES]
        values = [counts.get(cid, 0) for cid in VEHICLE_CLASSES]
        colors = ["#6C63FF", "#3B82F6", "#06B6D4", "#10B981"]

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1A1D27")
        ax.set_facecolor("#1A1D27")

        wedges, texts, autotexts = ax.pie(
            [v if v > 0 else 0.001 for v in values],
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
            colors=colors,
            startangle=140,
            wedgeprops=dict(width=0.6, edgecolor="#12141E", linewidth=2),
            textprops=dict(color="white"),
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")

        legend_patches = [mpatches.Patch(color=c, label=f"{l} ({v})")
                         for c, l, v in zip(colors, labels, values)]
        ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2, frameon=False, fontsize=9,
                  labelcolor="white")
        ax.set_title("Vehicle Category Split", color="white", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown('<div class="info-box">No vehicles crossed the counting line.</div>', unsafe_allow_html=True)

with table_col:
    st.markdown('<div class="section-header">📋 Breakdown</div>', unsafe_allow_html=True)
    rows = []
    for cid, name in VEHICLE_CLASSES.items():
        cnt = counts.get(cid, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        rows.append({"Category": f"{icons[cid]} {name}", "Count": cnt, "Share": f"{pct:.1f}%"})
    rows.append({"Category": "**🚦 Total**", "Count": total, "Share": "100.0%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">⚙️ Processing Stats</div>', unsafe_allow_html=True)
    stats = [
        {"Metric": "Frames Processed", "Value": str(res.get("frames_processed", "?"))},
        {"Metric": "Processing Speed", "Value": f"{res.get('fps_processed', 0):.1f} fps"},
        {"Metric": "Video FPS", "Value": f"{res.get('video_fps', 0):.1f}"},
        {"Metric": "Frame Skip", "Value": f"1 of every {res.get('frame_skip', 1)}"},
        {"Metric": "Resolution", "Value": f"{res.get('width','?')}×{res.get('height','?')}"},
    ]
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

# Timeline chart
crossing_log = res.get("crossing_log", [])
if crossing_log:
    st.markdown('<div class="section-header">📊 Crossings Over Time</div>', unsafe_allow_html=True)
    log_df = pd.DataFrame(crossing_log)
    log_df["vehicle"] = log_df["class_id"].map(VEHICLE_CLASSES)
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
    st.bar_chart(timeline, color=["#6C63FF", "#3B82F6", "#06B6D4", "#10B981"][:len(timeline.columns)])

st.markdown("""
<div class="info-box">
ℹ️ <strong>How it works:</strong> Each tracked vehicle is counted exactly once when its
centre point crosses the virtual line. Vehicles that never reach the line are not counted.
</div>
""", unsafe_allow_html=True)

# ── Step 4: Download ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="step-badge">Step 4 &nbsp;·&nbsp; Download Outputs</div>', unsafe_allow_html=True)

output_path = res.get("output_path", "")
if output_path and os.path.exists(output_path):
    file_size_mb = os.path.getsize(output_path) / 1_048_576
    st.markdown(f'<div class="section-header">🎬 Annotated Video <span style="color:#64748B;font-size:0.8rem;font-weight:400">— {os.path.basename(output_path)} ({file_size_mb:.1f} MB)</span></div>', unsafe_allow_html=True)
    try:
        with open(output_path, "rb") as vf:
            video_bytes = vf.read()
        st.video(video_bytes)
        dl_v_col, dl_c_col, dl_j_col = st.columns(3)
        with dl_v_col:
            st.download_button(
                "⬇️ Download Video",
                data=video_bytes,
                file_name=os.path.basename(output_path),
                mime="video/mp4",
                use_container_width=True,
            )
    except Exception:
        dl_v_col, dl_c_col, dl_j_col = st.columns(3)
        with dl_v_col:
            with open(output_path, "rb") as vf:
                st.download_button("⬇️ Download Video", data=vf,
                                   file_name=os.path.basename(output_path),
                                   mime="video/mp4", use_container_width=True)
else:
    dl_v_col, dl_c_col, dl_j_col = st.columns(3)
    st.warning("Output video not found — the writer may have failed for this codec.")

from src.report_generator import make_csv, make_json_report

try:
    dl_c_col
except NameError:
    _, dl_c_col, dl_j_col = st.columns(3)

with dl_c_col:
    csv_data = make_csv(counts)
    st.download_button(
        "⬇️ Download CSV",
        data=csv_data,
        file_name=f"vehicle_counts_{uploaded.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl_j_col:
    json_data = make_json_report(
        filename=res.get("filename", uploaded.name),
        counts=counts,
        video_info=res.get("video_info", info),
        processing_time=res.get("processing_time", 0),
        frame_skip=res.get("frame_skip", frame_skip),
    )
    st.download_button(
        "⬇️ Download JSON",
        data=json_data,
        file_name=f"analysis_report_{uploaded.name.rsplit('.', 1)[0]}.json",
        mime="application/json",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🚦 AI Traffic Vehicle Counter &nbsp;·&nbsp;
  YOLO + ByteTrack &nbsp;·&nbsp;
  Built with <a href="https://streamlit.io">Streamlit</a> &amp;
  <a href="https://ultralytics.com">Ultralytics</a>
</div>
""", unsafe_allow_html=True)

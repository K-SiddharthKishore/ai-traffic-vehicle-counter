# 🚦 AI Traffic Vehicle Counter Using YOLO

> **Mini-Project · AIML Department**
> Detect, track, and count vehicles in traffic videos using a pretrained YOLO model.

---

## Problem Statement

Manual traffic counting is time-consuming, expensive, and prone to human error. This project automates vehicle counting using deep learning — a YOLO object detection model combined with ByteTrack multi-object tracking — to count vehicles crossing a virtual line in any traffic video.

---

## Objectives

1. Detect vehicles (cars, motorcycles, buses, trucks) in every frame using YOLO.
2. Assign persistent track IDs to vehicles across frames with ByteTrack.
3. Count each vehicle exactly once as it crosses a configurable virtual counting line.
4. Display a live dashboard with counts, charts, and an annotated output video.
5. Export results as CSV and JSON reports.

---

## Features

| Feature | Details |
|---|---|
| YOLO Detection | Pretrained `yolo11n` or `yolov8n` (auto-downloads) |
| ByteTrack Tracking | Stable track IDs across frames |
| Virtual Counting Line | Horizontal or vertical, position adjustable |
| Four Vehicle Categories | Car, Motorcycle, Bus, Truck |
| Web UI | Streamlit — upload, configure, view results |
| Annotated Video | Bounding boxes, labels, track IDs, live count overlay |
| Export | CSV category totals + JSON analysis report |
| Speed Controls | Frame-skip slider, image-size selector |
| GPU Support | Auto-detects CUDA/MPS; falls back to CPU |

---

## Technology Stack

- **Python 3.11+**
- **Streamlit** — Web interface
- **Ultralytics YOLO** — Object detection (yolo11n / yolov8n)
- **OpenCV** — Video I/O and annotation drawing
- **ByteTrack** — Multi-object tracking (built into Ultralytics)
- **Pandas** — Results tables
- **Matplotlib** — Pie chart visualisation

---

## System Workflow

```
User uploads MP4/MOV/AVI video
         │
         ▼
┌─────────────────────┐
│  Video reader (OpenCV) reads frame-by-frame
└─────────────────────┘
         │  every Nth frame (configurable)
         ▼
┌─────────────────────────────────────────────┐
│  YOLO model detects vehicles + assigns      │
│  track IDs via ByteTrack                    │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  LineCounter checks if centre of each       │
│  track crossed the virtual line this frame  │
│  (previous position → current position)     │
│  → adds to count only once per track ID     │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Draw bounding boxes, labels, track IDs,    │
│  counting line, and live count overlay      │
│  → write annotated frame to output video    │
└─────────────────────────────────────────────┘
         │
         ▼
  Results dashboard: metrics, chart, table,
  annotated video preview, CSV/JSON download
```

---

## Project Structure

```
traffic_vehicle_counter/
├── app.py                  # Streamlit application (main entry point)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── detector.py         # YOLO model loader + per-frame detection
│   ├── tracker.py          # Line-crossing counter logic
│   ├── video_processor.py  # Full pipeline: read → detect → annotate → write
│   └── report_generator.py # CSV and JSON export
├── outputs/                # Annotated output videos saved here
│   └── .gitkeep
└── sample_videos/
    └── README.md           # Tips for getting test videos
```

---

## Installation

### 1. Clone / download the project

```bash
cd "traffic_vehicle_counter"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> The YOLO model weights (~6 MB for yolo11n) are downloaded automatically on first run.
> You need an internet connection the first time only.

---

## Running the Application

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**.

---

## How to Use

1. **Upload** a traffic video (MP4, MOV, or AVI).
2. **Configure** the counting line (position and orientation) in the sidebar.
3. Optionally adjust frame-skip and confidence threshold for speed/accuracy trade-off.
4. Click **Start Analysis**.
5. Watch the progress bar while the video is processed.
6. View the **Results Dashboard** — metric cards, pie chart, data table.
7. **Preview** the annotated output video.
8. **Download** the annotated video, CSV report, or JSON report.

---

## How Vehicle Tracking and Line Crossing Work

### Detection
YOLO (You Only Look Once) is a real-time object detection neural network.
Each frame is passed through the network and it outputs bounding boxes, class labels, and confidence scores for all detected objects.

### Tracking (ByteTrack)
ByteTrack assigns a unique **Track ID** to each vehicle. It links detections across frames using the Kalman filter for motion prediction and IoU (Intersection over Union) for matching, keeping the same ID even when a vehicle is briefly occluded.

### Counting
A virtual counting line is drawn across the frame.
For each tracked vehicle, we record its **previous centre position** and **current centre position**.
If the centre has moved from one side of the line to the other (e.g., y < line_y in the previous frame and y ≥ line_y now), that vehicle has crossed and is counted.
Each Track ID is counted **at most once**, so a slow vehicle sitting near the line is not double-counted.

---

## Limitations

- Counting accuracy depends on camera angle; overhead/side views work best.
- Frame-skipping can miss very fast vehicles.
- Re-identification across reappearances is not implemented; a vehicle leaving and re-entering the frame may get a new Track ID.
- Heavily occluded or overlapping vehicles may be merged into one track.
- Accuracy on night-time or very low-resolution footage is lower.
- Works only with the four COCO classes listed above.

---

## Example Screenshots

*(Add screenshots of the Streamlit dashboard and annotated video here for your submission.)*

---

## Future Enhancements

- Speed estimation (km/h) using homography transformation
- Number plate recognition with an OCR model
- Direction-aware counting (inbound vs outbound)
- Real-time webcam or RTSP stream support
- Multi-camera dashboard
- Cloud deployment (Streamlit Community Cloud)

---

## Project Abstract

This project implements an automated traffic vehicle counter using the YOLO (You Only Look Once) deep learning object detection model combined with ByteTrack multi-object tracking. A pretrained YOLO model detects four vehicle categories — cars, motorcycles, buses, and trucks — in each frame of an uploaded traffic video. ByteTrack assigns stable track IDs to each vehicle across frames, and a custom line-crossing algorithm counts each unique vehicle exactly once as its detected centre point crosses a configurable virtual counting line. The results are displayed in an interactive Streamlit web dashboard with metric cards, a distribution chart, an annotated video preview, and downloadable CSV/JSON reports. The system runs locally on CPU without any cloud API dependencies.

---

## Algorithm Explanation (for Viva)

1. **Input:** A traffic video is loaded frame by frame using OpenCV.
2. **Detection:** Each frame (or every Nth frame for speed) is passed to the YOLO neural network. YOLO divides the image into a grid and predicts bounding boxes and class probabilities for all cells simultaneously — that is why it is fast ("only once").
3. **Tracking:** ByteTrack uses a Kalman filter to predict where each vehicle will be in the next frame, then matches predictions with new detections using IoU. This creates stable track IDs that persist across frames.
4. **Counting:** For each active track, we check whether its centre has crossed the user-configured line since the last frame. If it has, and the track has not been counted yet, we increment the category counter and record the event.
5. **Output:** An annotated video is written with bounding boxes, labels, track IDs, the counting line, and a live count overlay. Results are exported as CSV and JSON.

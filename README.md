# AI-Powered Video Surveillance System

> Real-time video analytics dashboard with AI-powered threat detection for public safety monitoring.

A full-stack surveillance platform that combines **YOLOv8 object detection**, **ByteTrack multi-object tracking**, and **X3D violence classification** into a single, live-streaming web application — built with a FastAPI backend and a Next.js dashboard.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Using the Dashboard](#using-the-dashboard)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## Features

| Capability | Description |
|---|---|
| **Live Webcam Streaming** | MJPEG stream with real-time bounding-box overlays via your device's webcam |
| **Video File Analysis** | Upload `.mp4`, `.avi`, or `.mov` files and process them frame-by-frame |
| **RTSP Camera Support** | Connect to IP cameras using any RTSP stream URL |
| **Object Detection** | YOLOv8 detects persons, vehicles (car, truck, bus, motorcycle, bicycle), and weapons (knife, scissors, baseball bat) |
| **Accident Detection** | ByteTrack-powered vehicle trajectory analysis flags collision events in real time |
| **Violence Classification** | X3D video classifier identifies violent activity across rolling frame windows |
| **Real-time Alerts** | Threat events are pushed instantly to the dashboard via WebSocket |
| **Alert Acknowledgment** | Operators can acknowledge and dismiss alerts from the UI |
| **Live Statistics** | Running counts of detections, active alert types, and severity breakdown |
| **Swagger / ReDoc API Docs** | Auto-generated interactive API documentation at `/docs` and `/redoc` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard (Next.js)                   │
│  LiveStream ─── VideoUpload ─── AlertsPanel ─── Stats   │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                          │
│                                                          │
│  /api/cameras  ─── Camera Service                        │
│  /api/alerts   ─── Alert Service ──────► WebSocket (/ws) │
│  /api/incidents ── Incident Service                      │
│  /api/stream/  ─── Stream Service                        │
│                         │                                │
│              ┌──────────▼──────────┐                     │
│              │    AI Pipeline      │                     │
│              │  ┌───────────────┐  │                     │
│              │  │ YOLOv8 (nano) │  │ Detection           │
│              │  └───────────────┘  │                     │
│              │  ┌───────────────┐  │                     │
│              │  │  ByteTrack    │  │ Tracking            │
│              │  └───────────────┘  │                     │
│              │  ┌───────────────┐  │                     │
│              │  │ X3D Classifier│  │ Violence            │
│              │  └───────────────┘  │                     │
│              │  ┌───────────────┐  │                     │
│              │  │Accident Engine│  │ Trajectory          │
│              │  └───────────────┘  │                     │
│              └─────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai-surveillance/
├── backend/
│   ├── app/
│   │   ├── ai/
│   │   │   ├── classifiers/
│   │   │   │   └── violence_classifier.py   # X3D violence detection
│   │   │   ├── detectors/
│   │   │   │   ├── yolo_detector.py         # YOLOv8 + ByteTrack detection
│   │   │   │   └── accident_detector.py     # Vehicle trajectory analysis
│   │   │   └── pipelines/
│   │   │       └── video_pipeline.py        # Frame processing orchestration
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── cameras.py               # Camera management endpoints
│   │   │       ├── alerts.py                # Alert CRUD endpoints
│   │   │       ├── incidents.py             # Incident reporting endpoints
│   │   │       └── health.py               # Health check endpoint
│   │   ├── core/
│   │   │   └── config.py                   # Pydantic settings
│   │   ├── services/
│   │   │   ├── alert_service.py            # Alert business logic
│   │   │   ├── camera_service.py           # Camera session management
│   │   │   ├── incident_service.py         # Incident management
│   │   │   ├── stream_service.py           # MJPEG stream generator
│   │   │   └── websocket_service.py        # WebSocket connection manager
│   │   └── main.py                         # FastAPI app entry point
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                       # Root layout + font
│   │   ├── page.tsx                         # Main dashboard page
│   │   └── globals.css                      # Global styles
│   ├── components/
│   │   ├── LiveStream.tsx                   # Webcam / RTSP stream viewer
│   │   ├── VideoUpload.tsx                  # File upload + analysis UI
│   │   ├── AlertsPanel.tsx                  # Real-time alerts list
│   │   ├── StatsCards.tsx                  # Summary statistics cards
│   │   └── Navbar.tsx                       # Top navigation bar
│   └── Dockerfile
├── configs/                                  # YAML configuration files
├── models/                                   # ML model weights (.pt files)
├── docs/                                     # Extended documentation
│   ├── PRD.md                               # Product requirements
│   ├── TECH_STACK.md                        # Tech stack details
│   └── TODO.md                              # Development roadmap
├── uploads/                                  # Uploaded video files (auto-created)
├── output/                                   # Processed output videos (auto-created)
├── storage/                                  # Saved frames and clips (auto-created)
├── tests/                                    # Test suite
├── .env.example                              # Environment variable template
├── docker-compose.yml                        # Multi-container Docker setup
└── requirements.txt                          # Python dependencies
```

---

## Prerequisites

Make sure the following are installed on your machine before proceeding:

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.9 – 3.11 | Backend runtime |
| **Node.js** | 18+ | Frontend runtime |
| **npm** | 9+ | Frontend package manager |
| **Git** | any | Clone the repository |
| **CUDA Toolkit** *(optional)* | 11.8+ | GPU-accelerated inference |

> **CPU mode works fine.** CUDA is recommended only if you need high-throughput real-time performance. The system automatically detects the best available device (`cuda` → `cpu`).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ashishvishwakarma0406/ai-surveillance.git
cd ai-surveillance
```

### 2. Set Up the Python Backend

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** `requirements.txt` installs the CPU build of PyTorch by default. For GPU support, install the correct CUDA-specific build from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`.

### 3. Set Up the Frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment Variables

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Open `.env` and adjust any values to match your environment (see [Configuration](#configuration) for details).

---

## Configuration

All settings are controlled through the `.env` file in the project root.

```env
# API server binding
API_HOST=0.0.0.0
API_PORT=8000

# Allowed origins for CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# File storage paths (created automatically on first run)
UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
STORAGE_DIR=./storage

# Maximum allowed video upload size
MAX_UPLOAD_SIZE_MB=500

# YOLOv8 model file (placed in the models/ folder)
YOLO_MODEL=yolov8n.pt

# Detection confidence threshold (0.0 – 1.0)
YOLO_CONFIDENCE=0.5

# Violence classification threshold (0.0 – 1.0)
VIOLENCE_THRESHOLD=0.6

# Frontend → Backend URL (used by Next.js at runtime)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### ML Models

The **YOLOv8n** model (`yolov8n.pt`) is downloaded automatically by `ultralytics` on first run if it is not found locally. You can also pre-download a specific variant:

```bash
# Download inside the models/ directory to avoid re-downloading
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

Available model sizes (larger = more accurate, slower):

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 50 MB | Medium | Best |

Update `YOLO_MODEL=yolov8s.pt` in `.env` to switch models.

---

## Running the Application

Open **two separate terminals** from the project root.

### Terminal 1 — Backend

```bash
# Activate virtual environment first
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

python -m uvicorn backend.app.main:app --reload --port 8000
```

Expected output:
```
Starting AI Surveillance API...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Expected output:
```
▲ Next.js 14.1.0
- Local:        http://localhost:3000
- Ready in Xs
```

### Access Points

| Service | URL |
|---|---|
| **Dashboard** | [http://localhost:3000](http://localhost:3000) |
| **Swagger API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **ReDoc API Docs** | [http://localhost:8000/redoc](http://localhost:8000/redoc) |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) |

---

## Using the Dashboard

### Live Stream (Webcam / RTSP)

1. Navigate to the **Live** tab on the dashboard.
2. Click **Start Stream** to activate your webcam.
3. Detected objects will appear as labelled bounding boxes on the video.
4. Any threat (weapon, violence, accident) triggers a real-time alert in the **Alerts** panel.

To use an **RTSP camera** instead, enter the stream URL (e.g., `rtsp://192.168.1.10:554/stream`) in the source field before starting.

### Video File Analysis

1. Switch to the **Upload** tab.
2. Drag and drop (or browse for) a `.mp4`, `.avi`, or `.mov` file (max 500 MB by default).
3. Click **Analyze Video** to start processing.
4. Progress is reported live; detected events appear in the Alerts panel as they are discovered.

### Alert Management

- The **Alerts** panel auto-updates over WebSocket — no page refresh needed.
- Click **Acknowledge** on any alert to mark it as reviewed.
- Filter alerts by **severity** (critical / high / medium / low) or **type** (weapon, violence, accident, person).

---

## API Reference

Full interactive docs are available at `/docs`. Below is a quick summary of the main endpoints.

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns service status |

### Cameras

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/cameras` | List all active camera sessions |
| `POST` | `/api/cameras` | Register a new camera source |
| `DELETE` | `/api/cameras/{id}` | Stop and remove a camera session |

### Alerts

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/alerts` | List alerts (supports `severity`, `type`, `acknowledged` filters) |
| `POST` | `/api/alerts/{id}/acknowledge` | Acknowledge an alert |
| `DELETE` | `/api/alerts` | Clear all alerts |

### Incidents

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/incidents` | List recorded incidents |
| `GET` | `/api/incidents/{id}` | Get a single incident |

### Streaming

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/stream/video_feed` | MJPEG stream (query params: `source`, `camera_id`) |

### WebSocket

Connect to `ws://localhost:8000/ws` to receive push events:

```json
// Incoming alert event
{
  "type": "alert",
  "data": {
    "id": 1,
    "alert_type": "weapon",
    "severity": "critical",
    "message": "Knife detected with 87% confidence",
    "confidence": 0.87,
    "timestamp": "2026-03-31T00:00:00"
  }
}
```

---

## Docker Deployment

Docker Compose bundles both services into a single command.

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- For GPU support inside Docker, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) must be installed.

### Start All Services

```bash
# CPU-only (remove the `deploy` GPU section from docker-compose.yml if no GPU)
docker-compose up -d
```

### Stop All Services

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f backend
docker-compose logs -f frontend
```

> **Note:** Uploaded files and processed outputs are persisted in the `./uploads`, `./output`, and `./storage` host directories via Docker volume mounts, so your data survives container restarts.

---

## Troubleshooting

### `ModuleNotFoundError` for any package

Ensure your virtual environment is activated before running the backend:

```bash
venv\Scripts\activate    # Windows
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

### YOLO model not found

The model auto-downloads on first run. If it fails due to network issues, download it manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

Then move `yolov8n.pt` to the `models/` folder.

### Webcam stream not working

- Ensure no other application is using the camera.
- Check browser permissions: the dashboard must be served over `http://localhost` (not a remote URL) for webcam access.
- On Windows, verify the camera is not disabled in Device Manager.

### Alerts not appearing on the dashboard

- Confirm the backend is running and reachable at `http://localhost:8000/health`.
- Open browser DevTools → Network → WS and verify the WebSocket connection to `ws://localhost:8000/ws` is established.
- Check that `NEXT_PUBLIC_API_URL=http://localhost:8000` is set correctly in your `.env` file.

### High CPU / slow inference

Switch to a lighter model by setting `YOLO_MODEL=yolov8n.pt` in `.env` (the nano model is used by default). Alternatively, install a CUDA-enabled PyTorch build for GPU acceleration.

---

## Tech Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** — Async Python web framework
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/)** — Object detection (nano/small/medium variants)
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** — Multi-object tracking (bundled via ultralytics)
- **[PyTorchVideo / X3D](https://pytorchvideo.org/)** — Violence classification
- **[OpenCV](https://opencv.org/)** — Video I/O and frame processing
- **[Pydantic v2](https://docs.pydantic.dev/)** — Data validation and settings

### Frontend
- **[Next.js 14](https://nextjs.org/)** — React framework with App Router
- **[TypeScript](https://www.typescriptlang.org/)** — Type-safe JavaScript
- **[Tailwind CSS](https://tailwindcss.com/)** — Utility-first styling
- **[Axios](https://axios-http.com/)** — HTTP client
- **[socket.io-client](https://socket.io/docs/v4/client-api/)** — WebSocket client
- **[Lucide React](https://lucide.dev/)** — Icon set

### Infrastructure
- **[Docker](https://www.docker.com/)** + **[Docker Compose](https://docs.docker.com/compose/)** — Containerized deployment

---

## License

This project is developed for educational and research purposes.
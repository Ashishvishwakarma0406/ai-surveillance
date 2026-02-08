# AI Surveillance System

Real-time video surveillance with AI-powered detection for violence, weapons, and anomalies.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA GPU (optional, for faster inference)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
ai-surveillance-system/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── main.py         # Entry point
│   │   ├── core/           # Config, security
│   │   ├── api/routes/     # API endpoints
│   │   ├── schemas/        # Pydantic models
│   │   ├── services/       # Business logic
│   │   └── ai/             # ML components
│   │       ├── detectors/  # YOLO detector
│   │       ├── classifiers/# Violence classifier
│   │       └── pipelines/  # Processing pipelines
│   └── requirements.txt
│
├── frontend/               # Next.js Frontend
│   ├── app/               # Pages
│   ├── components/        # React components
│   └── lib/               # API & WebSocket
│
├── infra/                 # Docker & deployment
├── storage/               # Clips, frames, logs
└── models/                # ML model weights
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/cameras` | List cameras |
| POST | `/api/cameras/upload` | Upload video |
| GET | `/api/alerts` | List alerts |
| GET | `/api/stream/video_feed` | MJPEG stream |
| WS | `/ws` | WebSocket for real-time updates |

## 🛠 Tech Stack

**Backend:** FastAPI, PyTorch, YOLOv8, X3D
**Frontend:** Next.js 14, React, TailwindCSS
**Infra:** Docker, PostgreSQL (Phase 2), Redis (Phase 2)

## 📝 License

MIT

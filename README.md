# AI-Powered Video Surveillance System

Real-time video analytics dashboard with AI-powered detection for public safety monitoring.


## 🚀 Features

- **Real-time Video Analysis** - Process webcam, RTSP streams, or video files
- **Object Detection** - YOLOv8-based detection of people, weapons, trash
- **Violence Detection** - X3D-based video classification
- **Web Dashboard** - Modern Next.js UI with real-time alerts
- **REST API** - FastAPI backend with WebSocket support

## 📁 Project Structure

```
ai-surveillance-system/
├── backend/          # FastAPI application
├── frontend/         # Next.js application
├── configs/          # Configuration files
├── models/           # ML model weights
├── tests/            # Test suite
├── docs/             # Documentation
├── docker-compose.yml
├── README.md
└── requirements.txt
```

## 🔧 Quick Start

### Prerequisites
- Python 3.9 - 3.11
- Node.js 18+
- CUDA (optional)

### Installation

```bash
# Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### Running

```bash
# Terminal 1 - Backend
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### Access

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |

## 🐳 Docker

```bash
docker-compose up -d
```

## 📚 Documentation

See [docs/](docs/) for detailed documentation:
- `PRD.md` - Product Requirements
- `TECH_STACK.md` - Technology Stack
- `TODO.md` - Development Roadmap
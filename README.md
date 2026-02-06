# AI-Powered Video Surveillance System

An AI-powered video analytics system that processes live or uploaded video to detect public safety threats and civic issues using **pretrained models only**.

## 🚀 Features

- **Real-time Video Analysis** - Process webcam, RTSP streams, or video files
- **Object Detection** - YOLOv8-based detection of people, weapons, trash, etc.
- **Violence Detection** - X3D-based video classification for fight/assault detection
- **Rule Engine** - Configurable detection rules with temporal consistency
- **Alert System** - Real-time alerts with frame/clip saving
- **OpenCV UI** - Live video display with detection overlays

## 📋 Requirements

- Python 3.9 - 3.11
- CUDA (optional, for GPU acceleration)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   cd ai-surveillance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Basic Usage

```bash
# Run with webcam (default)
python -m src.main

# Run with video file
python -m src.main --source file --path input/video.mp4

# Run with RTSP stream
python -m src.main --source rtsp --url rtsp://username:password@ip:port/stream

# Run without UI (headless)
python -m src.main --headless
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--source`, `-s` | Video source: `webcam`, `file`, `rtsp` |
| `--path`, `-p` | Path to video file |
| `--url`, `-u` | RTSP stream URL |
| `--device`, `-d` | Webcam device index (default: 0) |
| `--config`, `-c` | Path to config file |
| `--headless` | Run without display |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `SPACE` | Pause/Resume |
| `f` | Toggle fullscreen |

## 📁 Project Structure

```
ai-surveillance/
├── src/
│   ├── __init__.py           # Package init
│   ├── main.py               # Main entry point
│   ├── video_input.py        # Video source handling
│   ├── frame_extractor.py    # Frame extraction
│   ├── object_detector.py    # YOLOv8 detection
│   ├── clip_buffer.py        # Temporal buffer
│   ├── violence_detector.py  # X3D violence detection
│   ├── rule_engine.py        # Detection rules
│   ├── alert_manager.py      # Alert handling
│   ├── ui_opencv.py          # OpenCV display
│   └── utils/
│       ├── config_loader.py  # Configuration
│       ├── logger.py         # Logging
│       └── helpers.py        # Utilities
├── configs/
│   ├── main_config.yaml      # Main settings
│   └── rules_config.yaml     # Rule thresholds
├── models/                   # Model weights
├── input/                    # Input videos
├── output/                   # Saved frames/clips
├── logs/                     # Alert logs
├── tests/                    # Test scripts
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Main Configuration (`configs/main_config.yaml`)

```yaml
video:
  source_type: webcam
  frame:
    target_fps: 10
    width: 1280
    height: 720

models:
  object_detection:
    model_name: yolov8n.pt
    confidence_threshold: 0.5
  violence_detection:
    model_name: x3d_m
    violence_threshold: 0.6
```

### Rule Configuration (`configs/rules_config.yaml`)

```yaml
weapon_detection:
  enabled: true
  consecutive_frames_threshold: 3
  min_confidence: 0.6

violence_detection:
  enabled: true
  probability_threshold: 0.65
  require_multiple_persons: true
```

## 🔍 Detection Types

| Type | Trigger | Severity |
|------|---------|----------|
| **Weapon** | Knife/scissors in 3+ frames | 🔴 Critical |
| **Violence** | Fight detected in video clip | 🔴 Critical |
| **Trash** | 2+ litter objects detected | 🟡 Warning |
| **Garbage Overflow** | Trash near garbage bin | 🟡 Warning |
| **Crowd Density** | 20+ persons in frame | 🔵 Info |

## 🧪 Testing

```bash
# Run test suite
python tests/test_pipeline.py
```

## ⚠️ Known Limitations

- Weapon detection is unreliable at long distances
- Violence detection may misclassify sports or play
- Pothole detection is limited with pretrained models
- Requires good lighting conditions

## 📝 License

This project is for educational/research purposes only.

## 🤝 Contributing

Contributions are welcome! Please read the PRD.md for scope guidelines.





# Install dependencies
pip install -r requirements.txt

# Run with webcam
python -m src.main

# Run with video file
python -m src.main --source file --path input/video.mp4

# Run tests
python tests/test_pipeline.py
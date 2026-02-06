# AI-Powered Video Surveillance System - Development To-Do List

> A comprehensive task breakdown for building the MVP surveillance system using pretrained models.

---

## 1. Project Setup & Environment

- [ ] Create project directory structure
  - [ ] Create `src/` folder for main source code
  - [ ] Create `models/` folder for pretrained model weights
  - [ ] Create `utils/` folder for helper functions
  - [ ] Create `configs/` folder for configuration files
  - [ ] Create `output/` folder for saved frames/clips
  - [ ] Create `logs/` folder for alert logs
  - [ ] Create `tests/` folder for test scripts

- [ ] Set up Python virtual environment
  - [ ] Create venv using `python -m venv venv`
  - [ ] Activate virtual environment
  - [ ] Verify Python version (3.9 - 3.11)

- [ ] Create `requirements.txt` with dependencies
  - [ ] Add `ultralytics` for YOLOv8
  - [ ] Add `opencv-python` for video processing
  - [ ] Add `torch` for deep learning
  - [ ] Add `torchvision` for vision models
  - [ ] Add `numpy` for array operations
  - [ ] Add `streamlit` (optional for web UI)
  - [ ] Add `pytorchvideo` for X3D model

- [ ] Install all dependencies
  - [ ] Run `pip install -r requirements.txt`
  - [ ] Verify CUDA availability for GPU support (optional)

---

## 2. Video Input Module

### 2.1 Video Source Handler
- [ ] Create `src/video_input.py` module
- [ ] Implement webcam capture functionality
  - [ ] Initialize webcam using OpenCV `VideoCapture(0)`
  - [ ] Add camera index configuration option
  - [ ] Handle webcam not found error gracefully

- [ ] Implement RTSP stream support
  - [ ] Accept RTSP URL as input parameter
  - [ ] Handle connection timeout errors
  - [ ] Implement reconnection logic for dropped streams

- [ ] Implement video file upload support
  - [ ] Accept file path for MP4/AVI files
  - [ ] Validate file format before processing
  - [ ] Handle file not found errors

- [ ] Add unified video source interface
  - [ ] Create abstract class/interface for video sources
  - [ ] Implement `get_frame()` method
  - [ ] Implement `release()` method for cleanup

### 2.2 Frame Extraction
- [ ] Create `src/frame_extractor.py` module
- [ ] Implement frame extraction logic
  - [ ] Read frames sequentially from video source
  - [ ] Implement configurable frame sampling rate (5-10 FPS default)
  - [ ] Add frame skip logic for performance optimization

- [ ] Implement frame preprocessing
  - [ ] Resize frames to target resolution (720p recommended)
  - [ ] Convert color space if needed (BGR to RGB)
  - [ ] Normalize pixel values for model input

- [ ] Handle edge cases
  - [ ] Gracefully handle dropped frames
  - [ ] Detect and handle end-of-stream condition
  - [ ] Add frame timestamp tracking

---

## 3. Object Detection Module (YOLOv8)

### 3.1 Model Setup
- [ ] Create `src/object_detector.py` module
- [ ] Download pretrained YOLOv8 model
  - [ ] Download `yolov8n.pt` weights (COCO dataset)
  - [ ] Store weights in `models/` directory
  - [ ] Verify model integrity

- [ ] Initialize YOLOv8 model
  - [ ] Load model using Ultralytics API
  - [ ] Configure device (CPU/GPU)
  - [ ] Set inference confidence threshold

### 3.2 Detection Implementation
- [ ] Implement object detection function
  - [ ] Accept preprocessed frame as input
  - [ ] Run inference on frame
  - [ ] Extract detection results

- [ ] Parse detection results
  - [ ] Extract bounding box coordinates
  - [ ] Extract class labels
  - [ ] Extract confidence scores
  - [ ] Filter detections by confidence threshold

- [ ] Implement target class filtering
  - [ ] Filter for persons (COCO class: person)
  - [ ] Filter for weapon-like objects (knife - class 43)
  - [ ] Filter for bottle/vase as trash proxy
  - [ ] Filter for potted plants as garbage bin proxy
  - [ ] Create configurable class mapping

### 3.3 Detection Utilities
- [ ] Create detection result data structure
  - [ ] Define `Detection` class with bbox, label, confidence
  - [ ] Add timestamp to each detection
  - [ ] Add frame index tracking

- [ ] Implement batch detection (optional)
  - [ ] Process multiple frames in batch
  - [ ] Optimize GPU memory usage

---

## 4. Temporal Clip Buffer

### 4.1 Buffer Implementation
- [ ] Create `src/clip_buffer.py` module
- [ ] Implement rolling frame buffer
  - [ ] Use Python `collections.deque` with maxlen=16
  - [ ] Add frames sequentially to buffer
  - [ ] Automatic oldest frame removal when full

- [ ] Add buffer management functions
  - [ ] `add_frame(frame)` - Add new frame to buffer
  - [ ] `get_clip()` - Get all frames as numpy array
  - [ ] `is_full()` - Check if buffer has 16 frames
  - [ ] `clear()` - Reset buffer

### 4.2 Clip Preprocessing
- [ ] Implement clip preprocessing for X3D model
  - [ ] Stack frames into video tensor
  - [ ] Resize frames to model input size
  - [ ] Normalize pixel values
  - [ ] Add temporal dimension

---

## 5. Violence Detection Module (X3D)

### 5.1 Model Setup
- [ ] Create `src/violence_detector.py` module
- [ ] Download pretrained X3D model
  - [ ] Load X3D from PyTorchVideo / Torch Hub
  - [ ] Configure model for inference mode
  - [ ] Map output to violence/non-violence classes

### 5.2 Violence Classification
- [ ] Implement violence detection function
  - [ ] Accept clip tensor from buffer
  - [ ] Run inference on clip
  - [ ] Extract violence probability score

- [ ] Configure classification threshold
  - [ ] Set default threshold (e.g., 0.7)
  - [ ] Make threshold configurable
  - [ ] Return boolean flag + confidence score

### 5.3 Scene-Level Analysis
- [ ] Implement scene-level violence detection
  - [ ] Analyze entire clip, not individual persons
  - [ ] Track violence probability over time
  - [ ] Implement smoothing to reduce false positives

---

## 6. Rule Engine

### 6.1 Rule Engine Core
- [ ] Create `src/rule_engine.py` module
- [ ] Design rule structure
  - [ ] Define `Rule` base class
  - [ ] Implement condition evaluation
  - [ ] Implement action triggering

### 6.2 Implement Detection Rules
- [ ] Weapon detection rule
  - [ ] Track weapon detections across frames
  - [ ] Trigger alert if weapon in ≥3 consecutive frames
  - [ ] Reset counter on non-detection

- [ ] Violence detection rule
  - [ ] Check violence probability against threshold
  - [ ] Trigger alert if above threshold
  - [ ] Add cooldown period to prevent spam

- [ ] Litter/Trash detection rule
  - [ ] Detect trash-like objects near garbage bins
  - [ ] Mark area as "dirty" condition
  - [ ] Generate informational alert

- [ ] Garbage overflow detection rule
  - [ ] Detect garbage bin with surrounding objects
  - [ ] Estimate overflow condition
  - [ ] Generate warning alert

- [ ] Pothole detection rule
  - [ ] Detect road defect patterns
  - [ ] Generate informational alert
  - [ ] Log location/timestamp

### 6.3 Rule Configuration
- [ ] Create `configs/rules_config.yaml`
  - [ ] Define thresholds for each rule
  - [ ] Define cooldown periods
  - [ ] Define alert severity levels

- [ ] Implement rule loading from config
  - [ ] Parse YAML configuration
  - [ ] Instantiate rules dynamically
  - [ ] Validate rule parameters

---

## 7. Alert System

### 7.1 Alert Manager
- [ ] Create `src/alert_manager.py` module
- [ ] Define alert data structure
  - [ ] Alert type (weapon, violence, trash, pothole)
  - [ ] Severity level (informational, warning, critical)
  - [ ] Timestamp
  - [ ] Confidence score
  - [ ] Associated frame/clip

### 7.2 Alert Generation
- [ ] Implement alert creation
  - [ ] Create alert from rule output
  - [ ] Add visual evidence (frame snapshot)
  - [ ] Add metadata (location, camera ID)

- [ ] Implement alert deduplication
  - [ ] Prevent duplicate alerts for same event
  - [ ] Implement cooldown between same-type alerts
  - [ ] Track active alerts

### 7.3 Alert Output
- [ ] Implement console logging
  - [ ] Print formatted alert messages
  - [ ] Color-code by severity
  - [ ] Include timestamp and details

- [ ] Implement file logging
  - [ ] Create `logs/alerts.log` file
  - [ ] Log alerts with full details
  - [ ] Implement log rotation (optional)

- [ ] Implement frame saving
  - [ ] Save flagged frames to `output/frames/`
  - [ ] Name files with timestamp
  - [ ] Include bounding box overlays

- [ ] Implement clip saving (optional)
  - [ ] Save video clips for critical alerts
  - [ ] Store in `output/clips/`
  - [ ] Include 5-second context window

---

## 8. User Interface

### 8.1 OpenCV Display (Default)
- [ ] Create `src/ui_opencv.py` module
- [ ] Implement real-time video display
  - [ ] Create OpenCV window
  - [ ] Display frames at target FPS
  - [ ] Handle window close event

- [ ] Implement visual overlays
  - [ ] Draw bounding boxes on detections
  - [ ] Add class labels with confidence scores
  - [ ] Color-code boxes by detection type
  - [ ] Add alert banners for active alerts

- [ ] Add status panel
  - [ ] Show current FPS
  - [ ] Show detection count
  - [ ] Show active alerts summary

### 8.2 Streamlit Dashboard (Optional)
- [ ] Create `src/ui_streamlit.py` module
- [ ] Implement web-based dashboard
  - [ ] Video feed display
  - [ ] Alert history panel
  - [ ] Configuration controls

- [ ] Add event log viewer
  - [ ] Display recent alerts
  - [ ] Filter by alert type
  - [ ] View associated frames

---

## 9. Main Pipeline Integration

### 9.1 Pipeline Orchestrator
- [ ] Create `src/main.py` entry point
- [ ] Implement main processing loop
  - [ ] Initialize all modules
  - [ ] Start video capture
  - [ ] Process frames continuously
  - [ ] Handle graceful shutdown

### 9.2 Component Integration
- [ ] Connect video input to frame extractor
- [ ] Connect frame extractor to object detector
- [ ] Connect frames to clip buffer
- [ ] Connect clip buffer to violence detector
- [ ] Connect all detections to rule engine
- [ ] Connect rule engine to alert manager
- [ ] Connect all outputs to UI

### 9.3 Configuration Management
- [ ] Create `configs/main_config.yaml`
  - [ ] Video source settings
  - [ ] Model paths and parameters
  - [ ] Detection thresholds
  - [ ] UI settings

- [ ] Implement config loading
  - [ ] Parse YAML configuration
  - [ ] Apply settings to modules
  - [ ] Support command-line overrides

---

## 10. Testing & Validation

### 10.1 Unit Tests
- [ ] Create test files in `tests/` directory
- [ ] Test video input module
  - [ ] Test webcam capture
  - [ ] Test file input
  - [ ] Test error handling

- [ ] Test object detector
  - [ ] Test model loading
  - [ ] Test detection accuracy
  - [ ] Test class filtering

- [ ] Test clip buffer
  - [ ] Test frame addition
  - [ ] Test buffer overflow
  - [ ] Test clip extraction

- [ ] Test rule engine
  - [ ] Test individual rules
  - [ ] Test temporal consistency
  - [ ] Test alert generation

### 10.2 Integration Tests
- [ ] Create end-to-end test script
- [ ] Test full pipeline with sample video
- [ ] Verify alert generation
- [ ] Check output files

### 10.3 Demo Validation
- [ ] Prepare test videos
  - [ ] Video with visible weapon
  - [ ] Video with fighting scene
  - [ ] Video with litter/trash
  - [ ] Video with normal activity (negative case)

- [ ] Run demo scenarios
  - [ ] Verify weapon detection triggers alert
  - [ ] Verify violence detection triggers alert
  - [ ] Verify trash detection triggers alert
  - [ ] Verify no false alerts on normal video

---

## 11. Documentation

- [ ] Update README.md
  - [ ] Add project overview
  - [ ] Add installation instructions
  - [ ] Add usage examples
  - [ ] Add configuration guide

- [ ] Create user guide
  - [ ] Explain system capabilities
  - [ ] Document known limitations
  - [ ] Provide troubleshooting tips

- [ ] Add inline code documentation
  - [ ] Add docstrings to all functions
  - [ ] Add type hints
  - [ ] Add module-level documentation

---

## 12. Performance Optimization (Optional)

- [ ] Profile pipeline performance
  - [ ] Measure frame processing time
  - [ ] Identify bottlenecks
  - [ ] Track memory usage

- [ ] Implement optimizations
  - [ ] Enable GPU acceleration if available
  - [ ] Optimize frame resolution
  - [ ] Implement frame skipping if needed
  - [ ] Use batch inference where possible

- [ ] Verify performance targets
  - [ ] Achieve minimum 5 FPS throughput
  - [ ] Maintain latency under 3 seconds
  - [ ] Ensure stable memory usage

---

## Progress Tracking

| Phase | Status | Completion |
|-------|--------|------------|
| 1. Project Setup | [ ] Not Started | 0% |
| 2. Video Input | [ ] Not Started | 0% |
| 3. Object Detection | [ ] Not Started | 0% |
| 4. Clip Buffer | [ ] Not Started | 0% |
| 5. Violence Detection | [ ] Not Started | 0% |
| 6. Rule Engine | [ ] Not Started | 0% |
| 7. Alert System | [ ] Not Started | 0% |
| 8. User Interface | [ ] Not Started | 0% |
| 9. Pipeline Integration | [ ] Not Started | 0% |
| 10. Testing | [ ] Not Started | 0% |
| 11. Documentation | [ ] Not Started | 0% |
| 12. Optimization | [ ] Not Started | 0% |

---

## Notes

- All development uses **pretrained models only** - no custom training
- Focus on **demonstrability** over production-readiness
- Human review required for all alerts
- No face recognition or identity tracking
- Single-machine deployment target

# Tech Stack — AI-Powered Video Surveillance System (Basic / No-Training)

This document describes the complete technology stack used to build the **basic AI-powered video surveillance system** that operates using **pretrained models only**, without any custom training or fine-tuning.

---

## 1. Programming Language

### Python
- Version: **Python 3.9 – 3.11**
- Reason:
  - Strong ecosystem for computer vision and deep learning
  - Native support for rapid prototyping
  - Extensive pretrained model availability

---

## 2. Computer Vision & Video Processing

### OpenCV
- Library: `opencv-python`
- Purpose:
  - Video capture (webcam, RTSP, files)
  - Frame extraction
  - Frame resizing and preprocessing
  - Drawing bounding boxes and overlays
  - Displaying real-time video output

---

## 3. Deep Learning Framework

### PyTorch
- Libraries:
  - `torch`
  - `torchvision`
- Purpose:
  - Running pretrained deep learning models
  - Model inference (CPU/GPU)
  - Tensor operations

---

## 4. Object Detection

### YOLOv8 (Ultralytics)
- Model Type: **Pretrained object detection**
- Weights: `yolov8n.pt` (COCO dataset)
- Purpose:
  - Detect persons
  - Detect weapon-like objects (knife, gun proxy classes)
  - Detect trash-like objects
  - Detect garbage bins
  - Detect pothole-like structures (limited)

- Why YOLOv8:
  - Fast inference
  - High-quality pretrained weights
  - Simple Python API
  - Suitable for real-time demos

---

## 5. Video / Action Recognition

### Pretrained Video Classification Model (X3D)
- Source: PyTorchVideo / Torch Hub
- Purpose:
  - Classify short video clips as **violent** or **non-violent**
  - Uses temporal information instead of single-frame analysis

- Characteristics:
  - No fine-tuning
  - Scene-level classification
  - Requires fixed-length frame clips

---

## 6. Temporal Processing

### Clip Buffer (Custom Logic)
- Implementation: Python `deque`
- Purpose:
  - Maintain rolling window of recent frames
  - Provide temporal context for violence detection
  - Enable clip-based inference

---

## 7. Rule Engine

### Rule-Based Decision Logic
- Implementation: Custom Python logic
- Purpose:
  - Convert raw model outputs into meaningful alerts
  - Reduce false positives
  - Enforce temporal consistency

- Example Rules:
  - Weapon detected in multiple consecutive frames
  - Violence probability above threshold
  - Trash detected near garbage bin

---

## 8. User Interface

### OpenCV UI (Default)
- Purpose:
  - Display real-time video feed
  - Show bounding boxes and labels
  - Visualize alerts

### Optional: Streamlit
- Purpose:
  - Web-based dashboard
  - Event logs and video playback
  - Demo-friendly interface

---

## 9. Alerting & Logging

### Console-Based Alerts
- Real-time alert messages printed to console

### Local File Storage
- Saved frames or clips for detected events
- Simple log files with timestamps

---

## 10. Environment & Dependency Management

### Virtual Environment
- Tool: `venv`
- Purpose:
  - Isolate project dependencies
  - Avoid system-level conflicts

### Dependency Management
- File: `requirements.txt`
- Includes:
  - `ultralytics`
  - `opencv-python`
  - `torch`
  - `torchvision`
  - `numpy`
  - `streamlit` (optional)

---

## 11. Deployment Environment

### Local Machine (MVP)
- CPU-supported
- GPU-supported (CUDA, if available)
- Single-device deployment

### Intended Usage
- Research prototype
- Academic project
- Hackathon demo
- Proof-of-concept

---

## 12. What Is Explicitly NOT Used

- No model training or fine-tuning
- No face recognition
- No identity tracking
- No cloud services
- No databases
- No message queues (Kafka, RabbitMQ, etc.)
- No microservices architecture

These are intentionally excluded to keep the system simple, explainable, and demo-ready.

---

## 13. Future Stack Extensions (Out of Scope)

- Custom-trained detection models
- Edge deployment (Jetson, TensorRT)
- Cloud inference pipelines
- Scalable backend APIs
- Long-term data storage and analytics

---

## 14. Summary

This tech stack prioritizes:
- Simplicity
- Transparency
- Speed of implementation
- Demonstrability

It is intentionally designed as a **baseline system** that can later be extended into a production-grade solution with trained models and scalable infrastructure.

---

# Product Requirements Document (PRD)

## AI-Powered Video Surveillance & Civic Monitoring System  
### (Basic / No-Training / Pretrained Models Only)

---

## 1. Executive Summary

This document defines the requirements for building a **basic AI-powered video analytics system** that processes live or uploaded video to detect **public safety threats and civic issues** using **only pretrained computer vision and video understanding models**.

The system is designed as a **functional prototype / MVP**, not a production-grade smart city solution.  
There is **no model training**, **no fine-tuning**, and **no custom dataset creation** involved.

---

## 2. Problem Statement

Traditional CCTV systems:

- Passively record video
- Require continuous human monitoring
- Do not scale across large camera networks
- Detect incidents **after** damage has already occurred

Cities and institutions require a **real-time, automated first layer of monitoring** that flags suspicious or abnormal events for **human review**, not automated enforcement.

---

## 3. Product Objectives

### 3.1 Primary Objectives

- Automatically detect **critical anomalies** in video streams
- Reduce dependence on continuous human monitoring
- Provide **actionable alerts** with visual evidence
- Demonstrate the feasibility of AI-assisted surveillance

### 3.2 Secondary Objectives

- Serve as a foundation for future trained / production systems
- Be deployable on a **single local machine**
- Be explainable to **non-technical stakeholders**

---

## 4. Scope Definition

### 4.1 In-Scope (Must-Have)

The system **shall**:

- Accept **live camera feeds** and **uploaded video files**
- Perform **real-time or near-real-time inference**
- Use **only pretrained models**
- Detect:
  - Weapons (guns, knives)
  - Public violence (fights, assaults)
  - Trash / litter presence
  - Overflowing garbage bins
  - Potholes
- Generate:
  - Alerts
  - Bounding boxes
  - Confidence scores
  - Timestamped visual output

---

### 4.2 Out of Scope (Explicitly Excluded)

The system **will not**:

- Identify individuals (no face recognition)
- Attribute actions to specific people (e.g., “who littered”)
- Predict future crimes
- Automate enforcement actions
- Guarantee legal or forensic accuracy

Any request to add these features is a **scope violation**.

---

## 5. Target Users

| User Type                | Needs                             |
|-------------------------|----------------------------------|
| Demo Evaluators / Judges | Clear functionality & explanation |
| Researchers / Students  | Reference implementation          |
| Early Stakeholders      | Proof of feasibility              |
| Internal Teams          | Baseline system for extension     |

---

## 6. User Stories

### US-1: Violence Detection
As an operator, I want to be alerted when physical violence occurs so that I can immediately review the footage.

### US-2: Weapon Detection
As a security officer, I want to know if a weapon is visible in the scene so that I can escalate the situation.

### US-3: Litter / Garbage Monitoring
As a civic authority, I want to identify areas that appear dirty or unmanaged.

### US-4: Infrastructure Damage Detection
As a municipality, I want to visually identify potholes and garbage overflow.

---

## 7. Functional Requirements

### 7.1 Video Input

The system shall support:
- Webcam input
- RTSP camera streams
- Uploaded video files (MP4, AVI)

Constraints:
- Minimum resolution: **720p (recommended)**
- Configurable frame sampling rate (default: **5–10 FPS**)

---

### 7.2 Frame Extraction

- Frames shall be extracted sequentially from the video source
- Frames shall be resized appropriately before inference
- The system must handle:
  - Dropped frames
  - End-of-stream conditions gracefully

---

### 7.3 Object Detection

Using a pretrained object detection model, the system shall detect:
- Persons
- Weapons (knife, gun-like objects)
- Trash-like objects
- Garbage bins
- Road defects (potholes, where detectable)

Per-frame outputs:
- Bounding box coordinates
- Class label
- Confidence score

---

### 7.4 Temporal Clip Buffer

- The system shall maintain a rolling buffer of recent frames
- Default buffer size: **16 frames**
- The buffer shall be used for video-level inference (violence detection)

---

### 7.5 Violence Detection

- A pretrained video classification model shall analyze buffered clips
- The model shall output a **violence probability score**
- Violence detection is **scene-level**, not person-level

---

### 7.6 Rule Engine

The system shall implement a deterministic rule engine that:
- Converts raw ML outputs into alerts
- Applies temporal consistency checks

Example rules:
- Weapon detected in ≥3 consecutive frames → trigger alert
- Violence probability exceeds threshold → trigger alert
- Trash detected near garbage bin → mark area as “dirty”

Rules must be:
- Configurable
- Transparent
- Independent of ML internals

---

### 7.7 Alerts & Output

The system shall:
- Display alerts in real time
- Overlay bounding boxes on video frames
- Log alerts with timestamps
- Optionally save flagged frames or video clips

Alert severity levels:
- Informational
- Warning
- Critical

---

## 8. Non-Functional Requirements

### 8.1 Performance
- Real-time or near-real-time operation
- Acceptable latency: **1–3 seconds**
- Minimum throughput: **5 FPS**

### 8.2 Reliability
- No system crash due to a single faulty video source
- Graceful handling of inference failures

### 8.3 Usability
- Clear visual overlays
- Simple, intuitive UI
- Outputs that are easy to explain and interpret

### 8.4 Privacy
- No biometric identification
- No long-term video retention by default
- Human review required for all alerts

---

## 9. System Architecture

### 9.1 High-Level Pipeline

Video Source
↓
Frame Extractor
↓
YOLOv8 Object Detection
↓
Clip Buffer (Temporal Context)
↓
Violence Classifier
↓
Rule Engine
↓
Alerts + UI



---

## 10. Technology Stack

| Component            | Technology                |
|---------------------|---------------------------|
| Language            | Python                    |
| Video Processing    | OpenCV                    |
| Object Detection    | YOLOv8 (pretrained)       |
| Video Classification| Pretrained X3D or similar |
| UI                  | OpenCV window / Streamlit |
| Runtime             | Local machine             |

---

## 11. Assumptions & Constraints

### Assumptions
- Camera angles are reasonable
- Lighting conditions are not extreme
- Demo environment is controlled

### Constraints
- No labeled data available
- No training budget
- Single-machine deployment

---

## 12. Known Limitations

- Weapon detection is unreliable at long distances
- Violence detection may misclassify sports or play
- Litter detection is limited to object presence
- Pothole detection is highly dependent on camera angle

These limitations are **expected** for a no-training system.

---

## 13. Success Metrics

Success for this MVP is defined as:
- End-to-end pipeline runs without failure
- Alerts are generated for obvious events
- System behavior is explainable
- Observers understand system capabilities and limitations

Accuracy optimization is **out of scope** at this stage.

---

## 14. Risks & Mitigation

| Risk                 | Mitigation                         |
|----------------------|----------------------------------|
| High false positives | Rule-based filtering               |
| Model mismatch       | Clear communication of limitations |
| Over-promising       | Strict scope enforcement           |
| Demo instability     | Use of pre-recorded test videos   |

---

## 15. Future Enhancements (Out of Scope)

- Custom model training
- City-specific datasets
- Edge-device deployment
- Advanced analytics dashboards
- Multi-camera correlation

These belong to **Phase 2 and beyond**.

---

## 16. Final Notes

- This PRD defines a **real, buildable system**
- Claims of production readiness **without training are invalid**
- This MVP exists to:
  - Demonstrate feasibility
  - Attract stakeholder interest
  - Enable future iteration

---

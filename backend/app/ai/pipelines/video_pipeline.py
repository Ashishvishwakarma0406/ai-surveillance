"""
Video Pipeline

End-to-end video processing with detection, classification,
and vehicle accident detection.
"""

import cv2
import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Tuple, Optional, Dict, Any, List
from collections import deque

from backend.app.ai.detectors.yolo_detector import YOLODetector, Detection


class VideoPipeline:
    """
    Video processing pipeline.
    
    Orchestrates object detection, violence classification,
    and alert generation for video files.
    """
    
    def __init__(
        self,
        # Lowered to improve recall for small weapons like pistols
        confidence_threshold: float = 0.35,
        violence_threshold: float = 0.6,
        clip_length: int = 16,
        enable_accident_detection: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.violence_threshold = violence_threshold
        self.clip_length = clip_length
        self.enable_accident_detection = enable_accident_detection
        
        # Components (lazy loaded)
        self.detector = None
        self.violence_classifier = None
        self.accident_detector = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize detection components."""
        if self._initialized:
            return
        
        self.detector = YOLODetector(
            confidence_threshold=self.confidence_threshold
        )
        self.detector.load()
        
        try:
            from backend.app.ai.classifiers.violence_classifier import ViolenceClassifier
            self.violence_classifier = ViolenceClassifier(
                violence_threshold=self.violence_threshold
            )
            self.violence_classifier.load()
        except Exception as e:
            print(f"Violence classifier not available: {e}")
        
        if self.enable_accident_detection:
            try:
                from backend.app.ai.detectors.accident_detector import AccidentDetector
                self.accident_detector = AccidentDetector()
                print("Accident detector initialized for video pipeline")
            except Exception as e:
                print(f"Accident detector not available: {e}")
        
        self._initialized = True
    
    async def process(
        self,
        video_path: str,
        output_path: str = None
    ) -> AsyncGenerator[Tuple[float, Optional[Dict[str, Any]]], None]:
        """
        Process a video file and generate annotated output.
        
        Yields progress updates and final results.
        """
        self._initialize()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration_est = (total_frames / fps) if total_frames > 0 else 0.0
        print(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames ({duration_est:.1f}s)")
        
        # Downscale for speed, but keep enough detail for small weapons (Max width 720)
        target_width = 720
        if width > target_width:
            scale = target_width / width
            width = target_width
            height = int(height * scale)
            print(f"Downscaling to {width}x{height} for performance")
        
        # Setup output video writer
        if output_path is None:
            input_path = Path(video_path)
            output_dir = input_path.parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{input_path.stem}_detected.mp4")
        
        # Try H.264 (avc1) first
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Failed to init avc1 output, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            print(f"Failed to init video writer with mp4v either")
        
        print(f"Output Writer initialized: {width}x{height} @ {fps}fps")
        
        results = {
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration_est
            },
            "output_video": output_path,
            "detections": [],
            "alerts": [],
            "summary": {
                "max_persons": 0,
                "max_weapons": 0,
                "max_vehicles": 0,
                "violence_detected": False,
                "violence_confidence": 0.0,
                "accident_detected": False,
                "accident_confidence": 0.0,
                "accident_count": 0,
                "alert_count": 0
            }
        }
        
        frame_id = 0
        # Process every 3rd frame for speed while keeping tracking viable.
        # ByteTrack Kalman filters can tolerate small gaps.
        process_every_n = 3
        print(f"Processing 1 frame every {process_every_n} frames")
        
        clip_buffer: deque = deque(maxlen=self.clip_length)
        last_detections = []  # Cache detections for frames we don't process
        prev_frame = None     # For optical flow in accident detection
        
        print(f"Starting processing: {video_path}")
        start_time = time.time()
        
        try:
            from starlette.concurrency import run_in_threadpool
            while True:
                # Run OpenCV disk read in threadpool
                result = await run_in_threadpool(cap.read)
                ret, frame = result
                if not ret:
                    break
                
                frame_id += 1
                
                if frame_id % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = frame_id / elapsed if elapsed > 0 else 0
                    tf_display = total_frames if total_frames > 0 else "?"
                    print(f"Progress: {frame_id}/{tf_display} frames ({fps_proc:.1f} fps)")

                if total_frames > 0:
                    progress = (frame_id / total_frames) * 100
                else:
                    # OpenCV often reports 0 frame count for some codecs; avoid div-by-zero.
                    progress = min(95.0, frame_id * 0.25)
                
                # Resize if needed
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Process every nth frame for detection
                is_process_frame = (frame_id == 1 or frame_id % process_every_n == 0)
                if is_process_frame:
                    last_detections = await run_in_threadpool(
                        self._process_frame,
                        frame, frame_id, fps, 
                        results, clip_buffer, prev_frame
                    )
                    prev_frame = frame  # Only copy ref on processed frames
                
                # Draw cached detections on frame
                annotated_frame = self._draw_detections(frame, last_detections)
                
                # Draw accident overlays if active
                if self.accident_detector:
                    active_events = self.accident_detector.get_active_collisions()
                    if active_events:
                        annotated_frame = self.accident_detector.draw_overlays(
                            annotated_frame, active_events
                        )
                
                await run_in_threadpool(out.write, annotated_frame)
                
                # Yield progress every 30 frames to reduce overhead
                if frame_id % 30 == 0:
                    yield progress, results
                    await asyncio.sleep(0)

            if frame_id > 0 and total_frames <= 0:
                results["video_info"]["total_frames"] = frame_id
                results["video_info"]["duration"] = frame_id / fps

            # Final yield with complete results
            yield 100.0, results
            
        finally:
            cap.release()
            out.release()
            print(f"Output video saved: {output_path}")
    
    def _draw_detections(self, frame, detections: List[Detection]):
        """Draw detection boxes on frame."""
        annotated = frame.copy()
        
        def _is_weapon_label(name: str) -> bool:
            if not name:
                return False
            n = name.lower()
            weapon_keywords = [
                "gun", "pistol", "rifle", "revolver", "firearm",
                "knife", "blade", "machete", "scissors",
                "bat", "baseball bat", "stick", "club", "hammer", "axe", "ax"
            ]
            return any(k in n for k in weapon_keywords)
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # Color based on class
            if det.class_name == "person":
                color = (0, 255, 0)  # Green
            elif _is_weapon_label(det.class_name):
                color = (0, 0, 255)  # Red for weapons
            elif det.class_name in ["car", "truck", "bus"]:
                color = (255, 200, 0)  # Cyan for large vehicles
            elif det.class_name in ["motorcycle", "bicycle"]:
                color = (255, 150, 0)  # Light blue for two-wheelers
            else:
                color = (255, 165, 0)  # Orange for others
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def _process_frame(
        self,
        frame,
        frame_id: int,
        fps: float,
        results: dict,
        clip_buffer: deque,
        prev_frame=None,
    ):
        """Process a single frame with detection, classification, and accident detection."""
        timestamp = frame_id / fps
        
        # Object detection WITH tracking (for accident detection)
        detections = []
        tracked_objects = []
        
        if self.detector:
            if self.accident_detector:
                # Use tracking mode to get persistent IDs
                detections, tracked_objects, _ = self.detector.detect_and_track(
                    frame, frame_id
                )
            else:
                # Fallback to standard detection
                detections = self.detector.detect(frame, frame_id)
        
        frame_result = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": []
        }
        
        # Count max concurrent objects in this frame
        person_count = sum(1 for d in detections if d.class_name == "person")
        # Treat a broader set of labels as weapons using keyword matching to
        # capture model-specific synonyms (e.g., pistol/handgun/revolver).
        def _is_weapon_label(name: str) -> bool:
            if not name:
                return False
            n = name.lower()
            weapon_keywords = [
                "gun", "pistol", "rifle", "revolver", "firearm",
                "knife", "blade", "machete", "scissors",
                "bat", "baseball bat", "stick", "club", "hammer", "axe", "ax"
            ]
            return any(k in n for k in weapon_keywords)
        
        weapon_count = sum(1 for d in detections if _is_weapon_label(d.class_name))
        vehicle_count = sum(1 for d in detections if d.class_name in ["car", "motorcycle", "bus", "truck", "bicycle"])
        
        results["summary"]["max_persons"] = max(results["summary"]["max_persons"], person_count)
        results["summary"]["max_weapons"] = max(results["summary"]["max_weapons"], weapon_count)
        results["summary"]["max_vehicles"] = max(results["summary"]["max_vehicles"], vehicle_count)
        
        for det in detections:
            frame_result["objects"].append(det.to_dict())
        
        # --- Contextual Weapon/Danger Alerts ---
        weapon_names = [d.class_name for d in detections if _is_weapon_label(d.class_name)]
        if weapon_names and not any(a["type"] == "weapon" and abs(a["timestamp"] - timestamp) < 2.0 for a in results["alerts"]):
            # Build contextual message: "2 persons, 1 armed with knife"
            unique_weapons = list(set(weapon_names))
            weapon_str = " and ".join(unique_weapons)
            
            if person_count > 0 and len(weapon_names) > 0:
                if person_count == 1:
                    msg = f"Person armed with {weapon_str}"
                elif len(weapon_names) >= person_count:
                    msg = f"{person_count} persons, all armed with {weapon_str}"
                else:
                    msg = f"{person_count} persons, {len(weapon_names)} armed with {weapon_str}"
            else:
                msg = f"Weapon detected: {weapon_str}"
            
            # Use highest weapon confidence
            best_conf = max(d.confidence for d in detections if _is_weapon_label(d.class_name))
            best_weapon = next(d for d in detections if _is_weapon_label(d.class_name))
            
            results["alerts"].append({
                "frame_id": frame_id,
                "timestamp": timestamp,
                "type": "weapon",
                "category": "violence",
                "severity": "critical",
                "message": msg,
                "confidence": best_conf,
                "bbox": best_weapon.bbox,
                "person_count": person_count,
                "weapon_types": unique_weapons,
            })
            results["summary"]["alert_count"] += 1
        
        # NOTE: No "vehicle detected" alerts — vehicles are tracked for collision
        # detection only. Alerts fire only on actual collisions (below).
        
        if frame_result["objects"]:
            results["detections"].append(frame_result)
        
        # Add to clip buffer for violence detection
        clip_buffer.append(frame)
        
        # Violence detection when buffer is full
        if (
            len(clip_buffer) == self.clip_length 
            and self.violence_classifier
            and any(d.class_name == "person" for d in detections)
        ):
            try:
                violence_result = self.violence_classifier.classify(list(clip_buffer))
                
                if violence_result.is_violent:
                    results["summary"]["violence_detected"] = True
                    results["summary"]["violence_confidence"] = max(
                        results["summary"]["violence_confidence"],
                        violence_result.confidence
                    )
                    
                    # Contextual violence message
                    if person_count > 1:
                        violence_msg = f"Fighting detected between {person_count} persons"
                    else:
                        violence_msg = f"Violence detected, {person_count} person(s) involved"
                    
                    results["alerts"].append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "type": "violence",
                        "category": "violence",
                        "severity": "critical",
                        "message": violence_msg,
                        "confidence": violence_result.confidence,
                        "actions": violence_result.top_actions,
                        "person_count": person_count,
                    })
                    results["summary"]["alert_count"] += 1
                    
                    # Emit fine-grained action alerts for punches/slaps/kicks per clip
                    action_keywords = {
                        "punch": "Punch detected",
                        "slap": "Slap detected",
                        "kick": "Kick detected",
                        "headbutt": "Headbutt detected"
                    }
                    for act in violence_result.top_actions:
                        act_name = act.get("action", "").lower()
                        act_prob = float(act.get("probability", 0.0))
                        for key, title in action_keywords.items():
                            if key in act_name and act_prob >= 0.35:
                                results["alerts"].append({
                                    "frame_id": frame_id,
                                    "timestamp": timestamp,
                                    "type": "violence",
                                    "category": "violence",
                                    "severity": "critical",
                                    "message": f"{title} — {person_count} persons in frame",
                                    "confidence": act_prob,
                                    "action": act_name,
                                    "person_count": person_count,
                                })
                                results["summary"]["alert_count"] += 1
                    
                    clip_buffer.clear()  # Reset after detection to allow per-clip alerts
                    
            except Exception as e:
                print(f"Violence check error: {e}")
        
        # --- Accident / Collision Detection ---
        if self.accident_detector and tracked_objects:
            try:
                accident_events = self.accident_detector.update(
                    tracked_objects=tracked_objects,
                    frame=frame,
                    prev_frame=prev_frame,
                    frame_id=frame_id,
                    fps=fps,
                )
                
                for event in accident_events:
                    results["summary"]["accident_detected"] = True
                    results["summary"]["accident_count"] += 1
                    results["summary"]["accident_confidence"] = max(
                        results["summary"]["accident_confidence"],
                        event.confidence
                    )
                    
                    # Deduplicate
                    if not any(
                        a["type"] == "accident"
                        and abs(a["timestamp"] - timestamp) < 3.0
                        for a in results["alerts"]
                    ):
                        results["alerts"].append({
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "type": "accident",
                            "category": "traffic",
                            "severity": event.severity,
                            "message": event.collision_description,
                            "confidence": event.confidence,
                            "bbox": event.collision_zone,
                            "signals": event.signals,
                            "signal_details": event.signal_details,
                            "track_ids": list(event.track_ids),
                            "vehicle_types": list(event.vehicle_types),
                        })
                        results["summary"]["alert_count"] += 1
                        print(
                            f"COLLISION: {event.collision_description} "
                            f"at frame {frame_id} (t={timestamp:.1f}s) "
                            f"conf={event.confidence:.2f}"
                        )
            except Exception as e:
                print(f"Accident detection error: {e}")
        
        return detections

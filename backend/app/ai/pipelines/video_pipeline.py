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
        confidence_threshold: float = 0.5,
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
        
        # Aggressive downscale for speed (Max width 640)
        target_width = 640
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
        # For object tracking (ByteTrack) and violence detection (X3D) to work mathematically,
        # we CANNOT skip large gaps of frames. The Kalman filters lose their targets.
        # Process every 2nd frame (yielding 15fps temporal resolution from a 30fps video).
        process_every_n = 2
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
                if frame_id == 1 or frame_id % process_every_n == 0:
                    last_detections = await run_in_threadpool(
                        self._process_frame,
                        frame, frame_id, fps, 
                        results, clip_buffer, prev_frame
                    )
                
                prev_frame = frame.copy()
                
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
                
                # Yield progress periodically
                if frame_id % 10 == 0:  # Update progress every 10 frames
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
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # Color based on class
            if det.class_name == "person":
                color = (0, 255, 0)  # Green
            elif det.class_name in ["knife", "scissors", "baseball bat"]:
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
        weapon_count = sum(1 for d in detections if d.class_name in ["knife", "scissors", "gun", "pistol", "rifle", "baseball bat"])
        vehicle_count = sum(1 for d in detections if d.class_name in ["car", "motorcycle", "bus", "truck", "bicycle"])
        
        results["summary"]["max_persons"] = max(results["summary"]["max_persons"], person_count)
        results["summary"]["max_weapons"] = max(results["summary"]["max_weapons"], weapon_count)
        results["summary"]["max_vehicles"] = max(results["summary"]["max_vehicles"], vehicle_count)
        
        for det in detections:
            frame_result["objects"].append(det.to_dict())
            
            # Generate weapon alert
            if det.class_name in ["knife", "scissors", "gun", "pistol", "rifle", "baseball bat"]:
                # Check duplicate alerts (simple temporal filter)
                if not any(a["type"] == "weapon" and abs(a["timestamp"] - timestamp) < 2.0 for a in results["alerts"]):
                    results["alerts"].append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "type": "weapon",
                        "severity": "critical",
                        "message": f"Weapon detected: {det.class_name}",
                        "confidence": det.confidence,
                        "bbox": det.bbox
                    })
                    results["summary"]["alert_count"] += 1
            
            # Generate vehicle alert
            elif det.class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
                if not any(a["type"] == "vehicle" and abs(a["timestamp"] - timestamp) < 2.0 for a in results["alerts"]):
                    results["alerts"].append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "type": "vehicle",
                        "severity": "info",
                        "message": f"Vehicle detected: {det.class_name}",
                        "confidence": det.confidence,
                        "bbox": det.bbox
                    })
                    results["summary"]["alert_count"] += 1
        
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
                    
                    results["alerts"].append({
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "type": "violence",
                        "severity": "critical",
                        "message": "Violence detected in video",
                        "confidence": violence_result.confidence,
                        "actions": violence_result.top_actions
                    })
                    results["summary"]["alert_count"] += 1
                    
                    clip_buffer.clear()  # Reset after detection
                    
            except Exception as e:
                print(f"Violence check error: {e}")
        
        # --- Accident Detection ---
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
                            "severity": event.severity,
                            "message": f"Vehicle accident detected ({event.event_id})",
                            "confidence": event.confidence,
                            "bbox": event.collision_zone,
                            "signals": event.signals,
                            "signal_details": event.signal_details,
                            "track_ids": list(event.track_ids),
                        })
                        results["summary"]["alert_count"] += 1
                        print(
                            f"Accident detected at frame {frame_id} "
                            f"(t={timestamp:.1f}s) conf={event.confidence:.2f}"
                        )
            except Exception as e:
                print(f"Accident detection error: {e}")
        
        return detections

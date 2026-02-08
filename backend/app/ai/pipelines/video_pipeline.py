"""
Video Pipeline

End-to-end video processing with detection and classification.
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
        clip_length: int = 16
    ):
        self.confidence_threshold = confidence_threshold
        self.violence_threshold = violence_threshold
        self.clip_length = clip_length
        
        # Components (lazy loaded)
        self.detector = None
        self.violence_classifier = None
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
            print(f"⚠️ Violence classifier not available: {e}")
        
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
        
        print(f"🎬 Video Info: {width}x{height} @ {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")
        
        # Aggressive downscale for speed (Max width 640)
        target_width = 640
        if width > target_width:
            scale = target_width / width
            width = target_width
            height = int(height * scale)
            print(f"📉 Downscaling to {width}x{height} for performance")
        
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
            print(f"⚠️ Failed to init avc1 output, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            print(f"❌ Failed to init video writer with mp4v either")
        
        print(f"🎥 Output Writer initialized: {width}x{height} @ {fps}fps")
        
        results = {
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": total_frames / fps
            },
            "output_video": output_path,
            "detections": [],
            "alerts": [],
            "summary": {
                "max_persons": 0,
                "max_weapons": 0,
                "violence_detected": False,
                "violence_confidence": 0.0,
                "alert_count": 0
            }
        }
        
        frame_id = 0
        # Process approx 2 times per second (e.g. every 15 frames for 30fps)
        process_every_n = max(5, int(fps / 2)) 
        print(f"⚡ Processing 1 frame every {process_every_n} frames")
        
        clip_buffer: deque = deque(maxlen=self.clip_length)
        last_detections = []  # Cache detections for frames we don't process
        
        print(f"🚀 Starting processing: {video_path}")
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                
                if frame_id % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = frame_id / elapsed if elapsed > 0 else 0
                    print(f"⏱️ Progress: {frame_id}/{total_frames} frames ({fps_proc:.1f} fps)")
                
                progress = (frame_id / total_frames) * 100
                
                # Resize if needed
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
                
                # Process every nth frame for detection
                if frame_id == 1 or frame_id % process_every_n == 0:
                    last_detections = await self._process_frame(
                        frame, frame_id, fps, 
                        results, clip_buffer
                    )
                
                # Draw cached detections on frame
                annotated_frame = self._draw_detections(frame, last_detections)
                out.write(annotated_frame)
                
                # Yield progress periodically
                if frame_id % 10 == 0:  # Update progress every 10 frames
                    yield progress, None
                    await asyncio.sleep(0)
            
            # Final yield with complete results
            yield 100.0, results
            
        finally:
            cap.release()
            out.release()
            print(f"✅ Output video saved: {output_path}")
    
    def _draw_detections(self, frame, detections: List[Detection]):
        """Draw detection boxes on frame."""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # Color based on class
            if det.class_name == "person":
                color = (0, 255, 0)  # Green
            elif det.class_name in ["knife", "scissors"]:
                color = (0, 0, 255)  # Red for weapons
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
    
    async def _process_frame(
        self,
        frame,
        frame_id: int,
        fps: float,
        results: dict,
        clip_buffer: deque
    ):
        """Process a single frame."""
        timestamp = frame_id / fps
        
        # Object detection
        detections = self.detector.detect(frame, frame_id) if self.detector else []
        
        frame_result = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": []
        }
        
        # Count max concurrent objects in this frame
        person_count = sum(1 for d in detections if d.class_name == "person")
        weapon_count = sum(1 for d in detections if d.class_name in ["knife", "scissors", "gun", "pistol", "rifle"])
        
        results["summary"]["max_persons"] = max(results["summary"]["max_persons"], person_count)
        results["summary"]["max_weapons"] = max(results["summary"]["max_weapons"], weapon_count)
        
        for det in detections:
            frame_result["objects"].append(det.to_dict())
            
            # Generate weapon alert
            if det.class_name in ["knife", "scissors", "gun", "pistol", "rifle"]:
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
        
        return detections

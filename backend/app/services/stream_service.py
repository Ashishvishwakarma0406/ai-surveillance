"""
Stream Service

Manages real-time video streaming with detections and alert generation.
"""

import cv2
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional, Dict, Any, List
from pathlib import Path

from backend.app.schemas.alert import AlertCreate, AlertType, AlertSeverity


class StreamManager:
    """Manages video streaming with detection overlays and alert generation."""
    
    # Threat class IDs from YOLO COCO dataset
    WEAPON_CLASSES = {43: "knife", 76: "scissors", 38: "baseball bat"}
    PERSON_CLASS_ID = 0
    
    def __init__(self):
        self._streams: Dict[str, Any] = {}
        self._detector = None
        self._detector_loaded = False
        # Alert cooldowns to prevent spam
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldowns = {
            "weapon": timedelta(seconds=5),
            "crowd": timedelta(seconds=30),
        }
        # Detection stats
        self._detection_counts: Dict[str, int] = {}
        self._total_detections = 0
    
    def _load_detector(self):
        """Lazy load detector."""
        if self._detector_loaded:
            return
        
        try:
            from backend.app.ai.detectors.yolo_detector import YOLODetector
            self._detector = YOLODetector()
            self._detector.load()
            self._detector_loaded = True
        except Exception as e:
            print(f"⚠️ Detector not loaded: {e}")
            self._detector_loaded = True
    
    def _can_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_time = self._last_alert_time.get(alert_type)
        if last_time is None:
            return True
        cooldown = self._alert_cooldowns.get(alert_type, timedelta(seconds=5))
        return datetime.now() - last_time > cooldown
    
    async def _process_detections(
        self, 
        detections: List[Any], 
        frame_id: int,
        frame: Any = None
    ):
        """Process detections and generate alerts for threats."""
        from backend.app.services.alert_service import alert_service
        from backend.app.services.websocket_service import ws_manager
        
        if not detections:
            return
        
        person_count = 0
        weapon_detections = []
        
        for det in detections:
            class_id = det.class_id
            class_name = det.class_name
            confidence = det.confidence
            bbox = det.bbox
            
            # Count persons
            if class_id == self.PERSON_CLASS_ID:
                person_count += 1
            
            # Detect weapons
            if class_id in self.WEAPON_CLASSES:
                weapon_detections.append(det)
        
        self._total_detections += len(detections)
        
        # Generate weapon alerts
        for weapon in weapon_detections:
            if self._can_alert("weapon"):
                self._last_alert_time["weapon"] = datetime.now()
                try:
                    alert = AlertCreate(
                        alert_type=AlertType.WEAPON,
                        severity=AlertSeverity.CRITICAL,
                        message=f"WEAPON DETECTED: {weapon.class_name.upper()} with {weapon.confidence:.0%} confidence",
                        confidence=weapon.confidence,
                        frame_id=frame_id,
                        bbox=weapon.bbox,
                        metadata={"class_name": weapon.class_name, "person_count": person_count}
                    )
                    await alert_service.create_alert(alert)
                    print(f"🚨 ALERT: Weapon detected - {weapon.class_name}")
                except Exception as e:
                    print(f"Failed to create weapon alert: {e}")
        
        # Generate crowd density alert
        if person_count >= 10 and self._can_alert("crowd"):
            self._last_alert_time["crowd"] = datetime.now()
            try:
                alert = AlertCreate(
                    alert_type=AlertType.CROWD,
                    severity=AlertSeverity.INFO if person_count < 20 else AlertSeverity.WARNING,
                    message=f"HIGH CROWD DENSITY: {person_count} persons detected",
                    confidence=1.0,
                    frame_id=frame_id,
                    metadata={"person_count": person_count}
                )
                await alert_service.create_alert(alert)
                print(f"📢 ALERT: High crowd density - {person_count} persons")
            except Exception as e:
                print(f"Failed to create crowd alert: {e}")
        
        # Broadcast detection stats periodically (every 30 frames)
        if frame_id % 30 == 0:
            try:
                await ws_manager.broadcast({
                    "type": "detection_stats",
                    "data": {
                        "frame_id": frame_id,
                        "detections_in_frame": len(detections),
                        "person_count": person_count,
                        "total_detections": self._total_detections
                    }
                })
            except Exception:
                pass
    
    async def generate_frames(
        self, 
        source: str = "webcam",
        camera_id: str = None
    ) -> AsyncGenerator[bytes, None]:
        """Generate MJPEG frames with detection and alerting."""
        self._load_detector()
        
        cap = None
        frame_count = 0
        start_time = datetime.now()
        warmup_frames = 5  # Skip first few frames for camera warmup
        
        try:
            if source == "webcam":
                # Try different camera backends for Windows compatibility
                cap = None
                backends = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation"),
                    (cv2.CAP_ANY, "Default")
                ]
                
                for backend, backend_name in backends:
                    for cam_index in [0, 1, 2]:
                        try:
                            print(f"🔍 Trying camera {cam_index} with {backend_name}...")
                            cap = cv2.VideoCapture(cam_index, backend)
                            if cap.isOpened():
                                # Wait a moment for camera to initialize
                                await asyncio.sleep(0.1)
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None:
                                    print(f"✅ Camera opened: index={cam_index}, backend={backend_name}")
                                    break
                            cap.release()
                            cap = None
                        except Exception as e:
                            print(f"Camera {cam_index} with {backend_name} failed: {e}")
                            if cap:
                                cap.release()
                            cap = None
                    if cap and cap.isOpened():
                        break
                
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            else:
                yield self._error_frame("Invalid source type")
                return
            
            if not cap or not cap.isOpened():
                print("❌ Failed to open camera - no available camera found")
                error_msg = "No camera available. Please check:\n1. Camera permissions in Settings\n2. Other apps using camera\n3. Camera drivers installed"
                yield self._error_frame("No camera - check permissions")
                return
            
            # Broadcast stream started
            try:
                from backend.app.services.websocket_service import ws_manager
                await ws_manager.broadcast({
                    "type": "stream_status",
                    "data": {"status": "started", "source": source}
                })
            except Exception:
                pass
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Skip warmup frames
                if frame_count <= warmup_frames:
                    continue
                
                detections = []
                
                # Run detection
                if self._detector:
                    try:
                        detections, annotated = self._detector.detect_with_annotations(
                            frame, frame_count
                        )
                        frame = annotated
                        
                        # Process detections for alerts
                        await self._process_detections(detections, frame_count, frame)
                        
                    except Exception as e:
                        print(f"Detection error: {e}")
                
                # Add overlay
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # FPS overlay
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Detection count overlay
                cv2.putText(
                    frame, f"Detections: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                
                # Encode
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )
                
                await asyncio.sleep(1/30)
                
        finally:
            if cap:
                cap.release()
            # Broadcast stream stopped
            try:
                from backend.app.services.websocket_service import ws_manager
                await ws_manager.broadcast({
                    "type": "stream_status",
                    "data": {"status": "stopped", "source": source}
                })
            except Exception:
                pass
    
    def _error_frame(self, message: str) -> bytes:
        """Create error frame with helpful message."""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add error icon area
        cv2.rectangle(frame, (270, 150), (370, 250), (0, 0, 100), -1)
        cv2.putText(frame, "!", (305, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        # Error message
        cv2.putText(frame, message, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, "Check camera permissions", (120, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        _, buffer = cv2.imencode('.jpg', frame)
        return (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


# Singleton
stream_manager = StreamManager()

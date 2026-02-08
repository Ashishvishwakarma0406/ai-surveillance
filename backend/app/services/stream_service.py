"""
Stream Service

Manages real-time video streaming with detections.
"""

import cv2
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional, Dict, Any
from pathlib import Path


class StreamManager:
    """Manages video streaming with detection overlays."""
    
    def __init__(self):
        self._streams: Dict[str, Any] = {}
        self._detector = None
        self._detector_loaded = False
    
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
    
    async def generate_frames(
        self, 
        source: str = "webcam",
        camera_id: str = None
    ) -> AsyncGenerator[bytes, None]:
        """Generate MJPEG frames."""
        self._load_detector()
        
        cap = None
        frame_count = 0
        start_time = datetime.now()
        
        try:
            if source == "webcam":
                # Try different camera backends for Windows compatibility
                cap = None
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    for cam_index in [0, 1, 2]:
                        try:
                            cap = cv2.VideoCapture(cam_index, backend)
                            if cap.isOpened():
                                ret, test_frame = cap.read()
                                if ret and test_frame is not None:
                                    print(f"✅ Camera opened: index={cam_index}, backend={backend}")
                                    break
                            cap.release()
                            cap = None
                        except Exception as e:
                            print(f"Camera {cam_index} with backend {backend} failed: {e}")
                            if cap:
                                cap.release()
                            cap = None
                    if cap and cap.isOpened():
                        break
                
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            else:
                yield self._error_frame("Invalid source")
                return
            
            if not cap or not cap.isOpened():
                print("❌ Failed to open camera - no available camera found")
                yield self._error_frame("No camera available - check permissions")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                if self._detector:
                    try:
                        detections, annotated = self._detector.detect_with_annotations(
                            frame, frame_count
                        )
                        frame = annotated
                    except Exception:
                        pass
                
                # Add overlay
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
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
    
    def _error_frame(self, message: str) -> bytes:
        """Create error frame."""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        return (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


# Singleton
stream_manager = StreamManager()

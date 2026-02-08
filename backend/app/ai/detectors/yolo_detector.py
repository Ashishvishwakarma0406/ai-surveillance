"""
YOLO Object Detector

YOLOv8-based object detection for surveillance.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# Add project root for model imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox
        }


class YOLODetector:
    """
    YOLOv8 object detector.
    
    Detects persons, weapons, and other objects of interest.
    """
    
    # Classes of interest for surveillance
    TARGET_CLASSES = {
        0: "person",
        43: "knife",
        76: "scissors",
        # Add more as needed
    }
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load the YOLO model."""
        if self._loaded:
            return True
        
        try:
            from ultralytics import YOLO
            
            # Try project models folder first
            model_path = PROJECT_ROOT / "models" / self.model_name
            if not model_path.exists():
                model_path = PROJECT_ROOT / self.model_name
            if not model_path.exists():
                # Download default
                model_path = self.model_name
            
            self.model = YOLO(str(model_path))
            
            # Set device
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._loaded = True
            print(f"✅ YOLO model loaded: {self.model_name} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
            return False
    
    def detect(
        self, 
        frame: np.ndarray,
        frame_id: int = 0
    ) -> List[Detection]:
        """
        Run detection on a frame.
        
        Args:
            frame: BGR image as numpy array
            frame_id: Frame identifier
            
        Returns:
            List of Detection objects
        """
        if not self._loaded:
            self.load()
        
        if self.model is None:
            return []
        
        detections = []
        
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].tolist()
                    
                    # Get class name
                    class_name = self.TARGET_CLASSES.get(
                        class_id,
                        result.names.get(class_id, f"class_{class_id}")
                    )
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox
                    ))
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections
    
    def detect_with_annotations(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> tuple:
        """
        Detect and return annotated frame.
        
        Returns:
            Tuple of (detections, annotated_frame)
        """
        if not self._loaded:
            self.load()
        
        if self.model is None:
            return [], frame
        
        detections = []
        annotated_frame = frame.copy()
        
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].tolist()
                    
                    class_name = self.TARGET_CLASSES.get(
                        class_id,
                        result.names.get(class_id, f"class_{class_id}")
                    )
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox
                    ))
                
                # Get annotated frame
                annotated_frame = result.plot()
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections, annotated_frame

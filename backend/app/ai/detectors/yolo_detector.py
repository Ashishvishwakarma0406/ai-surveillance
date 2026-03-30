"""
YOLO Object Detector

YOLOv8-based object detection and multi-object tracking for surveillance.
Supports vehicle tracking via ByteTrack for accident detection.
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
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        43: "knife",
        76: "scissors",
        38: "baseball bat",
        39: "baseball glove",
    }
    
    # Vehicle class IDs for accident detection
    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
    
    # Weapon class IDs that should trigger alerts
    WEAPON_CLASS_IDS = {43, 76, 38}  # knife, scissors, baseball bat
    
    # Lower confidence threshold for weapons
    WEAPON_CONFIDENCE = 0.35
    PERSON_CONFIDENCE = 0.5
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,  # Lower default for weapons
        device: str = "auto",
        weapon_model_name: Optional[str] = None,  # Optional secondary model for guns
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.weapon_model_name = weapon_model_name
        self.weapon_model = None
        self._loaded = False
        # Use agnostic NMS to allow overlapping detections of different classes
        self.agnostic_nms = True
    
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
            
            # Optionally load a dedicated weapon model to detect guns/pistols/rifles.
            # If weapon_model_name is not provided, we will look for common filenames.
            try:
                from os import getenv
                weapon_name = self.weapon_model_name or getenv("WEAPON_MODEL")
                if weapon_name is None:
                    # Try default filenames in models/
                    for candidate in ["weapons_yolov8s.pt", "weapons_yolov8n.pt", "weapon_yolov8s.pt", "weapons.pt"]:
                        candidate_path = PROJECT_ROOT / "models" / candidate
                        if candidate_path.exists():
                            weapon_name = str(candidate_path)
                            break
                    # If still not found, default to OpenImagesV7 YOLOv8 model which contains Handgun/Shotgun/Rifle explicitly.
                    if weapon_name is None:
                        weapon_name = "yolov8n-oiv7.pt"
                        
                if weapon_name:
                    self.weapon_model = YOLO(str(weapon_name))
                    print(f"✅ Weapon model loaded: {weapon_name}")
                else:
                    print("ℹ️ No dedicated weapon model found (optional). Guns may be missed by COCO.")
            except Exception as e:
                self.weapon_model = None
                print(f"⚠️ Failed to load dedicated weapon model (optional): {e}")
            
            self._loaded = True
            print(f"✅ YOLO model loaded: {self.model_name} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
            return False
    
    @staticmethod
    def _is_weapon_label(name: str) -> bool:
        if not name:
            return False
        n = name.lower()
        weapon_keywords = [
            "gun", "pistol", "rifle", "shotgun", "revolver", "firearm", "handgun",
            "knife", "blade", "sword", "machete", "scissors", "weapon",
            "bat", "baseball bat", "stick", "club", "hammer", "axe", "ax"
        ]
        return any(k in n for k in weapon_keywords)
    
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
                verbose=False,
                agnostic_nms=self.agnostic_nms  # Allow person+weapon overlap
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
            
            # If available, run the dedicated weapon model and merge
            if self.weapon_model is not None:
                w_results = self.weapon_model(
                    frame,
                    conf=max(0.25, self.confidence_threshold - 0.1),
                    device=self.device,
                    verbose=False,
                    agnostic_nms=True
                )
                for w in w_results:
                    boxes = w.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].tolist()
                        class_name = w.names.get(class_id, f"class_{class_id}")
                        if self._is_weapon_label(class_name):
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
                verbose=False,
                agnostic_nms=self.agnostic_nms  # Allow person+weapon overlap
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
            
            # Merge optional weapon model detections
            if self.weapon_model is not None:
                w_results = self.weapon_model(
                    frame,
                    conf=max(0.25, self.confidence_threshold - 0.1),
                    device=self.device,
                    verbose=False,
                    agnostic_nms=True
                )
                for w in w_results:
                    boxes = w.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].tolist()
                        class_name = w.names.get(class_id, f"class_{class_id}")
                        if self._is_weapon_label(class_name):
                            detections.append(Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                bbox=bbox
                            ))
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return detections, annotated_frame

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        persist: bool = True,
    ) -> tuple:
        """
        Run detection WITH multi-object tracking (ByteTrack).
        
        Returns tracked objects with persistent IDs for motion analysis.
        Existing Detection objects are also returned for backward compat.
        
        Args:
            frame: BGR image as numpy array
            frame_id: Frame identifier
            persist: Whether to persist tracks across calls
            
        Returns:
            Tuple of (detections: List[Detection], 
                      tracked_objects: List[dict],
                      annotated_frame: np.ndarray)
            
            Each tracked_object dict:
                track_id: int (persistent ID from ByteTrack)
                bbox: [x1, y1, x2, y2]
                class_id: int
                class_name: str
                confidence: float
                centroid: (cx, cy)
        """
        if not self._loaded:
            self.load()
        
        if self.model is None:
            return [], [], frame
        
        detections = []
        tracked_objects = []
        annotated_frame = frame.copy()
        
        try:
            results = self.model.track(
                frame,
                conf=0.35,  # Increased vehicle/target confidence to prevent phantom objects
                device=self.device,
                verbose=False,
                persist=persist,
                tracker="bytetrack.yaml",
                agnostic_nms=self.agnostic_nms,  # Prevent same vehicle being detected as car AND truck
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
                    
                    # Standard detection
                    det = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox
                    )
                    detections.append(det)
                    
                    # Extract track ID if available
                    track_id = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i])
                    
                    if track_id is not None:
                        cx = (bbox[0] + bbox[2]) / 2.0
                        cy = (bbox[1] + bbox[3]) / 2.0
                        tracked_objects.append({
                            "track_id": track_id,
                            "bbox": bbox,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "centroid": (cx, cy),
                        })
                
                # Get annotated frame
                annotated_frame = result.plot()
            
            # Optionally merge weapon model detections (tracking IDs not available)
            if self.weapon_model is not None:
                w_results = self.weapon_model(
                    frame,
                    conf=max(0.25, self.confidence_threshold - 0.1),
                    device=self.device,
                    verbose=False,
                    agnostic_nms=True
                )
                for w in w_results:
                    boxes = w.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        confidence = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].tolist()
                        class_name = w.names.get(class_id, f"class_{class_id}")
                        if self._is_weapon_label(class_name):
                            detections.append(Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                bbox=bbox
                            ))
        
        except Exception as e:
            print(f"⚠️ Track detection error: {e}. Falling back to standard detection without tracking.")
            fallback_detections, fallback_annotated = self.detect_with_annotations(frame, frame_id)
            return fallback_detections, [], fallback_annotated
        
        return detections, tracked_objects, annotated_frame

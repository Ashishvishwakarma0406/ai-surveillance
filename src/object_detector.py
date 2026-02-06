"""
Object Detection Module

Provides YOLOv8-based object detection for the surveillance system.
Handles model loading, inference, and result parsing.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from src.utils.logger import get_logger
from src.utils.helpers import format_detection, clip_box_to_frame


@dataclass
class Detection:
    """
    Data class representing a single detection.
    
    Attributes:
        class_id: COCO class ID
        class_name: Human-readable class name
        confidence: Detection confidence (0-1)
        bbox: Bounding box as (x1, y1, x2, y2)
        center: Box center as (cx, cy)
        area: Box area in pixels
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int] = field(init=False)
    area: int = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties."""
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area = (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area
        }
    
    def __str__(self) -> str:
        """String representation."""
        return format_detection(self.class_name, self.confidence, self.bbox)


@dataclass
class DetectionResult:
    """
    Container for all detections in a frame.
    
    Attributes:
        frame_id: Associated frame ID
        detections: List of Detection objects
        inference_time: Time taken for inference (ms)
    """
    frame_id: int
    detections: List[Detection]
    inference_time: float = 0.0
    
    def filter_by_class(self, class_ids: List[int]) -> List[Detection]:
        """Get detections matching specified class IDs."""
        return [d for d in self.detections if d.class_id in class_ids]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def get_by_category(self, categories: Dict[str, List[int]]) -> Dict[str, List[Detection]]:
        """
        Group detections by category.
        
        Args:
            categories: Dict mapping category names to class IDs
            
        Returns:
            Dict mapping category names to detections
        """
        result = {}
        for category, class_ids in categories.items():
            result[category] = self.filter_by_class(class_ids)
        return result
    
    @property
    def count(self) -> int:
        """Total number of detections."""
        return len(self.detections)
    
    def has_class(self, class_id: int) -> bool:
        """Check if any detection has the specified class."""
        return any(d.class_id == class_id for d in self.detections)


class ObjectDetector:
    """
    YOLOv8-based object detector.
    
    Provides object detection using pretrained YOLO models
    with configurable confidence and class filtering.
    """
    
    # COCO class names (80 classes)
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }
    
    # Surveillance-relevant class mappings
    SURVEILLANCE_CLASSES = {
        'person': [0],
        'weapon': [43, 76],  # knife, scissors
        'trash': [39, 41],   # bottle, cup
        'garbage_bin': [75], # vase (proxy)
        'vehicle': [1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
    }
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize object detector.
        
        Args:
            model_name: YOLO model name/path
            confidence_threshold: Minimum detection confidence
            iou_threshold: NMS IoU threshold
            device: Device to run on ("auto", "cpu", "cuda:0")
            target_classes: List of class IDs to detect (None = all)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package not installed. Run: pip install ultralytics")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.target_classes = target_classes
        
        self.model: Optional[YOLO] = None
        self.logger = get_logger()
        
        # Statistics
        self.total_inferences = 0
        self.total_detections = 0
    
    def load(self) -> bool:
        """
        Load the YOLO model.
        
        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info(f"Loading YOLOv8 model: {self.model_name}")
            
            self.model = YOLO(self.model_name)
            
            # Set device
            if self.device != "auto":
                self.model.to(self.device)
            
            self.logger.info(f"Model loaded successfully on device: {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> DetectionResult:
        """
        Run object detection on a frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            DetectionResult with all detections
        """
        if self.model is None:
            self.logger.warning("Model not loaded. Call load() first.")
            return DetectionResult(frame_id=frame_id, detections=[])
        
        import time
        start_time = time.time()
        
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            verbose=False
        )[0]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Parse results
        detections = self._parse_results(results, frame.shape[:2])
        
        # Update statistics
        self.total_inferences += 1
        self.total_detections += len(detections)
        
        return DetectionResult(
            frame_id=frame_id,
            detections=detections,
            inference_time=inference_time
        )
    
    def _parse_results(self, results: Any, frame_shape: Tuple[int, int]) -> List[Detection]:
        """
        Parse YOLO results into Detection objects.
        
        Args:
            results: YOLO results object
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if results.boxes is None:
            return detections
        
        boxes = results.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Clip to frame boundaries
            bbox = clip_box_to_frame(
                (x1, y1, x2, y2),
                frame_shape[1],  # width
                frame_shape[0]   # height
            )
            
            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            )
            
            detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_id: int = 0
    ) -> List[DetectionResult]:
        """
        Run detection on multiple frames.
        
        Args:
            frames: List of frames
            start_frame_id: Starting frame ID
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for i, frame in enumerate(frames):
            result = self.detect(frame, frame_id=start_frame_id + i)
            results.append(result)
        return results
    
    def get_stats(self) -> dict:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with inference stats
        """
        return {
            'model_name': self.model_name,
            'total_inferences': self.total_inferences,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': (
                self.total_detections / self.total_inferences
                if self.total_inferences > 0 else 0
            ),
            'confidence_threshold': self.confidence_threshold,
            'target_classes': self.target_classes
        }
    
    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """Get class name from class ID."""
        return cls.COCO_CLASSES.get(class_id, f"unknown_{class_id}")
    
    @classmethod
    def get_surveillance_classes(cls) -> Dict[str, List[int]]:
        """Get surveillance-relevant class mappings."""
        return cls.SURVEILLANCE_CLASSES.copy()


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True
) -> np.ndarray:
    """
    Draw detection bounding boxes on a frame.
    
    Args:
        frame: Input frame
        detections: List of Detection objects
        color_map: Dict mapping class names to BGR colors
        thickness: Box line thickness
        font_scale: Text font scale
        show_confidence: Whether to show confidence percentage
        
    Returns:
        Frame with drawn detections
    """
    # Default colors
    default_colors = {
        'person': (0, 255, 0),      # Green
        'knife': (0, 0, 255),       # Red
        'scissors': (0, 0, 255),    # Red
        'bottle': (255, 165, 0),    # Orange
        'cup': (255, 165, 0),       # Orange
        'default': (255, 255, 255)  # White
    }
    
    colors = {**default_colors, **(color_map or {})}
    
    result = frame.copy()
    
    for det in detections:
        # Get color for this class
        color = colors.get(det.class_name, colors['default'])
        
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if show_confidence:
            label = f"{det.class_name}: {det.confidence:.0%}"
        else:
            label = det.class_name
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            result,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            1,
            cv2.LINE_AA
        )
    
    return result

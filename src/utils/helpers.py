"""
Helper Utilities Module

Common utility functions for the AI Surveillance System.
"""

import os
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import numpy as np


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)


def get_timestamp_ms() -> str:
    """
    Get current timestamp with milliseconds.
    
    Returns:
        Timestamp string with milliseconds
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_distance(point1: Tuple[int, int], 
                       point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Get center point of a bounding box.
    
    Args:
        box: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Center point (cx, cy)
    """
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    return (cx, cy)


def format_detection(
    class_name: str,
    confidence: float,
    bbox: Tuple[int, int, int, int]
) -> str:
    """
    Format a detection result as a string.
    
    Args:
        class_name: Detected class name
        confidence: Detection confidence (0-1)
        bbox: Bounding box coordinates
        
    Returns:
        Formatted detection string
    """
    return f"{class_name} ({confidence:.2%}) at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"


def format_alert(
    alert_type: str,
    severity: str,
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format an alert message.
    
    Args:
        alert_type: Type of alert (WEAPON, VIOLENCE, etc.)
        severity: Alert severity level
        details: Optional additional details
        
    Returns:
        Formatted alert string
    """
    timestamp = get_timestamp("%Y-%m-%d %H:%M:%S")
    message = f"[{severity}] {alert_type} detected at {timestamp}"
    
    if details:
        detail_str = ", ".join(f"{k}: {v}" for k, v in details.items())
        message += f" | {detail_str}"
    
    return message


def resize_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize a frame to target dimensions.
    
    Args:
        frame: Input frame as numpy array
        target_width: Target width
        target_height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    import cv2
    
    if maintain_aspect:
        h, w = frame.shape[:2]
        aspect = w / h
        target_aspect = target_width / target_height
        
        if aspect > target_aspect:
            # Width is the limiting factor
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height is the limiting factor
            new_height = target_height
            new_width = int(target_height * aspect)
        
        return cv2.resize(frame, (new_width, new_height))
    else:
        return cv2.resize(frame, (target_width, target_height))


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize frame pixel values to 0-1 range.
    
    Args:
        frame: Input frame (uint8, 0-255)
        
    Returns:
        Normalized frame (float32, 0-1)
    """
    return frame.astype(np.float32) / 255.0


def denormalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert normalized frame back to uint8.
    
    Args:
        frame: Normalized frame (float32, 0-1)
        
    Returns:
        Denormalized frame (uint8, 0-255)
    """
    return (frame * 255).astype(np.uint8)


def clip_box_to_frame(
    box: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int, int, int]:
    """
    Clip bounding box coordinates to frame boundaries.
    
    Args:
        box: Bounding box (x1, y1, x2, y2)
        frame_width: Frame width
        frame_height: Frame height
        
    Returns:
        Clipped bounding box
    """
    x1 = max(0, min(box[0], frame_width - 1))
    y1 = max(0, min(box[1], frame_height - 1))
    x2 = max(0, min(box[2], frame_width))
    y2 = max(0, min(box[3], frame_height))
    return (x1, y1, x2, y2)


def is_box_valid(box: Tuple[int, int, int, int], min_size: int = 10) -> bool:
    """
    Check if a bounding box is valid (has minimum size).
    
    Args:
        box: Bounding box (x1, y1, x2, y2)
        min_size: Minimum width/height in pixels
        
    Returns:
        True if box is valid
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width >= min_size and height >= min_size


def boxes_are_close(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    threshold: float = 100.0
) -> bool:
    """
    Check if two bounding boxes are close to each other.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        threshold: Maximum distance between centers
        
    Returns:
        True if boxes are close
    """
    center1 = get_box_center(box1)
    center2 = get_box_center(box2)
    distance = calculate_distance(center1, center2)
    return distance <= threshold

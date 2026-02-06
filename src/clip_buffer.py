"""
Clip Buffer Module

Provides a rolling frame buffer for temporal analysis.
Used to collect frames for violence detection with X3D model.
"""

import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger


@dataclass
class ClipData:
    """
    Container for a video clip.
    
    Attributes:
        frames: Numpy array of frames (T, H, W, C)
        start_frame_id: First frame ID in clip
        end_frame_id: Last frame ID in clip
        timestamps: List of frame timestamps
    """
    frames: np.ndarray
    start_frame_id: int
    end_frame_id: int
    timestamps: List[str]
    
    @property
    def length(self) -> int:
        """Number of frames in clip."""
        return len(self.frames)
    
    def to_tensor(self, normalize: bool = True) -> np.ndarray:
        """
        Convert clip to tensor format for model input.
        
        Args:
            normalize: Whether to normalize pixel values
            
        Returns:
            Tensor of shape (C, T, H, W)
        """
        # Convert from (T, H, W, C) to (C, T, H, W)
        tensor = np.transpose(self.frames, (3, 0, 1, 2))
        
        if normalize:
            tensor = tensor.astype(np.float32) / 255.0
        
        return tensor


class ClipBuffer:
    """
    Rolling frame buffer for temporal video analysis.
    
    Maintains a fixed-size buffer of recent frames
    for use with video classification models like X3D.
    """
    
    def __init__(
        self,
        buffer_size: int = 16,
        stride: int = 1
    ):
        """
        Initialize clip buffer.
        
        Args:
            buffer_size: Maximum number of frames to store
            stride: Frame stride for clip extraction
        """
        self.buffer_size = buffer_size
        self.stride = stride
        
        self._frames: deque = deque(maxlen=buffer_size)
        self._frame_ids: deque = deque(maxlen=buffer_size)
        self._timestamps: deque = deque(maxlen=buffer_size)
        
        self.logger = get_logger()
    
    def add_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: str
    ) -> None:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame as numpy array (H, W, C)
            frame_id: Frame identifier
            timestamp: Frame timestamp
        """
        self._frames.append(frame.copy())
        self._frame_ids.append(frame_id)
        self._timestamps.append(timestamp)
    
    def is_full(self) -> bool:
        """Check if buffer has required number of frames."""
        return len(self._frames) >= self.buffer_size
    
    def get_clip(self) -> Optional[ClipData]:
        """
        Get the current clip from buffer.
        
        Returns:
            ClipData if buffer is full, None otherwise
        """
        if not self.is_full():
            return None
        
        # Get frames with stride
        indices = list(range(0, self.buffer_size, self.stride))
        frames = [self._frames[i] for i in indices]
        
        return ClipData(
            frames=np.array(frames),
            start_frame_id=self._frame_ids[0],
            end_frame_id=self._frame_ids[-1],
            timestamps=list(self._timestamps)
        )
    
    def get_latest_frames(self, n: int) -> List[np.ndarray]:
        """
        Get the n most recent frames.
        
        Args:
            n: Number of frames to get
            
        Returns:
            List of frames
        """
        n = min(n, len(self._frames))
        return [self._frames[-i-1] for i in range(n)][::-1]
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._frames.clear()
        self._frame_ids.clear()
        self._timestamps.clear()
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._frames)
    
    @property
    def current_size(self) -> int:
        """Get current number of frames in buffer."""
        return len(self._frames)
    
    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self.buffer_size


class SlidingWindowBuffer:
    """
    Sliding window buffer for continuous clip extraction.
    
    Provides overlapping clips for continuous analysis
    with configurable window size and step.
    """
    
    def __init__(
        self,
        window_size: int = 16,
        step_size: int = 8
    ):
        """
        Initialize sliding window buffer.
        
        Args:
            window_size: Number of frames per window
            step_size: Frames to slide between windows
        """
        self.window_size = window_size
        self.step_size = step_size
        
        self._frames: List[np.ndarray] = []
        self._frame_ids: List[int] = []
        self._timestamps: List[str] = []
        
        self._last_clip_end = 0
        self.logger = get_logger()
    
    def add_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: str
    ) -> Optional[ClipData]:
        """
        Add a frame and potentially return a new clip.
        
        Args:
            frame: Frame as numpy array
            frame_id: Frame identifier
            timestamp: Frame timestamp
            
        Returns:
            ClipData if a new clip is ready, None otherwise
        """
        self._frames.append(frame.copy())
        self._frame_ids.append(frame_id)
        self._timestamps.append(timestamp)
        
        # Check if we have enough frames for a new clip
        frames_since_last = len(self._frames) - self._last_clip_end
        
        if len(self._frames) >= self.window_size and frames_since_last >= self.step_size:
            # Extract clip
            start_idx = len(self._frames) - self.window_size
            clip = ClipData(
                frames=np.array(self._frames[start_idx:start_idx + self.window_size]),
                start_frame_id=self._frame_ids[start_idx],
                end_frame_id=self._frame_ids[start_idx + self.window_size - 1],
                timestamps=self._timestamps[start_idx:start_idx + self.window_size]
            )
            
            self._last_clip_end = len(self._frames)
            
            # Trim old frames to prevent memory growth
            self._trim_old_frames()
            
            return clip
        
        return None
    
    def _trim_old_frames(self) -> None:
        """Remove frames no longer needed."""
        # Keep only frames needed for next window
        keep_count = self.window_size + self.step_size
        
        if len(self._frames) > keep_count:
            trim_count = len(self._frames) - keep_count
            self._frames = self._frames[trim_count:]
            self._frame_ids = self._frame_ids[trim_count:]
            self._timestamps = self._timestamps[trim_count:]
            self._last_clip_end -= trim_count
    
    def clear(self) -> None:
        """Clear all frames."""
        self._frames.clear()
        self._frame_ids.clear()
        self._timestamps.clear()
        self._last_clip_end = 0


class DetectionTracker:
    """
    Tracks detections across frames for temporal consistency.
    
    Used by the rule engine to enforce temporal thresholds
    (e.g., weapon in 3+ consecutive frames).
    """
    
    def __init__(self, max_history: int = 30):
        """
        Initialize detection tracker.
        
        Args:
            max_history: Maximum frames to track
        """
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history)
    
    def add_detections(self, frame_id: int, class_ids: List[int]) -> None:
        """
        Add detections for a frame.
        
        Args:
            frame_id: Frame identifier
            class_ids: List of detected class IDs
        """
        self._history.append({
            'frame_id': frame_id,
            'class_ids': set(class_ids)
        })
    
    def get_consecutive_count(self, class_id: int) -> int:
        """
        Get number of consecutive frames with a class detected.
        
        Args:
            class_id: Class ID to check
            
        Returns:
            Number of consecutive frames (from most recent)
        """
        count = 0
        for entry in reversed(self._history):
            if class_id in entry['class_ids']:
                count += 1
            else:
                break
        return count
    
    def get_occurrence_count(self, class_id: int, window: int = 10) -> int:
        """
        Get total occurrences of a class in recent frames.
        
        Args:
            class_id: Class ID to check
            window: Number of recent frames to check
            
        Returns:
            Total occurrence count
        """
        recent = list(self._history)[-window:]
        return sum(1 for entry in recent if class_id in entry['class_ids'])
    
    def has_recent_detection(self, class_id: int, within_frames: int = 5) -> bool:
        """
        Check if a class was detected recently.
        
        Args:
            class_id: Class ID to check
            within_frames: Number of frames to look back
            
        Returns:
            True if class was detected within window
        """
        return self.get_occurrence_count(class_id, within_frames) > 0
    
    def clear(self) -> None:
        """Clear detection history."""
        self._history.clear()

"""
Frame Extractor Module

Handles frame extraction, preprocessing, and sampling from video sources.
"""

import cv2
import time
import numpy as np
from typing import Optional, Generator, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.helpers import get_timestamp_ms, resize_frame


@dataclass
class FrameData:
    """
    Data class to hold frame information.
    
    Attributes:
        frame: The actual frame as numpy array
        frame_id: Sequential frame identifier
        timestamp: Capture timestamp
        original_size: Original frame dimensions (width, height)
        processed_size: Processed frame dimensions
    """
    frame: np.ndarray
    frame_id: int
    timestamp: str
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'original_size': self.original_size,
            'processed_size': self.processed_size
        }


class FrameExtractor:
    """
    Frame extraction and preprocessing pipeline.
    
    Extracts frames from video source at configurable rate
    and applies preprocessing operations.
    """
    
    def __init__(
        self,
        target_fps: float = 10.0,
        target_width: int = 1280,
        target_height: int = 720,
        skip_frames: int = 0,
        maintain_aspect: bool = True
    ):
        """
        Initialize frame extractor.
        
        Args:
            target_fps: Target processing FPS
            target_width: Target frame width
            target_height: Target frame height
            skip_frames: Number of frames to skip between processing
            maintain_aspect: Whether to maintain aspect ratio when resizing
        """
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.skip_frames = skip_frames
        self.maintain_aspect = maintain_aspect
        
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time = 0.0
        self.frame_count = 0
        self.skip_counter = 0
        
        self.logger = get_logger()
    
    def extract(
        self,
        video_source,
        preprocess: bool = True
    ) -> Generator[FrameData, None, None]:
        """
        Generate frames from video source.
        
        Args:
            video_source: Video source object with read() method
            preprocess: Whether to apply preprocessing
            
        Yields:
            FrameData objects with extracted frames
        """
        self.logger.info(
            f"Starting frame extraction at {self.target_fps} FPS, "
            f"target size: {self.target_width}x{self.target_height}"
        )
        
        while True:
            # Check FPS timing
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                continue
            
            # Read frame from source
            ret, frame = video_source.read()
            
            if not ret or frame is None:
                break
            
            # Handle frame skipping
            if self.skip_frames > 0:
                self.skip_counter += 1
                if self.skip_counter <= self.skip_frames:
                    continue
                self.skip_counter = 0
            
            # Store original size
            original_size = (frame.shape[1], frame.shape[0])
            
            # Preprocess frame
            if preprocess:
                frame = self._preprocess(frame)
            
            processed_size = (frame.shape[1], frame.shape[0])
            
            # Create frame data
            self.frame_count += 1
            frame_data = FrameData(
                frame=frame,
                frame_id=self.frame_count,
                timestamp=get_timestamp_ms(),
                original_size=original_size,
                processed_size=processed_size
            )
            
            self.last_frame_time = current_time
            
            yield frame_data
        
        self.logger.info(f"Frame extraction complete. Total frames: {self.frame_count}")
    
    def extract_single(
        self,
        frame: np.ndarray,
        preprocess: bool = True
    ) -> FrameData:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            preprocess: Whether to apply preprocessing
            
        Returns:
            Processed FrameData
        """
        original_size = (frame.shape[1], frame.shape[0])
        
        if preprocess:
            frame = self._preprocess(frame)
        
        processed_size = (frame.shape[1], frame.shape[0])
        
        self.frame_count += 1
        return FrameData(
            frame=frame,
            frame_id=self.frame_count,
            timestamp=get_timestamp_ms(),
            original_size=original_size,
            processed_size=processed_size
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame
        """
        # Resize if needed
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = resize_frame(
                frame,
                self.target_width,
                self.target_height,
                self.maintain_aspect
            )
        
        return frame
    
    def reset(self) -> None:
        """Reset extractor state."""
        self.frame_count = 0
        self.skip_counter = 0
        self.last_frame_time = 0.0
    
    def get_stats(self) -> dict:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary with extraction stats
        """
        return {
            'frames_extracted': self.frame_count,
            'target_fps': self.target_fps,
            'target_size': (self.target_width, self.target_height),
            'skip_frames': self.skip_frames
        }


class FrameBuffer:
    """
    Simple frame buffer for storing recent frames.
    
    Different from ClipBuffer - this stores individual FrameData
    objects for general access.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum frames to store
        """
        self.max_size = max_size
        self.buffer: List[FrameData] = []
    
    def add(self, frame_data: FrameData) -> None:
        """
        Add a frame to the buffer.
        
        Args:
            frame_data: Frame data to add
        """
        self.buffer.append(frame_data)
        
        # Remove oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_latest(self, n: int = 1) -> List[FrameData]:
        """
        Get the n most recent frames.
        
        Args:
            n: Number of frames to get
            
        Returns:
            List of most recent FrameData objects
        """
        return self.buffer[-n:] if n <= len(self.buffer) else self.buffer.copy()
    
    def get_by_id(self, frame_id: int) -> Optional[FrameData]:
        """
        Get a specific frame by ID.
        
        Args:
            frame_id: Frame ID to find
            
        Returns:
            FrameData if found, None otherwise
        """
        for frame_data in self.buffer:
            if frame_data.frame_id == frame_id:
                return frame_data
        return None
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class FrameRateLimiter:
    """
    Utility class for limiting frame rate.
    
    Ensures consistent frame timing regardless of
    processing speed variations.
    """
    
    def __init__(self, target_fps: float = 30.0):
        """
        Initialize rate limiter.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_time = 0.0
        self.actual_fps = 0.0
        self._fps_samples: List[float] = []
    
    def wait(self) -> float:
        """
        Wait to maintain target FPS.
        
        Returns:
            Actual time elapsed since last frame
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        # Wait if processing was faster than target
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
            elapsed = self.frame_time
        
        self.last_time = time.time()
        
        # Update FPS measurement
        self._update_fps(elapsed)
        
        return elapsed
    
    def _update_fps(self, elapsed: float) -> None:
        """Update rolling FPS average."""
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._fps_samples.append(instant_fps)
        
        # Keep last 30 samples
        if len(self._fps_samples) > 30:
            self._fps_samples.pop(0)
        
        self.actual_fps = sum(self._fps_samples) / len(self._fps_samples)
    
    def get_actual_fps(self) -> float:
        """
        Get actual measured FPS.
        
        Returns:
            Rolling average FPS
        """
        return self.actual_fps
    
    def should_process(self) -> bool:
        """
        Check if enough time has passed to process next frame.
        
        Returns:
            True if frame should be processed
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        return elapsed >= self.frame_time

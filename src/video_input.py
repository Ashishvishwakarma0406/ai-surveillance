"""
Video Input Module

Handles video capture from multiple sources: webcam, RTSP streams, and video files.
Provides a unified interface for frame acquisition.
"""

import cv2
import time
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Generator
from queue import Queue, Empty
import numpy as np

from src.utils.logger import get_logger


class VideoSourceError(Exception):
    """Exception raised for video source errors."""
    pass


class VideoSource(ABC):
    """
    Abstract base class for video sources.
    
    Provides a unified interface for capturing frames from
    different video sources (webcam, RTSP, file).
    """
    
    def __init__(self):
        """Initialize video source."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.logger = get_logger()
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if successfully opened
        """
        pass
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame
    
    def release(self) -> None:
        """Release the video source."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.info(f"Video source released. Total frames: {self.frame_count}")
    
    def is_opened(self) -> bool:
        """Check if video source is open."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_properties(self) -> dict:
        """
        Get video source properties.
        
        Returns:
            Dictionary with fps, width, height, frame_count
        """
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'is_running': self.is_running
        }
    
    def _update_properties(self) -> None:
        """Update video properties from capture object."""
        if self.cap is not None:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


class WebcamSource(VideoSource):
    """
    Video source for webcam capture.
    
    Captures frames from a local webcam device.
    """
    
    def __init__(self, device_index: int = 0):
        """
        Initialize webcam source.
        
        Args:
            device_index: Camera device index (default 0)
        """
        super().__init__()
        self.device_index = device_index
    
    def open(self) -> bool:
        """
        Open webcam for capture.
        
        Returns:
            True if successfully opened
            
        Raises:
            VideoSourceError: If webcam cannot be opened
        """
        self.logger.info(f"Opening webcam at index {self.device_index}...")
        
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            raise VideoSourceError(f"Failed to open webcam at index {self.device_index}")
        
        self._update_properties()
        self.is_running = True
        
        self.logger.info(f"Webcam opened: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True


class RTSPSource(VideoSource):
    """
    Video source for RTSP stream capture.
    
    Captures frames from an RTSP network stream.
    Includes reconnection logic for dropped connections.
    """
    
    def __init__(self, url: str, timeout: int = 10, max_retries: int = 3):
        """
        Initialize RTSP source.
        
        Args:
            url: RTSP stream URL
            timeout: Connection timeout in seconds
            max_retries: Maximum reconnection attempts
        """
        super().__init__()
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_count = 0
    
    def open(self) -> bool:
        """
        Open RTSP stream for capture.
        
        Returns:
            True if successfully opened
            
        Raises:
            VideoSourceError: If stream cannot be opened
        """
        self.logger.info(f"Connecting to RTSP stream: {self.url[:50]}...")
        
        # Set timeout for connection
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
        
        if not self.cap.isOpened():
            raise VideoSourceError(f"Failed to connect to RTSP stream: {self.url}")
        
        self._update_properties()
        self.is_running = True
        self.retry_count = 0
        
        self.logger.info(f"RTSP stream connected: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with reconnection logic.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = super().read()
        
        if not ret:
            # Try to reconnect
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self.logger.warning(f"RTSP stream dropped. Reconnecting ({self.retry_count}/{self.max_retries})...")
                time.sleep(1)
                try:
                    self.release()
                    self.open()
                    return self.read()
                except VideoSourceError:
                    pass
        else:
            self.retry_count = 0
        
        return ret, frame


class FileSource(VideoSource):
    """
    Video source for video file playback.
    
    Reads frames from a video file (MP4, AVI, etc.)
    """
    
    def __init__(self, file_path: str, loop: bool = False):
        """
        Initialize file source.
        
        Args:
            file_path: Path to video file
            loop: Whether to loop the video
        """
        super().__init__()
        self.file_path = file_path
        self.loop = loop
        self.total_frames = 0
    
    def open(self) -> bool:
        """
        Open video file for playback.
        
        Returns:
            True if successfully opened
            
        Raises:
            VideoSourceError: If file cannot be opened
        """
        self.logger.info(f"Opening video file: {self.file_path}")
        
        self.cap = cv2.VideoCapture(self.file_path)
        
        if not self.cap.isOpened():
            raise VideoSourceError(f"Failed to open video file: {self.file_path}")
        
        self._update_properties()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_running = True
        
        self.logger.info(
            f"Video file opened: {self.width}x{self.height} @ {self.fps:.1f} FPS, "
            f"{self.total_frames} frames"
        )
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with optional looping.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = super().read()
        
        if not ret and self.loop:
            # Reset to beginning of video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.logger.info("Video looped to beginning")
            ret, frame = super().read()
        
        return ret, frame
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_number: Target frame number
            
        Returns:
            True if seek successful
        """
        if self.cap is None:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return True
    
    def get_progress(self) -> float:
        """
        Get playback progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total_frames == 0:
            return 0.0
        return (self.frame_count / self.total_frames) * 100


class ThreadedVideoSource:
    """
    Threaded wrapper for video sources.
    
    Captures frames in a background thread for improved
    performance and reduced latency.
    """
    
    def __init__(self, source: VideoSource, queue_size: int = 30):
        """
        Initialize threaded video source.
        
        Args:
            source: Underlying video source
            queue_size: Maximum frames to buffer
        """
        self.source = source
        self.queue = Queue(maxsize=queue_size)
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.logger = get_logger()
    
    def start(self) -> 'ThreadedVideoSource':
        """
        Start the capture thread.
        
        Returns:
            Self for method chaining
        """
        if not self.source.is_opened():
            self.source.open()
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        self.logger.info("Threaded video capture started")
        return self
    
    def _capture_loop(self) -> None:
        """Background thread capture loop."""
        while self.is_running:
            if not self.queue.full():
                ret, frame = self.source.read()
                if ret:
                    self.queue.put((ret, frame))
                else:
                    if not self.source.is_opened():
                        break
                    time.sleep(0.01)
            else:
                # Queue full, skip frame
                time.sleep(0.01)
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from queue.
        
        Args:
            timeout: Queue wait timeout
            
        Returns:
            Tuple of (success, frame)
        """
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return False, None
    
    def stop(self) -> None:
        """Stop the capture thread."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.source.release()
        self.logger.info("Threaded video capture stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def create_video_source(
    source_type: str,
    **kwargs
) -> VideoSource:
    """
    Factory function to create video sources.
    
    Args:
        source_type: Type of source ("webcam", "rtsp", "file")
        **kwargs: Source-specific arguments
        
    Returns:
        Configured VideoSource instance
        
    Raises:
        ValueError: If unknown source type
    """
    if source_type == "webcam":
        return WebcamSource(
            device_index=kwargs.get('device_index', 0)
        )
    elif source_type == "rtsp":
        return RTSPSource(
            url=kwargs.get('url', ''),
            timeout=kwargs.get('timeout', 10),
            max_retries=kwargs.get('max_retries', 3)
        )
    elif source_type == "file":
        return FileSource(
            file_path=kwargs.get('file_path', ''),
            loop=kwargs.get('loop', False)
        )
    else:
        raise ValueError(f"Unknown video source type: {source_type}")

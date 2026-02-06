"""
OpenCV UI Module

Provides real-time video display with detection overlays and alert visualization.
"""

import cv2
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.object_detector import Detection, DetectionResult
from src.alert_manager import Alert


@dataclass
class UIConfig:
    """
    Configuration for the UI display.
    
    Attributes:
        window_name: Name of the display window
        fullscreen: Whether to run in fullscreen
        show_fps: Whether to show FPS counter
        show_detections: Whether to show detection boxes
        show_alerts: Whether to show alert notifications
        bbox_thickness: Bounding box line thickness
        font_scale: Text font scale
    """
    window_name: str = "AI Surveillance System"
    fullscreen: bool = False
    show_fps: bool = True
    show_detections: bool = True
    show_alerts: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.6


class ColorPalette:
    """Predefined color schemes for the UI."""
    
    # BGR colors
    DETECTION_COLORS = {
        'person': (0, 255, 0),        # Green
        'weapon': (0, 0, 255),        # Red
        'knife': (0, 0, 255),         # Red
        'scissors': (0, 0, 255),      # Red
        'trash': (0, 165, 255),       # Orange
        'bottle': (0, 165, 255),      # Orange
        'cup': (0, 165, 255),         # Orange
        'violence': (255, 0, 255),    # Magenta
        'default': (255, 255, 255)    # White
    }
    
    SEVERITY_COLORS = {
        'CRITICAL': (0, 0, 255),      # Red
        'WARNING': (0, 165, 255),     # Orange
        'INFORMATIONAL': (255, 200, 0) # Cyan
    }
    
    STATUS_COLORS = {
        'normal': (0, 255, 0),        # Green
        'warning': (0, 165, 255),     # Orange
        'critical': (0, 0, 255)       # Red
    }
    
    @classmethod
    def get_detection_color(cls, class_name: str) -> Tuple[int, int, int]:
        """Get color for a detection class."""
        return cls.DETECTION_COLORS.get(class_name, cls.DETECTION_COLORS['default'])
    
    @classmethod
    def get_severity_color(cls, severity: str) -> Tuple[int, int, int]:
        """Get color for a severity level."""
        return cls.SEVERITY_COLORS.get(severity, (255, 255, 255))


class DisplayOverlay:
    """
    Handles drawing overlays on video frames.
    
    Provides methods for drawing bounding boxes, labels,
    status panels, and alert notifications.
    """
    
    def __init__(self, config: UIConfig):
        """
        Initialize display overlay.
        
        Args:
            config: UI configuration
        """
        self.config = config
        self.colors = ColorPalette()
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence values
            
        Returns:
            Frame with detections drawn
        """
        result = frame.copy()
        
        for det in detections:
            color = self.colors.get_detection_color(det.class_name)
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                color,
                self.config.bbox_thickness
            )
            
            # Prepare label
            if show_confidence:
                label = f"{det.class_name}: {det.confidence:.0%}"
            else:
                label = det.class_name
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                1
            )
            
            cv2.rectangle(
                result,
                (x1, y1 - th - 8),
                (x1 + tw + 4, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        return result
    
    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input frame
            fps: Current FPS value
            position: Text position
            
        Returns:
            Frame with FPS drawn
        """
        result = frame.copy()
        
        text = f"FPS: {fps:.1f}"
        
        # Draw background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            result,
            (position[0] - 5, position[1] - th - 5),
            (position[0] + tw + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return result
    
    def draw_status_panel(
        self,
        frame: np.ndarray,
        detection_count: int,
        person_count: int,
        alert_count: int
    ) -> np.ndarray:
        """
        Draw status information panel.
        
        Args:
            frame: Input frame
            detection_count: Total detections
            person_count: Person count
            alert_count: Active alerts
            
        Returns:
            Frame with status panel
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Panel background
        panel_h = 80
        cv2.rectangle(
            result,
            (0, h - panel_h),
            (w, h),
            (50, 50, 50),
            -1
        )
        
        # Draw stats
        y_base = h - 55
        
        stats = [
            (f"Detections: {detection_count}", (0, 200, 255)),
            (f"Persons: {person_count}", (0, 255, 0)),
            (f"Alerts: {alert_count}", (0, 0, 255) if alert_count > 0 else (0, 255, 0))
        ]
        
        x = 20
        for text, color in stats:
            cv2.putText(
                result,
                text,
                (x, y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            x += 200
        
        # Timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            result,
            timestamp,
            (w - 220, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        return result
    
    def draw_alert_banner(
        self,
        frame: np.ndarray,
        alert: Alert
    ) -> np.ndarray:
        """
        Draw alert banner at top of frame.
        
        Args:
            frame: Input frame
            alert: Alert to display
            
        Returns:
            Frame with alert banner
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Banner height
        banner_h = 50
        
        # Get severity color
        color = self.colors.get_severity_color(alert.severity)
        
        # Draw banner background
        cv2.rectangle(result, (0, 0), (w, banner_h), color, -1)
        
        # Draw alert icon and message
        icon = "⚠" if alert.severity != "INFORMATIONAL" else "ℹ"
        message = f"  {alert.message}"
        
        cv2.putText(
            result,
            message,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw confidence
        conf_text = f"Conf: {alert.confidence:.0%}"
        cv2.putText(
            result,
            conf_text,
            (w - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return result
    
    def draw_alert_list(
        self,
        frame: np.ndarray,
        alerts: List[Alert],
        max_display: int = 4
    ) -> np.ndarray:
        """
        Draw list of recent alerts.
        
        Args:
            frame: Input frame
            alerts: List of alerts
            max_display: Maximum alerts to show
            
        Returns:
            Frame with alert list
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        if not alerts:
            return result
        
        # Display most recent
        display_alerts = alerts[-max_display:]
        
        y_offset = 70
        for alert in display_alerts:
            color = self.colors.get_severity_color(alert.severity)
            
            # Alert background
            cv2.rectangle(
                result,
                (w - 340, y_offset),
                (w - 10, y_offset + 28),
                color,
                -1
            )
            
            # Alert message (truncated)
            msg = alert.message[:38] + "..." if len(alert.message) > 38 else alert.message
            cv2.putText(
                result,
                msg,
                (w - 335, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            y_offset += 32
        
        return result
    
    def draw_status_indicator(
        self,
        frame: np.ndarray,
        status: str = "normal"
    ) -> np.ndarray:
        """
        Draw system status indicator.
        
        Args:
            frame: Input frame
            status: Status level (normal, warning, critical)
            
        Returns:
            Frame with status indicator
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        color = self.colors.STATUS_COLORS.get(status, (0, 255, 0))
        
        # Draw circle indicator
        cv2.circle(result, (w - 30, 30), 12, color, -1)
        cv2.circle(result, (w - 30, 30), 12, (255, 255, 255), 2)
        
        return result


class OpenCVUI:
    """
    Main OpenCV-based user interface.
    
    Provides real-time video display with detection and
    alert overlays, keyboard controls, and status panels.
    """
    
    def __init__(self, config: Optional[UIConfig] = None):
        """
        Initialize OpenCV UI.
        
        Args:
            config: UI configuration
        """
        self.config = config or UIConfig()
        self.overlay = DisplayOverlay(self.config)
        self.logger = get_logger()
        
        # State
        self.is_running = False
        self.is_paused = False
        self.window_created = False
        
        # FPS calculation
        self.fps = 0.0
        self._frame_times: List[float] = []
        self._last_frame_time = time.time()
        
        # Keyboard callbacks
        self._key_callbacks: Dict[int, Callable] = {}
    
    def create_window(self) -> None:
        """Create the display window."""
        if self.window_created:
            return
        
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        
        if self.config.fullscreen:
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        
        self.window_created = True
        self.logger.info(f"Created window: {self.config.window_name}")
    
    def destroy_window(self) -> None:
        """Destroy the display window."""
        if self.window_created:
            cv2.destroyWindow(self.config.window_name)
            self.window_created = False
    
    def register_key_callback(self, key: int, callback: Callable) -> None:
        """
        Register a callback for a specific key.
        
        Args:
            key: Key code (e.g., ord('q'))
            callback: Function to call when key pressed
        """
        self._key_callbacks[key] = callback
    
    def update(
        self,
        frame: np.ndarray,
        detections: Optional[DetectionResult] = None,
        alerts: Optional[List[Alert]] = None,
        current_alert: Optional[Alert] = None
    ) -> bool:
        """
        Update the display with a new frame.
        
        Args:
            frame: Frame to display
            detections: Optional detection results
            alerts: Optional list of recent alerts
            current_alert: Optional active critical alert
            
        Returns:
            False if window was closed, True otherwise
        """
        if not self.window_created:
            self.create_window()
        
        # Update FPS
        self._update_fps()
        
        # Build display frame
        display_frame = frame.copy()
        
        # Draw detections
        if self.config.show_detections and detections:
            display_frame = self.overlay.draw_detections(
                display_frame,
                detections.detections
            )
        
        # Draw current alert banner
        if current_alert:
            display_frame = self.overlay.draw_alert_banner(
                display_frame,
                current_alert
            )
        
        # Draw alert list
        if self.config.show_alerts and alerts:
            display_frame = self.overlay.draw_alert_list(display_frame, alerts)
        
        # Draw FPS
        if self.config.show_fps:
            display_frame = self.overlay.draw_fps(display_frame, self.fps)
        
        # Draw status panel
        det_count = len(detections.detections) if detections else 0
        person_count = len([d for d in detections.detections if d.class_id == 0]) if detections else 0
        alert_count = len(alerts) if alerts else 0
        
        display_frame = self.overlay.draw_status_panel(
            display_frame,
            det_count,
            person_count,
            alert_count
        )
        
        # Draw status indicator
        if current_alert and current_alert.severity == "CRITICAL":
            status = "critical"
        elif alerts:
            status = "warning"
        else:
            status = "normal"
        display_frame = self.overlay.draw_status_indicator(display_frame, status)
        
        # Display frame
        cv2.imshow(self.config.window_name, display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            return False
        elif key == ord(' '):  # Space - pause
            self.is_paused = not self.is_paused
        elif key == ord('f'):  # f - fullscreen toggle
            self._toggle_fullscreen()
        elif key in self._key_callbacks:
            self._key_callbacks[key]()
        
        return True
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        current_time = time.time()
        elapsed = current_time - self._last_frame_time
        self._last_frame_time = current_time
        
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        
        avg_time = sum(self._frame_times) / len(self._frame_times)
        self.fps = 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        self.config.fullscreen = not self.config.fullscreen
        
        if self.config.fullscreen:
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.setWindowProperty(
                self.config.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )
    
    def show_message(
        self,
        message: str,
        duration_ms: int = 2000,
        frame: Optional[np.ndarray] = None
    ) -> None:
        """
        Show a temporary message overlay.
        
        Args:
            message: Message to display
            duration_ms: Display duration in milliseconds
            frame: Optional base frame (black if None)
        """
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Message background
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cx, cy = w // 2, h // 2
        
        cv2.rectangle(
            display,
            (cx - tw//2 - 20, cy - th//2 - 20),
            (cx + tw//2 + 20, cy + th//2 + 20),
            (50, 50, 50),
            -1
        )
        
        cv2.putText(
            display,
            message,
            (cx - tw//2, cy + th//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        cv2.imshow(self.config.window_name, display)
        cv2.waitKey(duration_ms)
    
    def __enter__(self):
        """Context manager entry."""
        self.create_window()
        self.is_running = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.is_running = False
        self.destroy_window()
        return False


def create_placeholder_frame(
    width: int = 1280,
    height: int = 720,
    message: str = "No Video Feed"
) -> np.ndarray:
    """
    Create a placeholder frame when no video is available.
    
    Args:
        width: Frame width
        height: Frame height
        message: Message to display
        
    Returns:
        Placeholder frame
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        frame[i, :] = (30 + i // 10, 30 + i // 10, 30 + i // 10)
    
    # Draw message
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
    cx, cy = width // 2, height // 2
    
    cv2.putText(
        frame,
        message,
        (cx - tw//2, cy + th//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (100, 100, 100),
        2,
        cv2.LINE_AA
    )
    
    return frame

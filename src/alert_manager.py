"""
Alert Manager Module

Handles alert generation, logging, and output management.
Provides console logging, file logging, and frame/clip saving.
"""

import os
import cv2
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import deque

from src.utils.logger import get_logger
from src.utils.helpers import get_timestamp, get_timestamp_ms, ensure_directory
from src.rule_engine import RuleOutput, AlertSeverity


@dataclass
class Alert:
    """
    Represents a generated alert.
    
    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of alert (weapon, violence, etc.)
        severity: Alert severity level
        message: Alert message
        timestamp: Alert generation timestamp
        frame_id: Associated frame ID
        confidence: Detection confidence
        details: Additional details
        frame_path: Path to saved frame (if any)
        clip_path: Path to saved clip (if any)
    """
    alert_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: str
    frame_id: int
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    frame_path: Optional[str] = None
    clip_path: Optional[str] = None
    acknowledged: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_rule_output(
        cls,
        output: RuleOutput,
        frame_id: int
    ) -> 'Alert':
        """
        Create alert from rule output.
        
        Args:
            output: Rule output
            frame_id: Current frame ID
            
        Returns:
            New Alert instance
        """
        return cls(
            alert_id=f"ALERT-{get_timestamp_ms()}",
            alert_type=output.rule_name,
            severity=output.severity.name,
            message=output.message,
            timestamp=datetime.now().isoformat(),
            frame_id=frame_id,
            confidence=output.confidence,
            details=output.details
        )


class AlertManager:
    """
    Central alert management system.
    
    Handles alert creation, storage, logging, and output.
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        log_dir: str = "logs",
        save_frames: bool = True,
        save_clips: bool = True,
        max_alerts: int = 100
    ):
        """
        Initialize alert manager.
        
        Args:
            output_dir: Directory for saved frames/clips
            log_dir: Directory for alert logs
            save_frames: Whether to save flagged frames
            save_clips: Whether to save video clips
            max_alerts: Maximum alerts to keep in memory
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.save_frames = save_frames
        self.save_clips = save_clips
        
        # Alert storage
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_count = 0
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'by_type': {},
            'by_severity': {}
        }
        
        self.logger = get_logger()
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_directories(self) -> None:
        """Create required directories."""
        ensure_directory(self.output_dir / "frames")
        ensure_directory(self.output_dir / "clips")
        ensure_directory(self.log_dir)
    
    def create_alert(
        self,
        rule_output: RuleOutput,
        frame_id: int,
        frame: Optional[Any] = None,
        clip_frames: Optional[List[Any]] = None
    ) -> Alert:
        """
        Create and register a new alert.
        
        Args:
            rule_output: Rule output that triggered alert
            frame_id: Current frame ID
            frame: Optional frame to save
            clip_frames: Optional video clip frames
            
        Returns:
            Created Alert
        """
        # Create alert
        alert = Alert.from_rule_output(rule_output, frame_id)
        
        # Save frame if provided
        if self.save_frames and frame is not None:
            alert.frame_path = self._save_frame(alert, frame)
        
        # Save clip if provided
        if self.save_clips and clip_frames is not None:
            alert.clip_path = self._save_clip(alert, clip_frames)
        
        # Store alert
        self.alerts.append(alert)
        self.alert_count += 1
        
        # Update statistics
        self._update_stats(alert)
        
        # Log alert
        self._log_alert(alert)
        
        return alert
    
    def _save_frame(self, alert: Alert, frame: Any) -> str:
        """
        Save alert frame to disk.
        
        Args:
            alert: Alert object
            frame: Frame to save
            
        Returns:
            Path to saved frame
        """
        try:
            filename = f"{alert.alert_type}_{alert.alert_id}.jpg"
            filepath = self.output_dir / "frames" / filename
            
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.logger.debug(f"Saved alert frame: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
            return ""
    
    def _save_clip(self, alert: Alert, frames: List[Any]) -> str:
        """
        Save alert video clip to disk.
        
        Args:
            alert: Alert object
            frames: List of frames
            
        Returns:
            Path to saved clip
        """
        try:
            if not frames:
                return ""
            
            filename = f"{alert.alert_type}_{alert.alert_id}.mp4"
            filepath = self.output_dir / "clips" / filename
            
            # Get frame properties
            height, width = frames[0].shape[:2]
            fps = 10.0
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            
            self.logger.debug(f"Saved alert clip: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save clip: {e}")
            return ""
    
    def _update_stats(self, alert: Alert) -> None:
        """Update alert statistics."""
        self.stats['total_alerts'] += 1
        
        # By type
        if alert.alert_type not in self.stats['by_type']:
            self.stats['by_type'][alert.alert_type] = 0
        self.stats['by_type'][alert.alert_type] += 1
        
        # By severity
        if alert.severity not in self.stats['by_severity']:
            self.stats['by_severity'][alert.severity] = 0
        self.stats['by_severity'][alert.severity] += 1
    
    def _log_alert(self, alert: Alert) -> None:
        """Log alert to file and console."""
        # Console logging with severity-based formatting
        severity = alert.severity
        self.logger.alert(alert.message, severity=severity)
        
        # File logging
        log_file = self.log_dir / "alerts.log"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    'timestamp': alert.timestamp,
                    'alert_id': alert.alert_id,
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'message': alert.message,
                    'confidence': alert.confidence,
                    'frame_id': alert.frame_id
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write alert log: {e}")
    
    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        """
        Get the n most recent alerts.
        
        Args:
            n: Number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return list(self.alerts)[-n:]
    
    def get_alerts_by_type(self, alert_type: str) -> List[Alert]:
        """
        Get alerts of a specific type.
        
        Args:
            alert_type: Alert type to filter
            
        Returns:
            List of matching alerts
        """
        return [a for a in self.alerts if a.alert_type == alert_type]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """
        Get alerts of a specific severity.
        
        Args:
            severity: Severity level to filter
            
        Returns:
            List of matching alerts
        """
        return [a for a in self.alerts if a.severity == severity]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as acknowledged.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert found and acknowledged
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_unacknowledged_alerts(self) -> List[Alert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]
    
    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self.alerts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert stats
        """
        return {
            **self.stats,
            'alerts_in_memory': len(self.alerts),
            'unacknowledged': len(self.get_unacknowledged_alerts())
        }
    
    def export_alerts(self, filepath: str) -> bool:
        """
        Export all alerts to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        try:
            alerts_data = [a.to_dict() for a in self.alerts]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'total_alerts': len(alerts_data),
                    'alerts': alerts_data
                }, f, indent=2)
            
            self.logger.info(f"Exported {len(alerts_data)} alerts to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
            return False


class AlertRenderer:
    """
    Renders alert overlays on video frames.
    
    Provides visual feedback for active alerts.
    """
    
    # Colors for severity levels (BGR)
    SEVERITY_COLORS = {
        'CRITICAL': (0, 0, 255),     # Red
        'WARNING': (0, 165, 255),    # Orange
        'INFORMATIONAL': (255, 200, 0)  # Cyan
    }
    
    def __init__(
        self,
        banner_height: int = 60,
        font_scale: float = 0.8,
        thickness: int = 2
    ):
        """
        Initialize alert renderer.
        
        Args:
            banner_height: Height of alert banner
            font_scale: Text font scale
            thickness: Text thickness
        """
        self.banner_height = banner_height
        self.font_scale = font_scale
        self.thickness = thickness
    
    def render_alert(
        self,
        frame: Any,
        alert: Alert
    ) -> Any:
        """
        Render alert overlay on frame.
        
        Args:
            frame: Input frame
            alert: Alert to render
            
        Returns:
            Frame with alert overlay
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Get severity color
        color = self.SEVERITY_COLORS.get(alert.severity, (255, 255, 255))
        
        # Draw banner background
        cv2.rectangle(
            result,
            (0, 0),
            (w, self.banner_height),
            color,
            -1
        )
        
        # Draw alert message
        cv2.putText(
            result,
            alert.message,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            self.thickness,
            cv2.LINE_AA
        )
        
        # Draw timestamp
        cv2.putText(
            result,
            alert.timestamp[:19],
            (w - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return result
    
    def render_multiple_alerts(
        self,
        frame: Any,
        alerts: List[Alert],
        max_display: int = 3
    ) -> Any:
        """
        Render multiple alert indicators.
        
        Args:
            frame: Input frame
            alerts: List of alerts
            max_display: Maximum alerts to display
            
        Returns:
            Frame with alert overlays
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        if not alerts:
            return result
        
        # Display most recent alerts
        display_alerts = alerts[-max_display:]
        
        y_offset = 10
        for alert in display_alerts:
            color = self.SEVERITY_COLORS.get(alert.severity, (255, 255, 255))
            
            # Draw alert box
            cv2.rectangle(
                result,
                (w - 320, y_offset),
                (w - 10, y_offset + 30),
                color,
                -1
            )
            
            # Draw message
            cv2.putText(
                result,
                alert.message[:35],
                (w - 315, y_offset + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            y_offset += 35
        
        return result
    
    def render_alert_indicator(
        self,
        frame: Any,
        has_critical: bool = False,
        has_warning: bool = False
    ) -> Any:
        """
        Render simple alert status indicator.
        
        Args:
            frame: Input frame
            has_critical: Whether critical alerts exist
            has_warning: Whether warning alerts exist
            
        Returns:
            Frame with status indicator
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw indicator circle
        if has_critical:
            color = (0, 0, 255)  # Red
        elif has_warning:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        cv2.circle(result, (w - 30, 30), 15, color, -1)
        cv2.circle(result, (w - 30, 30), 15, (255, 255, 255), 2)
        
        return result

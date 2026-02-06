"""
AI-Powered Video Surveillance System

Main entry point and pipeline orchestrator.
Integrates all modules for end-to-end video analysis.

Usage:
    python -m src.main                          # Run with webcam
    python -m src.main --source file --path video.mp4  # Run with video file
    python -m src.main --config configs/main_config.yaml  # Custom config
"""

import os
import sys
import time
import argparse
import signal
from pathlib import Path
from typing import Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.helpers import ensure_directory

from src.video_input import (
    VideoSource,
    WebcamSource,
    RTSPSource,
    FileSource,
    ThreadedVideoSource,
    create_video_source
)
from src.frame_extractor import FrameExtractor, FrameData
from src.object_detector import ObjectDetector, DetectionResult
from src.clip_buffer import ClipBuffer, SlidingWindowBuffer
from src.violence_detector import ViolenceDetector, ViolenceResult
from src.rule_engine import RuleEngine, RuleOutput
from src.alert_manager import AlertManager, Alert
from src.ui_opencv import OpenCVUI, UIConfig, create_placeholder_frame


class SurveillancePipeline:
    """
    Main surveillance pipeline orchestrator.
    
    Coordinates video capture, detection, rule evaluation,
    alert generation, and UI display.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        source_type: str = "webcam",
        source_path: Optional[str] = None,
        headless: bool = False
    ):
        """
        Initialize surveillance pipeline.
        
        Args:
            config_path: Path to configuration file
            source_type: Video source type (webcam, rtsp, file)
            source_path: Path/URL for file/rtsp sources
            headless: Run without UI display
        """
        self.source_type = source_type
        self.source_path = source_path
        self.headless = headless
        
        # Load configuration
        self._load_config(config_path)
        
        # Setup logging
        self.logger = setup_logger(
            log_dir=str(PROJECT_ROOT / "logs"),
            console_output=True,
            file_output=True,
            debug_mode=self.config.get('global.debug', False)
        )
        
        # Initialize components
        self.video_source: Optional[VideoSource] = None
        self.frame_extractor: Optional[FrameExtractor] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.clip_buffer: Optional[ClipBuffer] = None
        self.violence_detector: Optional[ViolenceDetector] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.alert_manager: Optional[AlertManager] = None
        self.ui: Optional[OpenCVUI] = None
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0.0
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration files."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = PROJECT_ROOT / "configs" / "main_config.yaml"
        
        rules_file = PROJECT_ROOT / "configs" / "rules_config.yaml"
        
        if config_file.exists():
            self.config = load_config(str(config_file), str(rules_file))
        else:
            # Use defaults
            self.config = type('Config', (), {
                'get': lambda self, k, d=None: d,
                'video': {},
                'models': {},
                'rules': {}
            })()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received...")
        self.stop()
    
    def initialize(self) -> bool:
        """
        Initialize all pipeline components.
        
        Returns:
            True if all components initialized successfully
        """
        self.logger.info("=" * 60)
        self.logger.info("AI-Powered Video Surveillance System")
        self.logger.info("=" * 60)
        
        try:
            # Create directories
            ensure_directory(PROJECT_ROOT / "output" / "frames")
            ensure_directory(PROJECT_ROOT / "output" / "clips")
            ensure_directory(PROJECT_ROOT / "logs")
            
            # Initialize video source
            self._init_video_source()
            
            # Initialize frame extractor
            self._init_frame_extractor()
            
            # Initialize object detector
            self._init_object_detector()
            
            # Initialize clip buffer
            self._init_clip_buffer()
            
            # Initialize violence detector
            self._init_violence_detector()
            
            # Initialize rule engine
            self._init_rule_engine()
            
            # Initialize alert manager
            self._init_alert_manager()
            
            # Initialize UI
            if not self.headless:
                self._init_ui()
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _init_video_source(self) -> None:
        """Initialize video source."""
        self.logger.info(f"Initializing video source: {self.source_type}")
        
        if self.source_type == "webcam":
            device_idx = self.config.get('video.webcam.device_index', 0)
            self.video_source = WebcamSource(device_index=device_idx)
        elif self.source_type == "rtsp":
            url = self.source_path or self.config.get('video.rtsp.url', '')
            timeout = self.config.get('video.rtsp.timeout', 10)
            self.video_source = RTSPSource(url=url, timeout=timeout)
        elif self.source_type == "file":
            path = self.source_path or self.config.get('video.file.path', '')
            self.video_source = FileSource(file_path=path, loop=True)
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
        
        self.video_source.open()
    
    def _init_frame_extractor(self) -> None:
        """Initialize frame extractor."""
        self.frame_extractor = FrameExtractor(
            target_fps=self.config.get('video.frame.target_fps', 10),
            target_width=self.config.get('video.frame.width', 1280),
            target_height=self.config.get('video.frame.height', 720),
            skip_frames=self.config.get('video.frame.skip_frames', 0)
        )
    
    def _init_object_detector(self) -> None:
        """Initialize object detector."""
        self.object_detector = ObjectDetector(
            model_name=self.config.get('models.object_detection.model_name', 'yolov8n.pt'),
            confidence_threshold=self.config.get('models.object_detection.confidence_threshold', 0.5),
            device=self.config.get('models.object_detection.device', 'auto')
        )
        self.object_detector.load()
    
    def _init_clip_buffer(self) -> None:
        """Initialize clip buffer."""
        clip_length = self.config.get('models.violence_detection.clip_length', 16)
        self.clip_buffer = ClipBuffer(buffer_size=clip_length)
    
    def _init_violence_detector(self) -> None:
        """Initialize violence detector."""
        self.violence_detector = ViolenceDetector(
            model_name=self.config.get('models.violence_detection.model_name', 'x3d_m'),
            violence_threshold=self.config.get('models.violence_detection.violence_threshold', 0.6),
            device=self.config.get('models.violence_detection.device', 'auto')
        )
        # Try to load - may fail if pytorchvideo not available
        try:
            self.violence_detector.load()
        except Exception as e:
            self.logger.warning(f"Violence detector not loaded (will use fallback): {e}")
    
    def _init_rule_engine(self) -> None:
        """Initialize rule engine."""
        self.rule_engine = RuleEngine()
        
        # Load rules from config or use defaults
        if hasattr(self.config, 'rules') and self.config.rules:
            self.rule_engine.load_from_config(self.config.rules)
        else:
            self.rule_engine.create_default_rules()
    
    def _init_alert_manager(self) -> None:
        """Initialize alert manager."""
        self.alert_manager = AlertManager(
            output_dir=str(PROJECT_ROOT / "output"),
            log_dir=str(PROJECT_ROOT / "logs"),
            save_frames=self.config.get('alerts.save_frames.enabled', True),
            save_clips=True
        )
    
    def _init_ui(self) -> None:
        """Initialize UI."""
        ui_config = UIConfig(
            window_name=self.config.get('ui.opencv.window_name', 'AI Surveillance System'),
            fullscreen=self.config.get('ui.opencv.fullscreen', False),
            show_fps=self.config.get('ui.opencv.show_fps', True),
            show_detections=self.config.get('ui.opencv.show_detections', True),
            show_alerts=self.config.get('ui.opencv.show_alerts', True)
        )
        self.ui = OpenCVUI(config=ui_config)
    
    def run(self) -> None:
        """
        Run the main processing loop.
        
        Processes video frames through the detection pipeline
        until stopped or video ends.
        """
        if not self.is_running:
            if not self.initialize():
                return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info("Starting surveillance pipeline...")
        self.logger.info("Press 'q' or ESC to quit, SPACE to pause")
        
        # Active alerts for display
        recent_alerts: List[Alert] = []
        current_alert: Optional[Alert] = None
        alert_display_time = 0.0
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = self.video_source.read()
                
                if not ret or frame is None:
                    if isinstance(self.video_source, FileSource):
                        self.logger.info("Video file ended")
                        break
                    continue
                
                # Process frame
                frame_data = self.frame_extractor.extract_single(frame)
                self.frame_count += 1
                
                # Object detection
                detections = self.object_detector.detect(
                    frame_data.frame,
                    frame_id=frame_data.frame_id
                )
                
                # Add to clip buffer
                self.clip_buffer.add_frame(
                    frame_data.frame,
                    frame_data.frame_id,
                    frame_data.timestamp
                )
                
                # Violence detection (when buffer is full)
                violence_result = None
                if self.clip_buffer.is_full():
                    clip = self.clip_buffer.get_clip()
                    if clip:
                        person_count = len([
                            d for d in detections.detections if d.class_id == 0
                        ])
                        violence_result = self.violence_detector.detect_with_persons(
                            clip,
                            [person_count] * clip.length
                        )
                
                # Rule evaluation
                triggered_rules = self.rule_engine.evaluate_all(
                    detections=detections,
                    violence_result=violence_result,
                    frame_id=frame_data.frame_id
                )
                
                # Process triggered rules
                for rule_output in triggered_rules:
                    alert = self.alert_manager.create_alert(
                        rule_output=rule_output,
                        frame_id=frame_data.frame_id,
                        frame=frame_data.frame
                    )
                    recent_alerts.append(alert)
                    
                    # Show critical alerts prominently
                    if alert.severity == "CRITICAL":
                        current_alert = alert
                        alert_display_time = time.time()
                
                # Clear alert banner after 5 seconds
                if current_alert and time.time() - alert_display_time > 5.0:
                    current_alert = None
                
                # Keep only recent alerts for display
                recent_alerts = self.alert_manager.get_recent_alerts(5)
                
                # Update UI
                if self.ui:
                    if not self.ui.update(
                        frame=frame_data.frame,
                        detections=detections,
                        alerts=recent_alerts,
                        current_alert=current_alert
                    ):
                        break
                
                # Check pause
                if self.ui and self.ui.is_paused:
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the pipeline and cleanup resources."""
        self.is_running = False
        
        # Calculate runtime stats
        runtime = time.time() - self.start_time if self.start_time > 0 else 0
        avg_fps = self.frame_count / runtime if runtime > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("Shutting down surveillance pipeline...")
        
        # Cleanup components
        if self.video_source:
            self.video_source.release()
        
        if self.ui:
            self.ui.destroy_window()
        
        # Print summary
        self.logger.info(f"Runtime: {runtime:.1f} seconds")
        self.logger.info(f"Frames processed: {self.frame_count}")
        self.logger.info(f"Average FPS: {avg_fps:.1f}")
        
        if self.alert_manager:
            stats = self.alert_manager.get_stats()
            self.logger.info(f"Total alerts: {stats['total_alerts']}")
            if stats['by_type']:
                self.logger.info(f"Alerts by type: {stats['by_type']}")
        
        self.logger.info("Pipeline shutdown complete")
        self.logger.info("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Video Surveillance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main                           # Run with default webcam
    python -m src.main --source webcam --device 0   # Specific camera
    python -m src.main --source file --path input/video.mp4  # Video file
    python -m src.main --source rtsp --url rtsp://...  # RTSP stream
    python -m src.main --headless                # Run without UI
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        choices=['webcam', 'file', 'rtsp'],
        default='webcam',
        help='Video source type (default: webcam)'
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='Path to video file (for file source)'
    )
    
    parser.add_argument(
        '--url', '-u',
        type=str,
        default=None,
        help='RTSP stream URL (for rtsp source)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        help='Webcam device index (for webcam source)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without UI display'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine source path
    if args.source == 'file' and args.path:
        source_path = args.path
    elif args.source == 'rtsp' and args.url:
        source_path = args.url
    else:
        source_path = None
    
    # Create and run pipeline
    pipeline = SurveillancePipeline(
        config_path=args.config,
        source_type=args.source,
        source_path=source_path,
        headless=args.headless
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()

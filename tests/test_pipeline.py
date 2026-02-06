"""
Test Script for AI Surveillance System

Tests core functionality of all modules.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def test_config_loader():
    """Test configuration loading."""
    print("\n[TEST] Configuration Loader")
    print("-" * 40)
    
    from src.utils.config_loader import load_config, get_config
    
    config_path = PROJECT_ROOT / "configs" / "main_config.yaml"
    
    try:
        config = load_config(str(config_path))
        
        # Test get method
        source_type = config.get('video.source_type', 'webcam')
        print(f"✓ Config loaded successfully")
        print(f"  - Video source type: {source_type}")
        print(f"  - Model name: {config.get('models.object_detection.model_name', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_logger():
    """Test logging functionality."""
    print("\n[TEST] Logger")
    print("-" * 40)
    
    from src.utils.logger import setup_logger, get_logger
    
    try:
        logger = setup_logger(log_dir="logs", console_output=True, file_output=False)
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.alert("Test alert", severity="WARNING")
        
        print("✓ Logger working correctly")
        return True
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        return False


def test_helpers():
    """Test helper functions."""
    print("\n[TEST] Helper Functions")
    print("-" * 40)
    
    from src.utils.helpers import (
        get_timestamp,
        calculate_iou,
        get_box_center,
        boxes_are_close
    )
    
    try:
        # Test timestamp
        ts = get_timestamp()
        print(f"✓ Timestamp: {ts}")
        
        # Test IoU
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = calculate_iou(box1, box2)
        print(f"✓ IoU calculation: {iou:.3f}")
        
        # Test box center
        center = get_box_center(box1)
        print(f"✓ Box center: {center}")
        
        # Test box proximity
        close = boxes_are_close(box1, box2, threshold=100)
        print(f"✓ Boxes are close: {close}")
        
        return True
    except Exception as e:
        print(f"✗ Helper test failed: {e}")
        return False


def test_frame_extractor():
    """Test frame extractor."""
    print("\n[TEST] Frame Extractor")
    print("-" * 40)
    
    from src.frame_extractor import FrameExtractor, FrameData
    
    try:
        extractor = FrameExtractor(
            target_fps=10,
            target_width=640,
            target_height=480
        )
        
        # Create dummy frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_frame[:, :, 1] = 128  # Add some green
        
        # Process frame
        frame_data = extractor.extract_single(dummy_frame)
        
        print(f"✓ Frame extracted successfully")
        print(f"  - Frame ID: {frame_data.frame_id}")
        print(f"  - Original size: {frame_data.original_size}")
        print(f"  - Processed size: {frame_data.processed_size}")
        
        return True
    except Exception as e:
        print(f"✗ Frame extractor test failed: {e}")
        return False


def test_clip_buffer():
    """Test clip buffer."""
    print("\n[TEST] Clip Buffer")
    print("-" * 40)
    
    from src.clip_buffer import ClipBuffer
    
    try:
        buffer = ClipBuffer(buffer_size=16)
        
        # Add frames
        for i in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer.add_frame(frame, frame_id=i, timestamp=f"2024-01-01 00:00:{i:02d}")
        
        print(f"✓ Buffer size: {len(buffer)}/{buffer.capacity}")
        print(f"✓ Buffer is full: {buffer.is_full()}")
        
        # Get clip
        clip = buffer.get_clip()
        if clip:
            print(f"✓ Clip extracted: {clip.length} frames")
        
        return True
    except Exception as e:
        print(f"✗ Clip buffer test failed: {e}")
        return False


def test_object_detector():
    """Test object detector initialization."""
    print("\n[TEST] Object Detector")
    print("-" * 40)
    
    from src.object_detector import ObjectDetector, Detection
    
    try:
        detector = ObjectDetector(
            model_name="yolov8n.pt",
            confidence_threshold=0.5
        )
        
        # Test class mappings
        print(f"✓ Detector created")
        print(f"  - Model: {detector.model_name}")
        print(f"  - Confidence threshold: {detector.confidence_threshold}")
        
        # Test class name lookup
        person_class = ObjectDetector.get_class_name(0)
        print(f"  - Class 0 = {person_class}")
        
        # Note: Full model loading requires downloading weights
        print("  (Model loading skipped - requires ultralytics)")
        
        return True
    except Exception as e:
        print(f"✗ Object detector test failed: {e}")
        return False


def test_rule_engine():
    """Test rule engine."""
    print("\n[TEST] Rule Engine")
    print("-" * 40)
    
    from src.rule_engine import (
        RuleEngine,
        WeaponDetectionRule,
        TrashDetectionRule,
        AlertSeverity
    )
    from src.object_detector import Detection, DetectionResult
    
    try:
        engine = RuleEngine()
        engine.create_default_rules()
        
        print(f"✓ Rule engine created with {len(engine.rules)} rules")
        
        # Create mock detections
        detections = DetectionResult(
            frame_id=1,
            detections=[
                Detection(class_id=0, class_name="person", confidence=0.9, bbox=(100, 100, 200, 300)),
                Detection(class_id=39, class_name="bottle", confidence=0.8, bbox=(300, 200, 350, 280)),
                Detection(class_id=41, class_name="cup", confidence=0.7, bbox=(310, 210, 360, 290)),
            ]
        )
        
        # Evaluate rules
        triggered = engine.evaluate_all(detections=detections, frame_id=1)
        
        print(f"✓ Rules evaluated: {len(triggered)} triggered")
        for output in triggered:
            print(f"  - {output.rule_name}: {output.message}")
        
        return True
    except Exception as e:
        print(f"✗ Rule engine test failed: {e}")
        return False


def test_alert_manager():
    """Test alert manager."""
    print("\n[TEST] Alert Manager")
    print("-" * 40)
    
    from src.alert_manager import AlertManager, Alert
    from src.rule_engine import RuleOutput, AlertSeverity
    
    try:
        manager = AlertManager(
            output_dir="output",
            log_dir="logs",
            save_frames=False,
            save_clips=False
        )
        
        # Create test alert
        rule_output = RuleOutput(
            triggered=True,
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            confidence=0.85
        )
        
        alert = manager.create_alert(rule_output, frame_id=1)
        
        print(f"✓ Alert created: {alert.alert_id}")
        print(f"  - Type: {alert.alert_type}")
        print(f"  - Severity: {alert.severity}")
        print(f"  - Message: {alert.message}")
        
        # Check stats
        stats = manager.get_stats()
        print(f"✓ Total alerts: {stats['total_alerts']}")
        
        return True
    except Exception as e:
        print(f"✗ Alert manager test failed: {e}")
        return False


def test_ui():
    """Test UI components (without display)."""
    print("\n[TEST] UI Components")
    print("-" * 40)
    
    from src.ui_opencv import UIConfig, DisplayOverlay, ColorPalette
    from src.object_detector import Detection
    
    try:
        config = UIConfig(
            window_name="Test Window",
            show_fps=True,
            show_detections=True
        )
        
        overlay = DisplayOverlay(config)
        
        print(f"✓ UI config created")
        print(f"  - Window: {config.window_name}")
        print(f"  - Show FPS: {config.show_fps}")
        
        # Test color palette
        color = ColorPalette.get_detection_color("person")
        print(f"✓ Person color: {color}")
        
        # Test drawing on dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=(100, 100, 200, 300))
        ]
        
        result = overlay.draw_detections(frame, detections)
        print(f"✓ Detection overlay drawn: {result.shape}")
        
        return True
    except Exception as e:
        print(f"✗ UI test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("AI Surveillance System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Config Loader", test_config_loader),
        ("Logger", test_logger),
        ("Helpers", test_helpers),
        ("Frame Extractor", test_frame_extractor),
        ("Clip Buffer", test_clip_buffer),
        ("Object Detector", test_object_detector),
        ("Rule Engine", test_rule_engine),
        ("Alert Manager", test_alert_manager),
        ("UI Components", test_ui),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

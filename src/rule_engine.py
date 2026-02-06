"""
Rule Engine Module

Implements detection rules that convert raw ML outputs into actionable alerts.
Provides temporal consistency checking and configurable thresholds.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from src.utils.logger import get_logger
from src.object_detector import Detection, DetectionResult
from src.violence_detector import ViolenceResult
from src.utils.helpers import boxes_are_close, get_box_center


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFORMATIONAL = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class RuleOutput:
    """
    Output from a rule evaluation.
    
    Attributes:
        triggered: Whether the rule was triggered
        rule_name: Name of the rule
        severity: Alert severity level
        message: Alert message
        confidence: Detection confidence
        details: Additional details
        evidence: Visual evidence (frame, detections)
    """
    triggered: bool
    rule_name: str
    severity: AlertSeverity
    message: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'triggered': self.triggered,
            'rule_name': self.rule_name,
            'severity': self.severity.name,
            'message': self.message,
            'confidence': self.confidence,
            'details': self.details
        }


class Rule(ABC):
    """
    Abstract base class for detection rules.
    
    Rules convert raw model outputs into alert decisions
    with temporal consistency and threshold checking.
    """
    
    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        cooldown_seconds: float = 5.0,
        enabled: bool = True
    ):
        """
        Initialize rule.
        
        Args:
            name: Rule name
            severity: Default alert severity
            cooldown_seconds: Minimum time between alerts
            enabled: Whether rule is active
        """
        self.name = name
        self.severity = severity
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        
        self.last_trigger_time = 0.0
        self.trigger_count = 0
        self.logger = get_logger()
    
    @abstractmethod
    def evaluate(self, **kwargs) -> RuleOutput:
        """
        Evaluate the rule against current data.
        
        Returns:
            RuleOutput with evaluation result
        """
        pass
    
    def can_trigger(self) -> bool:
        """Check if cooldown has elapsed."""
        return time.time() - self.last_trigger_time >= self.cooldown_seconds
    
    def record_trigger(self) -> None:
        """Record that rule was triggered."""
        self.last_trigger_time = time.time()
        self.trigger_count += 1
    
    def reset(self) -> None:
        """Reset rule state."""
        self.last_trigger_time = 0.0


class WeaponDetectionRule(Rule):
    """
    Rule for detecting weapons in video frames.
    
    Triggers alert when weapon detected in N consecutive frames.
    """
    
    # Class IDs for weapons
    WEAPON_CLASSES = [43, 76]  # knife, scissors
    
    def __init__(
        self,
        consecutive_threshold: int = 3,
        min_confidence: float = 0.6,
        cooldown_seconds: float = 5.0
    ):
        """
        Initialize weapon detection rule.
        
        Args:
            consecutive_threshold: Frames with weapon to trigger
            min_confidence: Minimum detection confidence
            cooldown_seconds: Alert cooldown
        """
        super().__init__(
            name="weapon_detection",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=cooldown_seconds
        )
        
        self.consecutive_threshold = consecutive_threshold
        self.min_confidence = min_confidence
        
        # Track consecutive detections
        self._consecutive_count = 0
        self._last_frame_id = -1
        self._detected_weapons: List[Detection] = []
    
    def evaluate(
        self,
        detections: DetectionResult,
        frame_id: int = 0,
        **kwargs
    ) -> RuleOutput:
        """
        Evaluate weapon detection rule.
        
        Args:
            detections: Detection results from object detector
            frame_id: Current frame ID
            
        Returns:
            RuleOutput with weapon detection result
        """
        if not self.enabled:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Find weapon detections
        weapons = [
            d for d in detections.detections
            if d.class_id in self.WEAPON_CLASSES and d.confidence >= self.min_confidence
        ]
        
        # Check frame continuity
        if frame_id == self._last_frame_id + 1:
            if weapons:
                self._consecutive_count += 1
                self._detected_weapons = weapons
            else:
                self._consecutive_count = 0
                self._detected_weapons = []
        else:
            # Non-consecutive frame, reset
            self._consecutive_count = 1 if weapons else 0
            self._detected_weapons = weapons if weapons else []
        
        self._last_frame_id = frame_id
        
        # Check if threshold met and cooldown elapsed
        triggered = (
            self._consecutive_count >= self.consecutive_threshold and
            self.can_trigger()
        )
        
        if triggered:
            self.record_trigger()
            weapon_names = [d.class_name for d in self._detected_weapons]
            max_conf = max(d.confidence for d in self._detected_weapons)
            
            return RuleOutput(
                triggered=True,
                rule_name=self.name,
                severity=self.severity,
                message=f"WEAPON DETECTED: {', '.join(set(weapon_names))}",
                confidence=max_conf,
                details={
                    'weapon_types': weapon_names,
                    'consecutive_frames': self._consecutive_count,
                    'detections': [d.to_dict() for d in self._detected_weapons]
                }
            )
        
        return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)


class ViolenceDetectionRule(Rule):
    """
    Rule for detecting violence in video clips.
    
    Uses violence probability from video classifier.
    """
    
    def __init__(
        self,
        probability_threshold: float = 0.65,
        require_multiple_persons: bool = True,
        min_persons: int = 2,
        cooldown_seconds: float = 10.0
    ):
        """
        Initialize violence detection rule.
        
        Args:
            probability_threshold: Violence probability threshold
            require_multiple_persons: Whether to require multiple people
            min_persons: Minimum persons for violence detection
            cooldown_seconds: Alert cooldown
        """
        super().__init__(
            name="violence_detection",
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=cooldown_seconds
        )
        
        self.probability_threshold = probability_threshold
        self.require_multiple_persons = require_multiple_persons
        self.min_persons = min_persons
    
    def evaluate(
        self,
        violence_result: Optional[ViolenceResult] = None,
        person_count: int = 0,
        **kwargs
    ) -> RuleOutput:
        """
        Evaluate violence detection rule.
        
        Args:
            violence_result: Result from violence detector
            person_count: Number of persons in scene
            
        Returns:
            RuleOutput with violence detection result
        """
        if not self.enabled or violence_result is None:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Check person requirement
        if self.require_multiple_persons and person_count < self.min_persons:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Check probability threshold
        if violence_result.violence_probability < self.probability_threshold:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Check cooldown
        if not self.can_trigger():
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        self.record_trigger()
        
        return RuleOutput(
            triggered=True,
            rule_name=self.name,
            severity=self.severity,
            message="VIOLENCE DETECTED",
            confidence=violence_result.confidence,
            details={
                'violence_probability': violence_result.violence_probability,
                'person_count': person_count,
                'clip_frames': f"{violence_result.clip_start_id}-{violence_result.clip_end_id}"
            }
        )


class TrashDetectionRule(Rule):
    """
    Rule for detecting litter/trash in frames.
    
    Detects accumulation of trash-like objects.
    """
    
    TRASH_CLASSES = [39, 41]  # bottle, cup
    
    def __init__(
        self,
        min_objects: int = 2,
        min_confidence: float = 0.5,
        cooldown_seconds: float = 30.0
    ):
        """
        Initialize trash detection rule.
        
        Args:
            min_objects: Minimum trash objects to trigger
            min_confidence: Minimum detection confidence
            cooldown_seconds: Alert cooldown
        """
        super().__init__(
            name="trash_detection",
            severity=AlertSeverity.WARNING,
            cooldown_seconds=cooldown_seconds
        )
        
        self.min_objects = min_objects
        self.min_confidence = min_confidence
    
    def evaluate(
        self,
        detections: DetectionResult,
        **kwargs
    ) -> RuleOutput:
        """
        Evaluate trash detection rule.
        
        Args:
            detections: Detection results
            
        Returns:
            RuleOutput with trash detection result
        """
        if not self.enabled:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Find trash detections
        trash = [
            d for d in detections.detections
            if d.class_id in self.TRASH_CLASSES and d.confidence >= self.min_confidence
        ]
        
        if len(trash) < self.min_objects:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        if not self.can_trigger():
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        self.record_trigger()
        
        return RuleOutput(
            triggered=True,
            rule_name=self.name,
            severity=self.severity,
            message=f"LITTER DETECTED: {len(trash)} objects",
            confidence=max(d.confidence for d in trash),
            details={
                'trash_count': len(trash),
                'object_types': [d.class_name for d in trash]
            }
        )


class GarbageOverflowRule(Rule):
    """
    Rule for detecting overflowing garbage bins.
    
    Detects trash objects near garbage bin locations.
    """
    
    TRASH_CLASSES = [39, 41]  # bottle, cup
    BIN_CLASSES = [75]  # vase (proxy)
    
    def __init__(
        self,
        proximity_threshold: float = 150.0,
        trash_near_bin_threshold: int = 3,
        cooldown_seconds: float = 60.0
    ):
        """
        Initialize garbage overflow rule.
        
        Args:
            proximity_threshold: Max distance from bin (pixels)
            trash_near_bin_threshold: Trash objects near bin to trigger
            cooldown_seconds: Alert cooldown
        """
        super().__init__(
            name="garbage_overflow",
            severity=AlertSeverity.WARNING,
            cooldown_seconds=cooldown_seconds
        )
        
        self.proximity_threshold = proximity_threshold
        self.trash_near_bin_threshold = trash_near_bin_threshold
    
    def evaluate(
        self,
        detections: DetectionResult,
        **kwargs
    ) -> RuleOutput:
        """
        Evaluate garbage overflow rule.
        
        Args:
            detections: Detection results
            
        Returns:
            RuleOutput with overflow detection result
        """
        if not self.enabled:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Find bins and trash
        bins = [d for d in detections.detections if d.class_id in self.BIN_CLASSES]
        trash = [d for d in detections.detections if d.class_id in self.TRASH_CLASSES]
        
        if not bins or not trash:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Count trash near each bin
        max_nearby = 0
        for bin_det in bins:
            nearby = sum(
                1 for t in trash
                if boxes_are_close(bin_det.bbox, t.bbox, self.proximity_threshold)
            )
            max_nearby = max(max_nearby, nearby)
        
        if max_nearby < self.trash_near_bin_threshold:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        if not self.can_trigger():
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        self.record_trigger()
        
        return RuleOutput(
            triggered=True,
            rule_name=self.name,
            severity=self.severity,
            message="GARBAGE OVERFLOW DETECTED",
            confidence=0.7,
            details={
                'trash_near_bin': max_nearby,
                'bin_count': len(bins)
            }
        )


class CrowdDensityRule(Rule):
    """
    Rule for monitoring crowd density.
    
    Tracks person count and alerts on high density.
    """
    
    def __init__(
        self,
        low_threshold: int = 5,
        medium_threshold: int = 10,
        high_threshold: int = 20,
        cooldown_seconds: float = 120.0
    ):
        """
        Initialize crowd density rule.
        
        Args:
            low_threshold: Low density person count
            medium_threshold: Medium density person count
            high_threshold: High density person count (triggers alert)
            cooldown_seconds: Alert cooldown
        """
        super().__init__(
            name="crowd_density",
            severity=AlertSeverity.INFORMATIONAL,
            cooldown_seconds=cooldown_seconds
        )
        
        self.thresholds = {
            'low': low_threshold,
            'medium': medium_threshold,
            'high': high_threshold
        }
    
    def evaluate(
        self,
        detections: DetectionResult,
        **kwargs
    ) -> RuleOutput:
        """
        Evaluate crowd density rule.
        
        Args:
            detections: Detection results
            
        Returns:
            RuleOutput with crowd density result
        """
        if not self.enabled:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        # Count persons
        persons = [d for d in detections.detections if d.class_id == 0]
        count = len(persons)
        
        if count < self.thresholds['high']:
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        if not self.can_trigger():
            return RuleOutput(triggered=False, rule_name=self.name, severity=self.severity)
        
        self.record_trigger()
        
        return RuleOutput(
            triggered=True,
            rule_name=self.name,
            severity=self.severity,
            message=f"HIGH CROWD DENSITY: {count} persons",
            confidence=0.9,
            details={
                'person_count': count,
                'density_level': 'high'
            }
        )


class RuleEngine:
    """
    Central rule engine that orchestrates all detection rules.
    
    Manages rule evaluation, result aggregation, and temporal smoothing.
    """
    
    def __init__(self):
        """Initialize rule engine."""
        self.rules: Dict[str, Rule] = {}
        self.logger = get_logger()
        self._smoothing_buffer: Dict[str, deque] = {}
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the engine.
        
        Args:
            rule: Rule instance to add
        """
        self.rules[rule.name] = rule
        self._smoothing_buffer[rule.name] = deque(maxlen=5)
        self.logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> None:
        """Remove a rule by name."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            del self._smoothing_buffer[rule_name]
    
    def evaluate_all(
        self,
        detections: Optional[DetectionResult] = None,
        violence_result: Optional[ViolenceResult] = None,
        frame_id: int = 0
    ) -> List[RuleOutput]:
        """
        Evaluate all rules and return triggered alerts.
        
        Args:
            detections: Object detection results
            violence_result: Violence detection result
            frame_id: Current frame ID
            
        Returns:
            List of triggered RuleOutput objects
        """
        triggered = []
        
        # Count persons for context
        person_count = 0
        if detections:
            person_count = len([d for d in detections.detections if d.class_id == 0])
        
        for name, rule in self.rules.items():
            try:
                output = rule.evaluate(
                    detections=detections,
                    violence_result=violence_result,
                    frame_id=frame_id,
                    person_count=person_count
                )
                
                if output.triggered:
                    triggered.append(output)
                    self.logger.info(f"Rule triggered: {name} - {output.message}")
                    
            except Exception as e:
                self.logger.error(f"Rule {name} evaluation failed: {e}")
        
        return triggered
    
    def create_default_rules(self) -> None:
        """Create and add default rule set."""
        self.add_rule(WeaponDetectionRule())
        self.add_rule(ViolenceDetectionRule())
        self.add_rule(TrashDetectionRule())
        self.add_rule(GarbageOverflowRule())
        self.add_rule(CrowdDensityRule())
        
        self.logger.info(f"Created {len(self.rules)} default rules")
    
    def load_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load rules from configuration.
        
        Args:
            config: Rules configuration dictionary
        """
        # Weapon detection
        if config.get('weapon_detection', {}).get('enabled', True):
            self.add_rule(WeaponDetectionRule(
                consecutive_threshold=config.get('weapon_detection', {}).get('consecutive_frames_threshold', 3),
                min_confidence=config.get('weapon_detection', {}).get('min_confidence', 0.6),
                cooldown_seconds=config.get('weapon_detection', {}).get('cooldown', 5)
            ))
        
        # Violence detection
        if config.get('violence_detection', {}).get('enabled', True):
            self.add_rule(ViolenceDetectionRule(
                probability_threshold=config.get('violence_detection', {}).get('probability_threshold', 0.65),
                require_multiple_persons=config.get('violence_detection', {}).get('require_multiple_persons', True),
                min_persons=config.get('violence_detection', {}).get('min_persons', 2),
                cooldown_seconds=config.get('violence_detection', {}).get('cooldown', 10)
            ))
        
        # Trash detection
        if config.get('trash_detection', {}).get('enabled', True):
            self.add_rule(TrashDetectionRule(
                min_objects=config.get('trash_detection', {}).get('min_objects', 2),
                min_confidence=config.get('trash_detection', {}).get('min_confidence', 0.5),
                cooldown_seconds=config.get('trash_detection', {}).get('cooldown', 30)
            ))
        
        # Garbage overflow
        if config.get('garbage_overflow', {}).get('enabled', True):
            self.add_rule(GarbageOverflowRule(
                trash_near_bin_threshold=config.get('garbage_overflow', {}).get('trash_near_bin_threshold', 3),
                cooldown_seconds=config.get('garbage_overflow', {}).get('cooldown', 60)
            ))
        
        # Crowd density
        if config.get('crowd_detection', {}).get('enabled', True):
            thresholds = config.get('crowd_detection', {}).get('thresholds', {})
            self.add_rule(CrowdDensityRule(
                low_threshold=thresholds.get('low', 5),
                medium_threshold=thresholds.get('medium', 10),
                high_threshold=thresholds.get('high', 20),
                cooldown_seconds=config.get('crowd_detection', {}).get('cooldown', 120)
            ))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        return {
            'total_rules': len(self.rules),
            'rule_stats': {
                name: {
                    'trigger_count': rule.trigger_count,
                    'enabled': rule.enabled,
                    'severity': rule.severity.name
                }
                for name, rule in self.rules.items()
            }
        }
    
    def reset_all(self) -> None:
        """Reset all rules."""
        for rule in self.rules.values():
            rule.reset()

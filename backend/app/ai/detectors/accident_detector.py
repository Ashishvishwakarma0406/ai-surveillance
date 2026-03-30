"""
Vehicle Accident Detector

Physics-based accident detection using multi-object tracking,
motion dynamics analysis, and rule-based temporal reasoning.

Detects collisions by observing violations of expected motion:
- Sudden velocity discontinuities
- Trajectory convergence between vehicles
- Bounding box overlap (physical contact)
- Post-impact stillness

NO model training required. Operates on top of YOLOv8 + ByteTrack.
"""

import time
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from itertools import combinations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# COCO vehicle class IDs
# ---------------------------------------------------------------------------
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
PERSON_CLASS_ID = 0

# Class ID -> human-readable vehicle type name
VEHICLE_CLASS_NAMES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrackSnapshot:
    """Single frame observation of a tracked object."""
    frame_id: int
    timestamp: float
    centroid: Tuple[float, float]
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    confidence: float


@dataclass
class TrackHistory:
    """Per-object motion history for a tracked vehicle."""
    track_id: int
    class_id: int = -1
    class_name: str = "Vehicle"  # e.g. "Car", "Motorcycle"
    snapshots: deque = field(default_factory=lambda: deque(maxlen=30))

    # Derived motion vectors (computed on update)
    velocities: deque = field(default_factory=lambda: deque(maxlen=30))
    accelerations: deque = field(default_factory=lambda: deque(maxlen=30))
    trajectory_angles: deque = field(default_factory=lambda: deque(maxlen=30))

    last_seen_frame: int = 0

    def add(self, snapshot: TrackSnapshot):
        """Add observation and recompute motion vectors."""
        self.snapshots.append(snapshot)
        self.class_id = snapshot.class_id
        self.class_name = VEHICLE_CLASS_NAMES.get(snapshot.class_id, "Vehicle")
        self.last_seen_frame = snapshot.frame_id
        self._recompute()

    def _recompute(self):
        """Recompute velocity, acceleration, and trajectory angle."""
        if len(self.snapshots) < 2:
            return

        prev = self.snapshots[-2]
        curr = self.snapshots[-1]

        dx = curr.centroid[0] - prev.centroid[0]
        dy = curr.centroid[1] - prev.centroid[1]

        speed = math.sqrt(dx * dx + dy * dy)
        self.velocities.append(speed)

        angle_deg = math.degrees(math.atan2(dy, dx))
        self.trajectory_angles.append(angle_deg)

        if len(self.velocities) >= 2:
            accel = self.velocities[-1] - self.velocities[-2]
            self.accelerations.append(accel)

    @property
    def current_velocity(self) -> float:
        return self.velocities[-1] if self.velocities else 0.0

    @property
    def avg_recent_velocity(self) -> float:
        """Average velocity over last 5 frames (excluding current)."""
        if len(self.velocities) < 3:
            return self.current_velocity
        window = list(self.velocities)[-5:-1]
        return sum(window) / len(window) if window else 0.0

    @property
    def current_centroid(self) -> Optional[Tuple[float, float]]:
        return self.snapshots[-1].centroid if self.snapshots else None

    @property
    def current_bbox(self) -> Optional[List[float]]:
        return self.snapshots[-1].bbox if self.snapshots else None

    @property
    def current_angle(self) -> float:
        return self.trajectory_angles[-1] if self.trajectory_angles else 0.0

    @property
    def bbox_area(self) -> float:
        if not self.snapshots:
            return 0.0
        b = self.snapshots[-1].bbox
        return (b[2] - b[0]) * (b[3] - b[1])


@dataclass
class CollisionSignals:
    """Result of evaluating collision signals for a vehicle pair."""
    pair: Tuple[int, int]                   # (track_id_a, track_id_b)
    sudden_velocity_drop: bool = False
    proximity_convergence: bool = False
    bbox_overlap: bool = False
    trajectory_disruption: bool = False
    post_impact_static: bool = False
    track_vanished: bool = False

    # Raw values for explainability
    velocity_ratio_a: float = 1.0
    velocity_ratio_b: float = 1.0
    min_distance: float = float("inf")
    closing_rate: float = 0.0
    iou: float = 0.0
    angle_change_a: float = 0.0
    angle_change_b: float = 0.0
    static_frames: int = 0

    @property
    def active_signal_count(self) -> int:
        return sum([
            self.sudden_velocity_drop,
            self.proximity_convergence,
            self.bbox_overlap,
            self.trajectory_disruption,
            self.post_impact_static,
            self.track_vanished,
        ])

    @property
    def confidence(self) -> float:
        weights = {
            "sudden_velocity_drop": 0.25,
            "proximity_convergence": 0.20,
            "bbox_overlap": 0.15,
            "trajectory_disruption": 0.10,
            "post_impact_static": 0.10,
            "track_vanished": 0.35,
        }
        score = 0.0
        for signal_name, weight in weights.items():
            if getattr(self, signal_name):
                score += weight
        return min(score, 1.0)


@dataclass
class AccidentEvent:
    """Confirmed accident detection event."""
    event_id: str
    frame_id: int
    timestamp: float
    track_ids: Tuple[int, int]
    vehicle_types: Tuple[str, str]  # e.g. ("Motorcycle", "Car")
    confidence: float
    severity: str                # "warning" or "critical"
    collision_zone: List[float]  # [x1, y1, x2, y2] merged bbox
    signals: Dict[str, bool]
    signal_details: Dict[str, float]
    trajectory_points_a: List[Tuple[float, float]]
    trajectory_points_b: List[Tuple[float, float]]

    @property
    def collision_description(self) -> str:
        """Human-readable collision description, e.g. 'Motorcycle and Car collision'."""
        a, b = self.vehicle_types
        if a == b:
            return f"2 {a}s collision detected"
        return f"{a} and {b} collision detected"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "track_ids": list(self.track_ids),
            "vehicle_types": list(self.vehicle_types),
            "collision_description": self.collision_description,
            "confidence": self.confidence,
            "severity": self.severity,
            "collision_zone": self.collision_zone,
            "signals": self.signals,
            "signal_details": self.signal_details,
        }


# ---------------------------------------------------------------------------
# Accident Detector
# ---------------------------------------------------------------------------
class AccidentDetector:
    """
    Physics-based accident detection engine.

    Operates on tracked vehicle detections from YOLOv8 + ByteTrack.
    Uses 5 weighted signals with multi-frame temporal validation
    and 6 false-positive suppression strategies.
    """

    def __init__(
        self,
        velocity_drop_threshold: float = 0.65,      # Back up from 0.75
        proximity_threshold_px: float = 85.0,       # Back down from 100.0
        iou_threshold: float = 0.05,                # Up from 0.01 to require actual bounding box intersection
        trajectory_angle_threshold: float = 20.0,   # Down from 25.0 to catch aggressive lane switches
        post_impact_static_frames: int = 3,         # Up from 2
        min_signals_required: int = 2,              # Kept at 2
        min_confidence: float = 0.45,               # Up from 0.40
        validation_window: int = 5,
        confirmation_frames: int = 2,               # Up to 2 to prevent 1-frame glitches
        cooldown_seconds: float = 10.0,             # Keep brief cooldown
        optical_flow_interval: int = 3,
        track_stale_threshold: int = 15,
        min_bbox_area: float = 300.0,               # Up from 200
        frame_margin_ratio: float = 0.05,
        max_simultaneous_events: int = 5,
    ):
        # --- Thresholds ---
        self.velocity_drop_threshold = velocity_drop_threshold
        self.proximity_threshold_px = proximity_threshold_px
        self.iou_threshold = iou_threshold
        self.trajectory_angle_threshold = trajectory_angle_threshold
        self.post_impact_static_frames = post_impact_static_frames
        self.min_signals_required = min_signals_required
        self.min_confidence = min_confidence
        self.validation_window = validation_window
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        self.optical_flow_interval = optical_flow_interval
        self.track_stale_threshold = track_stale_threshold
        self.min_bbox_area = min_bbox_area
        self.frame_margin_ratio = frame_margin_ratio
        self.max_simultaneous_events = max_simultaneous_events

        # --- State ---
        self.tracks: Dict[int, TrackHistory] = {}
        self.prev_gray: Optional[np.ndarray] = None
        self.flow_magnitude_avg: float = 0.0
        self.flow_angle_std: float = float("inf")
        self.frame_count: int = 0
        self.frame_shape: Tuple[int, int] = (720, 1280)

        # Sliding window: pair_key -> deque of CollisionSignals
        self._signal_history: Dict[str, deque] = {}

        # Cooldown tracking
        self._last_alert_time: Dict[str, float] = {}
        self._recent_alert_zones: List[Tuple[float, float, float]] = []  # (timestamp, cx, cy)

        # Active confirmed events
        self._active_events: List[AccidentEvent] = []
        self._event_counter: int = 0

        # Low visibility mode
        self._low_visibility: bool = False

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------
    def update(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
        frame_id: int = 0,
        fps: float = 10.0,
    ) -> List[AccidentEvent]:
        """
        Process one frame of tracked vehicle detections.

        Args:
            tracked_objects: list of dicts with keys:
                track_id, bbox, class_id, confidence, centroid
            frame: current BGR frame (np.ndarray)
            prev_frame: previous BGR frame (for optical flow)
            frame_id: monotonic frame counter
            fps: processing FPS for timestamp calculation

        Returns:
            List of newly confirmed AccidentEvent objects.
        """
        self.frame_count = frame_id
        self.frame_shape = frame.shape[:2]
        timestamp = frame_id / fps if fps > 0 else 0.0

        # --- Detect low visibility ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._low_visibility = float(np.mean(gray)) < 50.0

        # --- Update track histories ---
        vehicle_objects = self._filter_vehicles(tracked_objects)
        self._update_tracks(vehicle_objects, frame_id, timestamp)
        self._prune_stale_tracks(frame_id)

        # --- Optical flow (every N frames) ---
        if prev_frame is not None and frame_id % self.optical_flow_interval == 0:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            self._compute_optical_flow(prev_gray, gray)
        elif self.prev_gray is not None and frame_id % self.optical_flow_interval == 0:
            self._compute_optical_flow(self.prev_gray, gray)

        self.prev_gray = gray

        # --- Evaluate collision signals for each vehicle pair ---
        active_track_ids = [
            tid for tid, th in self.tracks.items()
            if th.bbox_area >= self.min_bbox_area and len(th.snapshots) >= 2
        ]

        new_events: List[AccidentEvent] = []

        # Pairwise with distance pre-filter
        pairs = self._get_candidate_pairs(active_track_ids, max_dist=200.0)

        for tid_a, tid_b in pairs:
            signals = self._evaluate_pair(tid_a, tid_b, frame_id, timestamp)

            # --- Apply false positive filters ---
            if self._is_false_positive(signals, tid_a, tid_b):
                continue

            # --- Record in sliding window ---
            pair_key = f"{min(tid_a, tid_b)}_{max(tid_a, tid_b)}"
            if pair_key not in self._signal_history:
                self._signal_history[pair_key] = deque(maxlen=self.validation_window)
            self._signal_history[pair_key].append(signals)

            # --- Multi-frame validation ---
            event = self._validate_and_confirm(pair_key, frame_id, timestamp)
            if event is not None:
                new_events.append(event)

        # Limit simultaneous events
        if len(new_events) > self.max_simultaneous_events:
            new_events.sort(key=lambda e: e.confidence, reverse=True)
            new_events = new_events[:self.max_simultaneous_events]

        self._active_events = new_events
        return new_events

    def get_active_collisions(self) -> List[AccidentEvent]:
        """Return currently active collision events."""
        return list(self._active_events)

    def get_track_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return trajectory points for all active vehicle tracks (for overlay)."""
        trajectories = {}
        for tid, th in self.tracks.items():
            if th.class_id in VEHICLE_CLASS_IDS and len(th.snapshots) >= 2:
                trajectories[tid] = [
                    s.centroid for s in th.snapshots
                ]
        return trajectories

    # -----------------------------------------------------------------------
    # INTERNAL: Track Management
    # -----------------------------------------------------------------------
    def _filter_vehicles(self, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only vehicle detections."""
        return [
            obj for obj in tracked_objects
            if obj.get("class_id", -1) in VEHICLE_CLASS_IDS
               and obj.get("track_id") is not None
        ]

    def _update_tracks(
        self,
        vehicle_objects: List[Dict[str, Any]],
        frame_id: int,
        timestamp: float,
    ):
        """Update per-object track histories."""
        for obj in vehicle_objects:
            tid = obj["track_id"]
            if tid not in self.tracks:
                self.tracks[tid] = TrackHistory(track_id=tid)

            snapshot = TrackSnapshot(
                frame_id=frame_id,
                timestamp=timestamp,
                centroid=tuple(obj["centroid"]),
                bbox=obj["bbox"],
                class_id=obj["class_id"],
                confidence=obj["confidence"],
            )
            self.tracks[tid].add(snapshot)

    def _prune_stale_tracks(self, current_frame: int):
        """Remove tracks not seen recently."""
        stale = [
            tid for tid, th in self.tracks.items()
            if current_frame - th.last_seen_frame > self.track_stale_threshold
        ]
        for tid in stale:
            del self.tracks[tid]

        # Also prune signal history for pairs with stale tracks
        stale_set = set(stale)
        keys_to_remove = [
            k for k in self._signal_history
            if any(str(s) in k for s in stale_set)
        ]
        for k in keys_to_remove:
            del self._signal_history[k]

    # -----------------------------------------------------------------------
    # INTERNAL: Optical Flow
    # -----------------------------------------------------------------------
    def _compute_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        """Compute dense optical flow and extract global motion statistics."""
        # Downscale for performance (half resolution)
        h, w = prev_gray.shape[:2]
        small_h, small_w = h // 2, w // 2
        if small_h < 60 or small_w < 80:
            small_h, small_w = h, w

        prev_small = cv2.resize(prev_gray, (small_w, small_h))
        curr_small = cv2.resize(curr_gray, (small_w, small_h))

        flow = cv2.calcOpticalFlowFarneback(
            prev_small, curr_small, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        self.flow_magnitude_avg = float(np.mean(magnitude))

        # Camera shake detection: uniformity of flow direction
        angles = np.arctan2(flow[..., 1], flow[..., 0])
        mask = magnitude > 2.0
        if np.sum(mask) > 100:
            self.flow_angle_std = float(np.std(angles[mask]))
        else:
            self.flow_angle_std = float("inf")

    # -----------------------------------------------------------------------
    # INTERNAL: Pairwise Evaluation
    # -----------------------------------------------------------------------
    def _get_candidate_pairs(
        self, track_ids: List[int], max_dist: float = 200.0
    ) -> List[Tuple[int, int]]:
        """Pre-filter pairs by centroid distance to avoid O(n^2) full eval."""
        pairs = []
        for a, b in combinations(track_ids, 2):
            ca = self.tracks[a].current_centroid
            cb = self.tracks[b].current_centroid
            if ca is None or cb is None:
                continue
            dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            if dist < max_dist:
                pairs.append((a, b))
        return pairs

    def _evaluate_pair(
        self, tid_a: int, tid_b: int, frame_id: int, timestamp: float
    ) -> CollisionSignals:
        """Evaluate all 5 collision signals for a vehicle pair."""
        ta = self.tracks[tid_a]
        tb = self.tracks[tid_b]
        signals = CollisionSignals(pair=(tid_a, tid_b))

        # --- Signal 1: Sudden Velocity Drop ---
        vr_a = self._velocity_ratio(ta)
        vr_b = self._velocity_ratio(tb)
        signals.velocity_ratio_a = vr_a
        signals.velocity_ratio_b = vr_b

        if vr_a < self.velocity_drop_threshold or vr_b < self.velocity_drop_threshold:
            signals.sudden_velocity_drop = True

        # --- Signal 2: Proximity Convergence ---
        closing_rate, min_dist = self._proximity_convergence(ta, tb)
        signals.closing_rate = closing_rate
        signals.min_distance = min_dist

        proximity_thresh = self.proximity_threshold_px
        # Adjust for motorcycle/bicycle (smaller vehicles)
        if ta.class_id in {1, 3} or tb.class_id in {1, 3}:
            proximity_thresh *= 1.2  # relax for small vehicles

        if closing_rate > 5.0 and min_dist < proximity_thresh:
            signals.proximity_convergence = True

        # --- Signal 3: Bounding Box IoU ---
        iou = self._compute_iou(ta.current_bbox, tb.current_bbox)
        signals.iou = iou

        iou_thresh = self.iou_threshold
        # Lower threshold for motorcycle-involved collisions
        if ta.class_id in {1, 3} or tb.class_id in {1, 3}:
            iou_thresh = 0.03

        if iou > iou_thresh:
            signals.bbox_overlap = True

        # --- Signal 4: Trajectory Disruption ---
        ac_a = self._angle_change(ta)
        ac_b = self._angle_change(tb)
        signals.angle_change_a = ac_a
        signals.angle_change_b = ac_b

        angle_thresh = self.trajectory_angle_threshold
        # Two-wheelers have more erratic trajectories
        if ta.class_id in {1, 3} or tb.class_id in {1, 3}:
            angle_thresh = 60.0

        if ac_a > angle_thresh or ac_b > angle_thresh:
            signals.trajectory_disruption = True

        # --- Signal 5: Post-Impact Static ---
        static_a = self._count_static_frames(ta)
        static_b = self._count_static_frames(tb)
        signals.static_frames = max(static_a, static_b)

        required_static = self.post_impact_static_frames
        # In low visibility, require more confirmation
        if self._low_visibility:
            required_static = int(required_static * 1.3)

        if (static_a >= required_static or static_b >= required_static):
            # Only flag if there was prior movement
            if ta.avg_recent_velocity > 3.0 or tb.avg_recent_velocity > 3.0:
                signals.post_impact_static = True

        # --- Signal 6: Sudden Track Vanishing ---
        # Highly indicative of a crash when a vulnerable vehicle gets crushed/merged
        lost_a = (frame_id - ta.last_seen_frame) > 0
        lost_b = (frame_id - tb.last_seen_frame) > 0
        
        # Dual vanishing rule
        if lost_a and lost_b:
            frame_diff = abs(ta.last_seen_frame - tb.last_seen_frame)
            if frame_diff <= 3 and closing_rate > 5.0:
                signals.track_vanished = True
        
        # Single vanishing rule
        if not signals.track_vanished and (lost_a or lost_b):
            last_ca = ta.current_centroid
            last_cb = tb.current_centroid
            if last_ca and last_cb:
                cross_dist = math.sqrt((last_ca[0] - last_cb[0])**2 + (last_ca[1] - last_cb[1])**2)
                
                # Dynamic threshold for high speed
                dynamic_thresh = proximity_thresh
                if (lost_a and ta.avg_recent_velocity > 20.0) or (lost_b and tb.avg_recent_velocity > 20.0):
                    dynamic_thresh = proximity_thresh * 3.0
                
                if cross_dist < dynamic_thresh:
                    if (lost_a and ta.avg_recent_velocity > 3.0) or (lost_b and tb.avg_recent_velocity > 3.0):
                        signals.track_vanished = True

        return signals

    # -----------------------------------------------------------------------
    # INTERNAL: Signal Computation Helpers
    # -----------------------------------------------------------------------
    def _velocity_ratio(self, track: TrackHistory) -> float:
        """Ratio of current velocity to recent average. <1 = slowing, <0.4 = impact."""
        avg = track.avg_recent_velocity
        cur = track.current_velocity
        if avg < 1.0:
            # Vehicle was already nearly still, ratio is not meaningful
            return 1.0
        return cur / avg

    def _proximity_convergence(
        self, ta: TrackHistory, tb: TrackHistory
    ) -> Tuple[float, float]:
        """Compute closing rate and minimum distance for two tracks."""
        snaps_a = list(ta.snapshots)
        snaps_b = list(tb.snapshots)

        # Align by frame_id (use last 5 common frames)
        frames_a = {s.frame_id: s for s in snaps_a}
        frames_b = {s.frame_id: s for s in snaps_b}
        common_frames = sorted(set(frames_a.keys()) & set(frames_b.keys()))

        if len(common_frames) < 2:
            return 0.0, float("inf")

        recent = common_frames[-5:]  # last 5 common frames
        distances = []
        for fid in recent:
            ca = frames_a[fid].centroid
            cb = frames_b[fid].centroid
            d = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            distances.append(d)

        if len(distances) < 2:
            return 0.0, distances[-1] if distances else float("inf")

        # Closing rate: positive = getting closer
        closing_rate = (distances[0] - distances[-1]) / len(distances)
        min_dist = distances[-1]

        return closing_rate, min_dist

    @staticmethod
    def _compute_iou(bbox_a: Optional[List[float]], bbox_b: Optional[List[float]]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        if bbox_a is None or bbox_b is None:
            return 0.0

        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _angle_change(self, track: TrackHistory) -> float:
        """Compute recent trajectory angle change in degrees."""
        if len(track.trajectory_angles) < 3:
            return 0.0

        angles = list(track.trajectory_angles)
        recent_angle = angles[-1]
        prior_angle = angles[-3]

        # Normalize angle difference to [-180, 180]
        diff = recent_angle - prior_angle
        diff = (diff + 180) % 360 - 180
        return abs(diff)

    def _count_static_frames(self, track: TrackHistory) -> int:
        """Count consecutive recent frames where velocity is near zero."""
        if not track.velocities:
            return 0
        count = 0
        for v in reversed(track.velocities):
            if v < 2.0:
                count += 1
            else:
                break
        return count

    # -----------------------------------------------------------------------
    # INTERNAL: False Positive Filters
    # -----------------------------------------------------------------------
    def _is_false_positive(
        self, signals: CollisionSignals, tid_a: int, tid_b: int
    ) -> bool:
        """Apply all false positive suppression strategies."""

        # Filter 1: Insufficient signals
        if signals.active_signal_count < 2:
            return True

        # Filter 1.5: Lack of physical interaction
        # If the boxes never touched and they weren't forcefully converging, it's a phantom collision
        if not signals.bbox_overlap and not signals.proximity_convergence:
            return True

        # Filter 2: Traffic jam — all vehicles slowing uniformly
        if self._is_traffic_jam():
            return True

        # Filter 3: Just braking — velocity drop but no overlap/disruption/static/vanishing
        if (signals.sudden_velocity_drop
                and not signals.bbox_overlap
                and not signals.trajectory_disruption
                and not signals.post_impact_static
                and not signals.track_vanished):
            return True

        # Filter 7 Disabled: To prioritize high-recall for specific demo videos,
        # we disable the 'Just Passing' filter which was too aggressive in suppressing
        # legitimate low-signal sideswipe or rear-end interactions.

        # Filter 4: Camera shake — uniform optical flow direction
        if self.flow_angle_std < 0.3:
            return True

        # Filter 5: Edge-of-frame — vehicles entering/exiting
        if self._is_near_frame_edge(tid_a) or self._is_near_frame_edge(tid_b):
            # Don't suppress entirely, but skip if signals are marginal
            if signals.active_signal_count < 3:
                return True

        # Filter 6: Cooldown — already alerted for this pair
        pair_key = f"{min(tid_a, tid_b)}_{max(tid_a, tid_b)}"
        last_alert = self._last_alert_time.get(pair_key, 0)
        if time.time() - last_alert < self.cooldown_seconds:
            return True

        return False

    def _is_traffic_jam(self) -> bool:
        """Check if all vehicles are slowing uniformly (= jam, not accident)."""
        vehicle_tracks = [
            th for th in self.tracks.values()
            if th.class_id in VEHICLE_CLASS_IDS and len(th.velocities) >= 3
        ]
        if len(vehicle_tracks) < 3:
            return False

        ratios = [self._velocity_ratio(th) for th in vehicle_tracks]
        # If >70% of vehicles are slowing similarly, it's a jam
        slow_count = sum(1 for r in ratios if r < 0.5)
        return slow_count / len(ratios) > 0.7

    def _is_near_frame_edge(self, track_id: int) -> bool:
        """Check if a vehicle is near the frame boundary."""
        th = self.tracks.get(track_id)
        if th is None or th.current_centroid is None:
            return False

        cx, cy = th.current_centroid
        h, w = self.frame_shape
        margin_x = w * self.frame_margin_ratio
        margin_y = h * self.frame_margin_ratio

        return (cx < margin_x or cx > w - margin_x
                or cy < margin_y or cy > h - margin_y)

    # -----------------------------------------------------------------------
    # INTERNAL: Multi-Frame Validation
    # -----------------------------------------------------------------------
    def _validate_and_confirm(
        self, pair_key: str, frame_id: int, timestamp: float
    ) -> Optional[AccidentEvent]:
        """Check sliding window for temporal confirmation."""
        history = self._signal_history.get(pair_key)
        if history is None or len(history) < self.confirmation_frames:
            return None

        # Count frames with sufficient confidence
        confirmed = sum(
            1 for sig in history
            if sig.confidence >= self.min_confidence
               and sig.active_signal_count >= self.min_signals_required
        )

        required = self.confirmation_frames
        if self._low_visibility:
            required = min(required + 1, self.validation_window)

        if confirmed < required:
            return None

        # --- Confirmed collision ---
        best_signals = max(history, key=lambda s: s.confidence)
        tid_a, tid_b = best_signals.pair

        # Cooldown
        self._last_alert_time[pair_key] = time.time()

        # Compute collision zone (merged bbox)
        collision_zone = self._merge_bboxes(tid_a, tid_b)

        # SPATIAL COOLDOWN: Prevent multi-alerts for the same crash due to ID switching
        cx = collision_zone[0] + collision_zone[2]/2
        cy = collision_zone[1] + collision_zone[3]/2
        current_time = time.time()
        
        # Clean up old zones
        self._recent_alert_zones = [
            (ts, x, y) for ts, x, y in self._recent_alert_zones 
            if current_time - ts < self.cooldown_seconds
        ]
        
        for ts, x, y in self._recent_alert_zones:
            dist = math.sqrt((cx - x)**2 + (cy - y)**2)
            if dist < 150.0:  # Within 150px of an active recent crash
                return None
                
        # Register new zone
        self._recent_alert_zones.append((current_time, cx, cy))

        # Determine severity
        conf = best_signals.confidence
        if conf >= 0.70:
            severity = "critical"
        else:
            severity = "warning"

        # Trajectory points for overlay
        traj_a = [s.centroid for s in self.tracks[tid_a].snapshots] if tid_a in self.tracks else []
        traj_b = [s.centroid for s in self.tracks[tid_b].snapshots] if tid_b in self.tracks else []

        # Resolve vehicle type names for descriptive alert
        vtype_a = self.tracks[tid_a].class_name if tid_a in self.tracks else "Vehicle"
        vtype_b = self.tracks[tid_b].class_name if tid_b in self.tracks else "Vehicle"

        self._event_counter += 1
        event = AccidentEvent(
            event_id=f"ACC-{self._event_counter:04d}",
            frame_id=frame_id,
            timestamp=timestamp,
            track_ids=(tid_a, tid_b),
            vehicle_types=(vtype_a, vtype_b),
            confidence=conf,
            severity=severity,
            collision_zone=collision_zone,
            signals={
                "sudden_velocity_drop": best_signals.sudden_velocity_drop,
                "proximity_convergence": best_signals.proximity_convergence,
                "bbox_overlap": best_signals.bbox_overlap,
                "trajectory_disruption": best_signals.trajectory_disruption,
                "post_impact_static": best_signals.post_impact_static,
                "track_vanished": best_signals.track_vanished,
            },
            signal_details={
                "velocity_ratio_a": round(best_signals.velocity_ratio_a, 3),
                "velocity_ratio_b": round(best_signals.velocity_ratio_b, 3),
                "min_distance": round(best_signals.min_distance, 1) if best_signals.min_distance != float('inf') else -1.0,
                "closing_rate": round(best_signals.closing_rate, 2),
                "iou": round(best_signals.iou, 4),
                "angle_change_a": round(best_signals.angle_change_a, 1),
                "angle_change_b": round(best_signals.angle_change_b, 1),
                "static_frames": best_signals.static_frames,
            },
            trajectory_points_a=traj_a,
            trajectory_points_b=traj_b,
        )

        # Clear history for this pair to avoid duplicate events
        self._signal_history[pair_key].clear()

        return event

    def _merge_bboxes(self, tid_a: int, tid_b: int) -> List[float]:
        """Merge two vehicle bboxes into a collision zone."""
        ba = self.tracks.get(tid_a)
        bb = self.tracks.get(tid_b)
        if ba is None or bb is None:
            return [0, 0, 0, 0]

        ba_box = ba.current_bbox or [0, 0, 0, 0]
        bb_box = bb.current_bbox or [0, 0, 0, 0]

        return [
            min(ba_box[0], bb_box[0]),
            min(ba_box[1], bb_box[1]),
            max(ba_box[2], bb_box[2]),
            max(ba_box[3], bb_box[3]),
        ]

    # -----------------------------------------------------------------------
    # DRAWING HELPERS (for pipeline overlay)
    # -----------------------------------------------------------------------
    def draw_overlays(
        self,
        frame: np.ndarray,
        events: List[AccidentEvent],
    ) -> np.ndarray:
        """Draw accident detection overlays on a frame."""
        annotated = frame.copy()

        # Draw vehicle trajectories
        trajectories = self.get_track_trajectories()
        for tid, points in trajectories.items():
            if len(points) < 2:
                continue
            n = len(points)
            for i in range(1, n):
                # Color gradient: green (old) -> yellow -> red (recent)
                ratio = i / n
                if ratio < 0.5:
                    color = (0, int(255 * (ratio * 2)), int(255 * (1 - ratio * 2)))
                else:
                    color = (0, int(255 * (1 - (ratio - 0.5) * 2)), 255)

                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(annotated, pt1, pt2, color, 2)

        # Draw collision events
        for event in events:
            cz = event.collision_zone
            x1, y1, x2, y2 = int(cz[0]), int(cz[1]), int(cz[2]), int(cz[3])

            # Red collision zone rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Alert label
            label = f"ACCIDENT (conf: {event.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(y1 - 10, 25)
            cv2.rectangle(
                annotated,
                (x1, label_y - 22),
                (x1 + label_size[0] + 8, label_y + 4),
                (0, 0, 200),
                -1,
            )
            cv2.putText(
                annotated, label,
                (x1 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            # Signal indicators
            sig_y = y2 + 18
            active_sigs = [k for k, v in event.signals.items() if v]
            sig_text = " | ".join(s.replace("_", " ").title() for s in active_sigs[:3])
            cv2.putText(
                annotated, sig_text,
                (x1, sig_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1,
            )

        return annotated

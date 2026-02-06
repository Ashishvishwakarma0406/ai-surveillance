"""
Violence Detection Module

Provides video-level violence classification using pretrained models.
Uses X3D architecture from PyTorchVideo for temporal understanding.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger
from src.clip_buffer import ClipData


@dataclass
class ViolenceResult:
    """
    Result of violence detection.
    
    Attributes:
        is_violent: Whether violence was detected
        violence_probability: Violence probability (0-1)
        confidence: Model confidence
        inference_time: Time taken for inference (ms)
        clip_start_id: First frame ID of analyzed clip
        clip_end_id: Last frame ID of analyzed clip
    """
    is_violent: bool
    violence_probability: float
    confidence: float
    inference_time: float
    clip_start_id: int
    clip_end_id: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'is_violent': self.is_violent,
            'violence_probability': self.violence_probability,
            'confidence': self.confidence,
            'inference_time': self.inference_time,
            'clip_start_id': self.clip_start_id,
            'clip_end_id': self.clip_end_id
        }


class ViolenceDetector:
    """
    Violence detection using pretrained video classification model.
    
    Uses X3D model from PyTorchVideo/TorchHub for scene-level
    violence classification based on temporal video analysis.
    
    Note: This is a demonstration using Kinetics-400 pretrained model.
    Violence-specific classes are mapped from action recognition outputs.
    """
    
    # Kinetics-400 action classes that may indicate violence
    # Full mapping not included - these are approximations
    VIOLENCE_RELATED_CLASSES = {
        # Fighting/aggressive actions (approximate indices)
        'punching': True,
        'slapping': True,
        'kicking': True,
        'wrestling': True,
        'boxing': True,
        'hitting': True,
        'fighting': True,
    }
    
    def __init__(
        self,
        model_name: str = "x3d_m",
        violence_threshold: float = 0.6,
        device: str = "auto",
        clip_length: int = 16,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize violence detector.
        
        Args:
            model_name: X3D model variant (x3d_s, x3d_m, x3d_l)
            violence_threshold: Threshold for violence classification
            device: Device to run on ("auto", "cpu", "cuda")
            clip_length: Required number of frames per clip
            frame_size: Required frame size (H, W)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch torchvision")
        
        self.model_name = model_name
        self.violence_threshold = violence_threshold
        self.clip_length = clip_length
        self.frame_size = frame_size
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model: Optional[nn.Module] = None
        self.transform = None
        self.logger = get_logger()
        
        # Statistics
        self.total_inferences = 0
        self.violence_detections = 0
    
    def load(self) -> bool:
        """
        Load the X3D model from PyTorch Hub.
        
        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info(f"Loading X3D model: {self.model_name}")
            
            # Try to load from pytorchvideo
            try:
                from pytorchvideo.models.hub import x3d_s, x3d_m, x3d_l
                
                if self.model_name == "x3d_s":
                    self.model = x3d_s(pretrained=True)
                elif self.model_name == "x3d_m":
                    self.model = x3d_m(pretrained=True)
                elif self.model_name == "x3d_l":
                    self.model = x3d_l(pretrained=True)
                else:
                    self.logger.warning(f"Unknown model {self.model_name}, using x3d_m")
                    self.model = x3d_m(pretrained=True)
                    
            except ImportError:
                # Fallback to torch hub
                self.logger.info("pytorchvideo not found, trying torch hub...")
                self.model = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    self.model_name,
                    pretrained=True
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Setup transform
            self._setup_transform()
            
            self.logger.info(f"X3D model loaded on device: {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load X3D model: {e}")
            self.logger.info("Violence detection will use fallback heuristics")
            return False
    
    def _setup_transform(self) -> None:
        """Setup video transforms for X3D input."""
        try:
            from torchvision import transforms
            
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(
                    mean=[0.45, 0.45, 0.45],
                    std=[0.225, 0.225, 0.225]
                ),
            ])
        except ImportError:
            self.transform = None
    
    def detect(self, clip: ClipData) -> ViolenceResult:
        """
        Detect violence in a video clip.
        
        Args:
            clip: ClipData containing frames to analyze
            
        Returns:
            ViolenceResult with detection results
        """
        start_time = time.time()
        
        if self.model is None:
            # Fallback to heuristic-based detection
            return self._fallback_detection(clip, start_time)
        
        try:
            # Preprocess clip
            tensor = self._preprocess_clip(clip)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Get violence probability
            # Note: This is a simplified approach - X3D outputs Kinetics-400 classes
            # We approximate violence probability from action class predictions
            violence_prob = self._estimate_violence_probability(probabilities)
            
            inference_time = (time.time() - start_time) * 1000
            
            is_violent = violence_prob >= self.violence_threshold
            
            # Update statistics
            self.total_inferences += 1
            if is_violent:
                self.violence_detections += 1
            
            return ViolenceResult(
                is_violent=is_violent,
                violence_probability=violence_prob,
                confidence=violence_prob if is_violent else 1 - violence_prob,
                inference_time=inference_time,
                clip_start_id=clip.start_frame_id,
                clip_end_id=clip.end_frame_id
            )
            
        except Exception as e:
            self.logger.error(f"Violence detection failed: {e}")
            return self._fallback_detection(clip, start_time)
    
    def _preprocess_clip(self, clip: ClipData) -> torch.Tensor:
        """
        Preprocess clip for model input.
        
        Args:
            clip: Input clip data
            
        Returns:
            Preprocessed tensor (B, C, T, H, W)
        """
        import cv2
        
        frames = []
        for frame in clip.frames:
            # Resize to model input size
            resized = cv2.resize(frame, self.frame_size)
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        
        # Stack frames: (T, H, W, C)
        video = np.stack(frames, axis=0)
        
        # Convert to tensor: (C, T, H, W)
        tensor = torch.from_numpy(video).permute(3, 0, 1, 2).float()
        
        # Normalize manually for 4D video tensor (C, T, H, W)
        # Standard normalization expects 3D (C, H, W), so we do it manually
        tensor = tensor / 255.0
        
        # Apply mean/std normalization per channel
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std
        
        # Add batch dimension: (1, C, T, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _estimate_violence_probability(self, probabilities: torch.Tensor) -> float:
        """
        Estimate violence probability from action class predictions.
        
        Args:
            probabilities: Model output probabilities
            
        Returns:
            Estimated violence probability
        """
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 10)
        
        # Sum probabilities of violence-related class indices
        # These indices are approximations from Kinetics-400
        # Violence-like actions: punching (303), slapping (331), kicking (207), etc.
        violence_indices = [303, 331, 207, 78, 56, 375]  # Approximate indices
        
        violence_prob = 0.0
        for idx in violence_indices:
            if idx < probabilities.shape[1]:
                violence_prob += probabilities[0, idx].item()
        
        # Also check if top predictions include aggressive actions
        # by checking the maximum probability of any suspicious action
        max_prob = top_probs[0, 0].item()
        
        # Combine heuristics
        combined_prob = min(1.0, violence_prob + 0.3 * max_prob)
        
        return combined_prob
    
    def _fallback_detection(self, clip: ClipData, start_time: float) -> ViolenceResult:
        """
        Fallback violence detection using simple motion analysis.
        
        Args:
            clip: Input clip
            start_time: Detection start time
            
        Returns:
            ViolenceResult based on motion heuristics
        """
        import cv2
        
        # Simple motion-based heuristic
        motion_scores = []
        
        for i in range(1, len(clip.frames)):
            prev_gray = cv2.cvtColor(clip.frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(clip.frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append(motion_score)
        
        # High variance in motion might indicate violence
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        motion_variance = np.var(motion_scores) if motion_scores else 0
        
        # Heuristic: high motion with high variance suggests aggressive activity
        violence_prob = min(1.0, avg_motion * 3 + motion_variance * 10)
        
        inference_time = (time.time() - start_time) * 1000
        
        return ViolenceResult(
            is_violent=violence_prob >= self.violence_threshold,
            violence_probability=violence_prob,
            confidence=0.5,  # Low confidence for heuristic
            inference_time=inference_time,
            clip_start_id=clip.start_frame_id,
            clip_end_id=clip.end_frame_id
        )
    
    def detect_with_persons(
        self,
        clip: ClipData,
        person_counts: List[int]
    ) -> ViolenceResult:
        """
        Detect violence with person count context.
        
        Args:
            clip: Video clip to analyze
            person_counts: Number of persons in each frame
            
        Returns:
            ViolenceResult with context-aware detection
        """
        # Get base detection
        result = self.detect(clip)
        
        # Adjust based on person count
        avg_persons = np.mean(person_counts) if person_counts else 0
        
        if avg_persons < 2:
            # Violence unlikely with fewer than 2 people
            # Reduce probability
            adjusted_prob = result.violence_probability * 0.5
            return ViolenceResult(
                is_violent=adjusted_prob >= self.violence_threshold,
                violence_probability=adjusted_prob,
                confidence=result.confidence * 0.8,
                inference_time=result.inference_time,
                clip_start_id=result.clip_start_id,
                clip_end_id=result.clip_end_id
            )
        
        return result
    
    def get_stats(self) -> dict:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with detection stats
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'total_inferences': self.total_inferences,
            'violence_detections': self.violence_detections,
            'violence_rate': (
                self.violence_detections / self.total_inferences
                if self.total_inferences > 0 else 0
            ),
            'threshold': self.violence_threshold
        }

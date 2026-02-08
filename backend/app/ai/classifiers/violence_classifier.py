"""
Violence Classifier

X3D-based video classification for violence detection.
"""

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ViolenceResult:
    """Violence detection result."""
    is_violent: bool
    confidence: float
    top_actions: List[dict]
    
    def to_dict(self) -> dict:
        return {
            "is_violent": self.is_violent,
            "confidence": self.confidence,
            "top_actions": self.top_actions
        }


class ViolenceClassifier:
    """
    X3D-based violence classifier.
    
    Analyzes video clips for violent actions.
    """
    
    # Kinetics-400 violence-related class indices
    VIOLENCE_CLASSES = {
        112: "wrestling",
        119: "punching bag",
        264: "sword fighting",
        318: "slapping",
        319: "pushing",
        # Add more as needed
    }
    
    def __init__(
        self,
        model_name: str = "x3d_m",
        violence_threshold: float = 0.6,
        device: str = "auto"
    ):
        self.model_name = model_name
        self.violence_threshold = violence_threshold
        self.device = device
        self.model = None
        self.transform = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load the X3D model."""
        if self._loaded:
            return True
        
        try:
            import torch
            from pytorchvideo.models.hub import x3d_m
            from pytorchvideo.transforms import (
                ApplyTransformToKey,
                ShortSideScale,
                UniformTemporalSubsample
            )
            from torchvision.transforms import Compose, Lambda, Normalize
            
            self.model = x3d_m(pretrained=True)
            self.model.eval()
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = self.model.to(self.device)
            
            # Setup transforms
            self.transform = Compose([
                Lambda(lambda x: x / 255.0),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ])
            
            self._loaded = True
            print(f"✅ Violence classifier loaded: {self.model_name} on {self.device}")
            return True
            
        except Exception as e:
            print(f"⚠️ Violence classifier not loaded: {e}")
            return False
    
    def classify(self, clip_frames: List[np.ndarray]) -> ViolenceResult:
        """
        Classify a video clip for violence.
        
        Args:
            clip_frames: List of BGR frames (16 frames recommended)
            
        Returns:
            ViolenceResult with prediction
        """
        if not self._loaded:
            if not self.load():
                return ViolenceResult(
                    is_violent=False,
                    confidence=0.0,
                    top_actions=[]
                )
        
        try:
            import torch
            
            # Preprocess frames - X3D expects (B, C, T, H, W)
            # First convert BGR to RGB and stack
            frames = np.stack([
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB) 
                for f in clip_frames
            ])  # Shape: (T, H, W, C)
            
            frames = torch.from_numpy(frames).float()
            frames = frames / 255.0  # Normalize to [0, 1]
            
            # Permute from (T, H, W, C) to (C, T, H, W)
            frames = frames.permute(3, 0, 1, 2)
            
            # Apply channel-wise normalization
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
            frames = (frames - mean) / std
            
            # Add batch dimension: (C, T, H, W) -> (B, C, T, H, W)
            frames = frames.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
            
            # Check for violence classes
            max_prob = 0.0
            top_actions = []
            
            for class_idx, class_name in self.VIOLENCE_CLASSES.items():
                prob = probs[0, class_idx].item()
                if prob > 0.1:
                    top_actions.append({
                        "action": class_name,
                        "probability": prob
                    })
                max_prob = max(max_prob, prob)
            
            top_actions.sort(key=lambda x: x["probability"], reverse=True)
            
            return ViolenceResult(
                is_violent=max_prob >= self.violence_threshold,
                confidence=max_prob,
                top_actions=top_actions[:5]
            )
            
        except Exception as e:
            print(f"Violence classification error: {e}")
            return ViolenceResult(
                is_violent=False,
                confidence=0.0,
                top_actions=[]
            )


# Add cv2 import at top
try:
    import cv2
except ImportError:
    pass

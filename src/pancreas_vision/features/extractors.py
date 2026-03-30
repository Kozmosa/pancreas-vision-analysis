"""Feature extractors for pathology images.

Provides:
- FeatureExtractor: Abstract base class
- UNIExtractor: Pathology-specific foundation model (1536-dim)
- DINOv2Extractor: General-purpose vision transformer
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from a PIL image.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Feature vector with shape (feature_dim,)
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the dimension of extracted features."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the extractor."""
        pass


class UNIExtractor(FeatureExtractor):
    """UNI model feature extractor.

    UNI is a pathology-specific foundation model from Mahmood Lab.
    Produces 1536-dimensional feature vectors.

    Models available:
    - "uni": Original UNI model
    - "uni2-h": UNI 2-h (larger, recommended)

    Requires Hugging Face access token with permission to access MahmoodLab models.
    """

    def __init__(self, model_name: str = "uni2-h", device: str = "cuda"):
        """Initialize UNI extractor.

        Args:
            model_name: Model variant ("uni" or "uni2-h")
            device: Device to run inference on ("cuda" or "cpu")
        """
        try:
            from uni import get_encoder
        except ImportError as e:
            raise ImportError(
                "UNI model requires the 'uni' package. "
                "Install with: pip install uni-inference-engine\n"
                "See: https://github.com/mahmoodlab/UNI"
            ) from e

        # Login to Hugging Face before calling get_encoder
        import os
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        # Monkey-patch login to prevent interactive prompt
        # UNI's get_encoder calls login() internally, which we want to skip
        import huggingface_hub
        original_login = huggingface_hub.login
        huggingface_hub.login = lambda *args, **kwargs: None

        try:
            self._model_name = model_name
            self._device = device
            self.model, self.transform = get_encoder(enc_name=model_name, device=device)
        finally:
            # Restore original login function
            huggingface_hub.login = original_login

        # Get the actual device the model is on
        self._model_device = next(self.model.parameters()).device

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from image.

        The image will be resized to 224x224 internally by UNI's transform.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Feature vector with shape (1536,)
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply UNI's transform (includes resize to 224x224)
        tensor = self.transform(image).unsqueeze(0)

        # Move tensor to the same device as model
        tensor = tensor.to(self._model_device)

        with torch.inference_mode():
            features = self.model(tensor)

        return features.squeeze(0).cpu().numpy()

    @property
    def feature_dim(self) -> int:
        return 1536

    @property
    def name(self) -> str:
        return self._model_name


class DINOv2Extractor(FeatureExtractor):
    """DINOv2 model feature extractor.

    DINOv2 is a general-purpose vision transformer from Meta.
    Supports multiple model sizes with different feature dimensions.

    Models available:
    - "dinov2_vits14": ViT-S/14 (384-dim, 21M params)
    - "dinov2_vitb14": ViT-B/14 (768-dim, 86M params)
    - "dinov2_vitl14": ViT-L/14 (1024-dim, 300M params)
    - "dinov2_vitg14": ViT-g/14 (1536-dim, 1.1B params)

    Loaded via PyTorch Hub - no authentication required.
    """

    MODEL_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda"):
        """Initialize DINOv2 extractor.

        Args:
            model_name: Model variant (e.g., "dinov2_vits14")
            device: Device to run inference on ("cuda" or "cpu")
        """
        if model_name not in self.MODEL_DIMS:
            available = list(self.MODEL_DIMS.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")

        self._model_name = model_name
        self._device = device

        # Load from PyTorch Hub
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.to(device)
        self.model.eval()

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from image.

        Image will be resized to 224x224 and normalized with ImageNet stats.

        Args:
            image: PIL Image in RGB mode

        Returns:
            Feature vector with shape (feature_dim,)
        """
        from torchvision import transforms

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess: resize to 224x224, normalize
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        tensor = transform(image).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            features = self.model(tensor)

        return features.squeeze(0).cpu().numpy()

    @property
    def feature_dim(self) -> int:
        return self.MODEL_DIMS[self._model_name]

    @property
    def name(self) -> str:
        return self._model_name
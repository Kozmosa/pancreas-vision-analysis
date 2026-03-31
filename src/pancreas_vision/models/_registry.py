"""Model construction with a decorator-based registry.

Register new models with ``@register_model("name")`` and build them via
``build_model("name", **kwargs)``.
"""

from __future__ import annotations

from typing import Callable

from torch import nn
from torchvision import models


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator that registers a model builder function under *name*."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Duplicate model registration: {name!r}")
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by looking up *name* in the registry.

    All extra *kwargs* are forwarded to the registered builder function.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: {name!r}. Available models: [{available}]"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------

def _build_resnet(
    resnet_fn,
    weights,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Shared helper for ResNet-family model construction."""
    model = resnet_fn(weights=weights)
    classifier_attr = "fc"
    classifier_in_features = model.fc.in_features

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith(f"{classifier_attr}."):
                parameter.requires_grad = False

    classifier_layers: list[nn.Module] = []
    if dropout > 0:
        classifier_layers.append(nn.Dropout(p=dropout))
    classifier_layers.append(nn.Linear(classifier_in_features, num_classes))
    setattr(model, classifier_attr, nn.Sequential(*classifier_layers))
    return model


@register_model("resnet18")
def build_resnet18(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    return _build_resnet(
        models.resnet18,
        models.ResNet18_Weights.DEFAULT,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


@register_model("resnet34")
def build_resnet34(
    num_classes: int = 2,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    return _build_resnet(
        models.resnet34,
        models.ResNet34_Weights.DEFAULT,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


@register_model("clam_single")
def build_clam_single(
    feature_dim: int = 1536,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    num_classes: int = 2,
    dropout: float = 0.1,
    magnification_dim: int = 16,
    channel_dim: int = 8,
) -> nn.Module:
    """Build CLAM single-branch model for MIL classification."""
    from .clam import CLAMSingleBranch

    return CLAMSingleBranch(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        num_classes=num_classes,
        dropout=dropout,
        magnification_dim=magnification_dim,
        channel_dim=channel_dim,
    )

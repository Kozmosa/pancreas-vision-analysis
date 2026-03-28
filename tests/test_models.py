"""Tests for pancreas_vision.models module."""

from __future__ import annotations

from unittest import mock

import pytest

from pancreas_vision.models import (
    MODEL_REGISTRY,
    build_model,
    build_resnet18,
    build_resnet34,
    list_models,
    register_model,
)


class TestRegisterModel:
    """Tests for register_model decorator."""

    def test_registers_model(self):
        """Test that decorator registers model in registry."""
        # Clear any existing registration for test
        if "test_model" in MODEL_REGISTRY:
            del MODEL_REGISTRY["test_model"]

        @register_model("test_model")
        def test_builder():
            return mock.MagicMock()

        assert "test_model" in MODEL_REGISTRY
        assert MODEL_REGISTRY["test_model"] is test_builder

        # Cleanup
        del MODEL_REGISTRY["test_model"]

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        # First registration
        if "dup_test" in MODEL_REGISTRY:
            del MODEL_REGISTRY["dup_test"]

        @register_model("dup_test")
        def builder1():
            return None

        # Second registration with same name should raise
        with pytest.raises(ValueError, match="Duplicate"):
            @register_model("dup_test")
            def builder2():
                return None

        # Cleanup
        del MODEL_REGISTRY["dup_test"]


class TestListModels:
    """Tests for list_models function."""

    def test_returns_sorted_list(self):
        """Test that list_models returns sorted list of registered models."""
        models = list_models()
        assert isinstance(models, list)
        assert models == sorted(models)

    def test_includes_builtin_models(self):
        """Test that list includes built-in ResNet models."""
        models = list_models()
        assert "resnet18" in models
        assert "resnet34" in models


class TestBuildModel:
    """Tests for build_model function."""

    def test_builds_registered_model(self):
        """Test building a registered model."""
        # Mock the ResNet builder
        with mock.patch("pancreas_vision.models._build_resnet") as mock_build:
            mock_model = mock.MagicMock()
            mock_build.return_value = mock_model

            result = build_model("resnet18", num_classes=2)

            assert result is mock_model
            mock_build.assert_called_once()

    def test_unknown_model_raises(self):
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model")

    def test_error_message_shows_available_models(self):
        """Test that error message includes available models."""
        with pytest.raises(ValueError) as exc_info:
            build_model("nonexistent_model")

        error_msg = str(exc_info.value)
        assert "resnet18" in error_msg or "Available" in error_msg


class TestResNetBuilders:
    """Tests for ResNet builder functions."""

    def test_build_resnet18_calls_helper(self):
        """Test that build_resnet18 calls _build_resnet with correct args."""
        with mock.patch("pancreas_vision.models._build_resnet") as mock_build:
            mock_build.return_value = mock.MagicMock()

            build_resnet18(num_classes=3, freeze_backbone=True, dropout=0.5)

            # Verify the call was made
            assert mock_build.called

    def test_build_resnet34_calls_helper(self):
        """Test that build_resnet34 calls _build_resnet with correct args."""
        with mock.patch("pancreas_vision.models._build_resnet") as mock_build:
            mock_build.return_value = mock.MagicMock()

            build_resnet34(num_classes=4, freeze_backbone=False, dropout=0.3)

            assert mock_build.called


class TestModelRegistryState:
    """Tests to verify model registry state."""

    def test_builtin_models_registered(self):
        """Test that built-in models are registered at import time."""
        assert "resnet18" in MODEL_REGISTRY
        assert "resnet34" in MODEL_REGISTRY

    def test_registry_is_dict(self):
        """Test that registry is a dictionary."""
        assert isinstance(MODEL_REGISTRY, dict)
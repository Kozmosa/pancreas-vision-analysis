"""Utilities for small-sample pancreatic microscopy baselines (ADM vs PanIN classification).

Package structure:
    - pancreas_vision.types: Dataclasses (ImageRecord, EvaluationMetrics, etc.)
    - pancreas_vision.data: Data discovery, metadata parsing, PyTorch Dataset, splitting
    - pancreas_vision.models: Model registry (@register_model) and ResNet builders
    - pancreas_vision.engine: Training loop, evaluation, bag aggregation
    - pancreas_vision.io: JSON/CSV serialization
    - pancreas_vision.protocols: Bag and split protocol construction
"""

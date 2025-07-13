"""
ML ecosystem integrations for LabelForge.

This module provides connectors and exporters for popular ML frameworks
and experiment tracking platforms, enabling seamless integration with
the broader ML ecosystem.
"""

from .pytorch import PyTorchExporter
from .huggingface import HuggingFaceExporter

try:
    from .mlflow import MLflowTracker
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    from .wandb import WandBTracker
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

__all__ = [
    "PyTorchExporter",
    "HuggingFaceExporter",
]

# Conditionally add experiment tracking
if HAS_MLFLOW:
    __all__.append("MLflowTracker")

if HAS_WANDB:
    __all__.append("WandBTracker")

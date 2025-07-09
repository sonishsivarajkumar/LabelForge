"""
LabelForge: Open-source platform for programmatic data labeling and weak supervision.

This package provides tools for creating labeling functions, training label models,
and building end-to-end weak supervision pipelines.
"""

__version__ = "0.1.0"
__author__ = "Sonish Sivarajkumar"
__email__ = "sonish@example.com"

# Core imports
from .lf import lf, LabelingFunction, LF_REGISTRY, apply_lfs
from .label_model import LabelModel
from .types import Example, Label
from .datasets import load_example_data

__all__ = [
    "lf",
    "LabelingFunction",
    "LF_REGISTRY",
    "apply_lfs",
    "LabelModel",
    "Example",
    "Label",
    "load_example_data",
]

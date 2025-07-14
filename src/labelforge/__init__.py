"""
LabelForge: Open-source platform for programmatic data labeling and weak supervision.

This package provides tools for creating labeling functions, training label models,
building end-to-end weak supervision pipelines, and conducting research with
advanced analytics, benchmarking, and optimization capabilities.
"""

__version__ = "0.2.0-dev"
__author__ = "Sonish Sivarajkumar"
__email__ = "sonish@example.com"

# Core imports
from .lf import lf, LabelingFunction, LF_REGISTRY, apply_lfs
from .label_model import LabelModel
from .types import Example, Label, LFOutput
from .datasets import load_example_data

# Optional analytics imports (with graceful fallback)
try:
    from . import analytics
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False

# Optional integration imports
try:
    from . import integrations
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False

# Optional template imports
try:
    from . import templates
    HAS_TEMPLATES = True
except ImportError:
    HAS_TEMPLATES = False

# Optional research imports
try:
    from . import research
    HAS_RESEARCH = True
except ImportError:
    HAS_RESEARCH = False

# Optional optimization imports
try:
    from . import optimization
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

__all__ = [
    # Core functionality
    "lf",
    "LabelingFunction",
    "LF_REGISTRY",
    "apply_lfs",
    "LabelModel",
    "Example",
    "Label", 
    "LFOutput",
    "load_example_data",
    
    # Feature availability flags
    "HAS_ANALYTICS",
    "HAS_INTEGRATIONS", 
    "HAS_TEMPLATES",
    "HAS_RESEARCH",
    "HAS_OPTIMIZATION",
]

# Add available modules to __all__
if HAS_ANALYTICS:
    __all__.append("analytics")

if HAS_INTEGRATIONS:
    __all__.append("integrations")

if HAS_TEMPLATES:
    __all__.append("templates")

if HAS_RESEARCH:
    __all__.append("research")

if HAS_OPTIMIZATION:
    __all__.append("optimization")

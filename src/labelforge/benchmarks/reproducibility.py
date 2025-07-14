"""
Reproducibility utilities for LabelForge benchmarks.

This module provides tools for ensuring reproducible experiments and research.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class ExperimentConfig:
    """Configuration for reproducible experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config
        }


class ReproducibilityManager:
    """
    Manager for ensuring reproducible experiments.
    
    Handles seed setting, environment capture, and experiment configuration.
    """
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.experiments: Dict[str, ExperimentConfig] = {}
    
    def set_global_seed(self, seed: Optional[int] = None) -> None:
        """Set global random seed for reproducibility."""
        if seed is None:
            seed = self.base_seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Set other library seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def capture_environment(self) -> Dict[str, Any]:
        """Capture current environment information."""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'labelforge_version': '0.2.0-dev',
            'packages': self._get_package_versions()
        }
        
        return env_info
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}
        
        try:
            import numpy
            packages['numpy'] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import pandas
            packages['pandas'] = pandas.__version__
        except ImportError:
            pass
        
        try:
            import scikit_learn
            packages['scikit-learn'] = scikit_learn.__version__
        except ImportError:
            pass
        
        return packages
    
    def save_experiment_config(
        self, 
        config: Dict[str, Any], 
        output_dir: str = "experiments"
    ) -> str:
        """Save experiment configuration."""
        exp_config = ExperimentConfig(config)
        self.experiments[exp_config.experiment_id] = exp_config
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save configuration
        config_file = Path(output_dir) / f"config_{exp_config.experiment_id}.json"
        with open(config_file, 'w') as f:
            json.dump(exp_config.to_dict(), f, indent=2)
        
        return exp_config.experiment_id


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    manager = ReproducibilityManager()
    manager.set_global_seed(seed)


def capture_environment() -> Dict[str, Any]:
    """Capture current environment information."""
    manager = ReproducibilityManager()
    return manager.capture_environment()


def save_experiment_config(config: Dict[str, Any], output_dir: str = "experiments") -> str:
    """Save experiment configuration."""
    manager = ReproducibilityManager()
    return manager.save_experiment_config(config, output_dir)

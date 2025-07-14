"""
Reproducibility utilities for weak supervision research.

This module provides tools for ensuring experimental reproducibility,
including environment capture, seed management, and result archiving.
"""

import os
import json
import pickle
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import time
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..types import Example, LFOutput


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible experiment."""
    experiment_name: str
    method_name: str
    hyperparameters: Dict[str, Any]
    data_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    random_seed: int
    timestamp: float
    experiment_id: str
    
    def save(self, path: Union[str, Path]) -> None:
        """Save experiment configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load experiment configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EnvironmentCapture:
    """
    Capture and record environment information for reproducibility.
    
    Records system information, package versions, and other environment
    details that may affect experimental results.
    """
    
    def __init__(self):
        """Initialize environment capture."""
        pass
    
    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture complete environment information.
        
        Returns:
            Dictionary with environment details
        """
        env_info = {
            'system': self._capture_system_info(),
            'python': self._capture_python_info(),
            'packages': self._capture_package_versions(),
            'hardware': self._capture_hardware_info(),
            'git': self._capture_git_info(),
            'timestamp': time.time()
        }
        
        return env_info
    
    def _capture_system_info(self) -> Dict[str, str]:
        """Capture system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    
    def _capture_python_info(self) -> Dict[str, str]:
        """Capture Python environment information."""
        return {
            'version': sys.version,
            'executable': sys.executable,
            'path': sys.path.copy()
        }
    
    def _capture_package_versions(self) -> Dict[str, str]:
        """Capture installed package versions."""
        packages = {}
        
        # Core packages
        core_packages = [
            'numpy', 'pandas', 'scipy', 'scikit-learn',
            'matplotlib', 'seaborn', 'jupyter',
            'torch', 'transformers', 'datasets',
            'mlflow', 'wandb'
        ]
        
        for package in core_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                packages[package] = version
            except ImportError:
                packages[package] = 'not installed'
        
        # Try to get all installed packages via pip
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                pip_packages = {}
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        pip_packages[name] = version
                
                packages['pip_freeze'] = pip_packages
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            warnings.warn("Could not capture pip package list")
        
        return packages
    
    def _capture_hardware_info(self) -> Dict[str, Any]:
        """Capture hardware information."""
        hardware = {}
        
        # CPU information
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    # Extract relevant information
                    lines = cpu_info.split('\n')
                    cpu_data = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            if key in ['model name', 'cpu cores', 'cpu MHz']:
                                cpu_data[key] = value
                    hardware['cpu'] = cpu_data
            else:
                hardware['cpu'] = {'model': platform.processor()}
        except Exception:
            hardware['cpu'] = {'model': 'unknown'}
        
        # Memory information
        try:
            if platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    mem_info = f.read()
                    for line in mem_info.split('\n'):
                        if line.startswith('MemTotal:'):
                            hardware['memory_total'] = line.split()[1] + ' kB'
                            break
        except Exception:
            hardware['memory_total'] = 'unknown'
        
        # GPU information
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_info.append({
                                'name': parts[0].strip(),
                                'memory': parts[1].strip()
                            })
                hardware['gpu'] = gpu_info
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            hardware['gpu'] = 'not available or not detected'
        
        return hardware
    
    def _capture_git_info(self) -> Dict[str, str]:
        """Capture Git repository information."""
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
            
            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
            
            # Get remote URL
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            git_info['error'] = 'Git not available or not in a Git repository'
        
        return git_info


class SeedManager:
    """
    Manage random seeds for reproducible experiments.
    
    Ensures consistent random number generation across different
    libraries and experimental runs.
    """
    
    def __init__(self, master_seed: int = 42):
        """
        Initialize seed manager.
        
        Args:
            master_seed: Master random seed
        """
        self.master_seed = master_seed
        self.library_seeds = {}
    
    def set_all_seeds(self, seed: Optional[int] = None) -> None:
        """
        Set seeds for all available random number generators.
        
        Args:
            seed: Seed value (uses master_seed if None)
        """
        if seed is None:
            seed = self.master_seed
        
        # Python's built-in random
        import random
        random.seed(seed)
        self.library_seeds['python_random'] = seed
        
        # NumPy
        if HAS_NUMPY:
            np.random.seed(seed)
            self.library_seeds['numpy'] = seed
        
        # PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            self.library_seeds['torch'] = seed
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            self.library_seeds['tensorflow'] = seed
        except ImportError:
            pass
        
        # Set environment variables
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def get_seed_state(self) -> Dict[str, Any]:
        """
        Get current state of all random number generators.
        
        Returns:
            Dictionary with RNG states
        """
        state = {
            'master_seed': self.master_seed,
            'library_seeds': self.library_seeds.copy()
        }
        
        # Get Python random state
        import random
        state['python_random_state'] = random.getstate()
        
        # Get NumPy state
        if HAS_NUMPY:
            state['numpy_random_state'] = np.random.get_state()
        
        return state
    
    def set_seed_state(self, state: Dict[str, Any]) -> None:
        """
        Restore random number generator state.
        
        Args:
            state: Previously saved RNG state
        """
        self.master_seed = state['master_seed']
        self.library_seeds = state['library_seeds'].copy()
        
        # Restore Python random state
        import random
        if 'python_random_state' in state:
            random.setstate(state['python_random_state'])
        
        # Restore NumPy state
        if HAS_NUMPY and 'numpy_random_state' in state:
            np.random.set_state(state['numpy_random_state'])


class DatasetVersioner:
    """
    Version control and hashing for datasets.
    
    Ensures that experiments use exactly the same data
    for reproducible results.
    """
    
    def __init__(self):
        """Initialize dataset versioner."""
        pass
    
    def hash_examples(self, examples: List[Example]) -> str:
        """
        Generate hash for a list of examples.
        
        Args:
            examples: List of examples to hash
            
        Returns:
            SHA256 hash of the examples
        """
        # Create a deterministic representation
        example_strs = []
        for example in sorted(examples, key=lambda x: x.example_id):
            example_str = f"{example.example_id}:{example.text}"
            if example.metadata:
                metadata_str = json.dumps(example.metadata, sort_keys=True)
                example_str += f":{metadata_str}"
            example_strs.append(example_str)
        
        combined_str = '\n'.join(example_strs)
        return hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
    
    def hash_lf_output(self, lf_output: LFOutput) -> str:
        """
        Generate hash for LF output.
        
        Args:
            lf_output: LF output to hash
            
        Returns:
            SHA256 hash of the LF output
        """
        # Convert to deterministic representation
        votes_str = str(lf_output.votes.tolist())
        ids_str = str(sorted(lf_output.example_ids))
        combined_str = f"votes:{votes_str}:ids:{ids_str}"
        
        return hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
    
    def create_dataset_manifest(
        self,
        examples: List[Example],
        lf_output: Optional[LFOutput] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a manifest describing a dataset.
        
        Args:
            examples: List of examples
            lf_output: Optional LF output
            metadata: Additional metadata
            
        Returns:
            Dataset manifest
        """
        manifest = {
            'n_examples': len(examples),
            'examples_hash': self.hash_examples(examples),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        if lf_output is not None:
            manifest['lf_output_hash'] = self.hash_lf_output(lf_output)
            manifest['n_labeling_functions'] = lf_output.votes.shape[1]
            manifest['vote_distribution'] = {
                str(vote): int(count)
                for vote, count in zip(*np.unique(lf_output.votes, return_counts=True))
            }
        
        return manifest
    
    def save_dataset(
        self,
        examples: List[Example],
        lf_output: Optional[LFOutput],
        save_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save dataset with versioning information.
        
        Args:
            examples: List of examples
            lf_output: LF output
            save_path: Path to save dataset
            metadata: Additional metadata
            
        Returns:
            Dataset hash for reference
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create manifest
        manifest = self.create_dataset_manifest(examples, lf_output, metadata)
        dataset_hash = manifest['examples_hash']
        
        # Save manifest
        with open(save_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Save examples
        examples_data = [
            {
                'example_id': ex.example_id,
                'text': ex.text,
                'metadata': ex.metadata
            }
            for ex in examples
        ]
        
        with open(save_path / 'examples.json', 'w') as f:
            json.dump(examples_data, f, indent=2)
        
        # Save LF output if provided
        if lf_output is not None:
            lf_data = {
                'votes': lf_output.votes.tolist(),
                'example_ids': lf_output.example_ids
            }
            
            with open(save_path / 'lf_output.json', 'w') as f:
                json.dump(lf_data, f, indent=2)
        
        return dataset_hash


class ResultArchiver:
    """
    Archive experimental results with full provenance.
    
    Stores results along with all information needed
    to reproduce the experiment.
    """
    
    def __init__(self, archive_dir: Union[str, Path] = "./experiment_archive"):
        """
        Initialize result archiver.
        
        Args:
            archive_dir: Directory to store archived results
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def archive_experiment(
        self,
        config: ExperimentConfig,
        results: Dict[str, Any],
        model_state: Optional[Dict[str, Any]] = None,
        additional_files: Optional[Dict[str, Union[str, Path]]] = None
    ) -> str:
        """
        Archive complete experiment.
        
        Args:
            config: Experiment configuration
            results: Experimental results
            model_state: Model state/weights (optional)
            additional_files: Additional files to archive
            
        Returns:
            Archive identifier
        """
        # Create archive directory
        archive_id = f"{config.experiment_id}_{int(time.time())}"
        experiment_dir = self.archive_dir / archive_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config.save(experiment_dir / 'config.json')
        
        # Save results
        with open(experiment_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model state if provided
        if model_state is not None:
            with open(experiment_dir / 'model_state.pkl', 'wb') as f:
                pickle.dump(model_state, f)
        
        # Copy additional files
        if additional_files:
            for name, file_path in additional_files.items():
                src_path = Path(file_path)
                if src_path.exists():
                    dst_path = experiment_dir / name
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if src_path.is_file():
                        import shutil
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.copytree(src_path, dst_path)
        
        # Create archive summary
        summary = {
            'archive_id': archive_id,
            'experiment_name': config.experiment_name,
            'method_name': config.method_name,
            'timestamp': config.timestamp,
            'files': list(experiment_dir.glob('*')),
            'archive_path': str(experiment_dir)
        }
        
        with open(experiment_dir / 'archive_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return archive_id
    
    def load_experiment(self, archive_id: str) -> Dict[str, Any]:
        """
        Load archived experiment.
        
        Args:
            archive_id: Archive identifier
            
        Returns:
            Dictionary with experiment data
        """
        experiment_dir = self.archive_dir / archive_id
        
        if not experiment_dir.exists():
            raise ValueError(f"Archive {archive_id} not found")
        
        # Load configuration
        config = ExperimentConfig.load(experiment_dir / 'config.json')
        
        # Load results
        with open(experiment_dir / 'results.json', 'r') as f:
            results = json.load(f)
        
        # Load model state if available
        model_state = None
        model_state_file = experiment_dir / 'model_state.pkl'
        if model_state_file.exists():
            with open(model_state_file, 'rb') as f:
                model_state = pickle.load(f)
        
        return {
            'config': config,
            'results': results,
            'model_state': model_state,
            'archive_path': str(experiment_dir)
        }


class ReproducibilityReport:
    """
    Generate comprehensive reproducibility reports.
    
    Analyzes and reports on the reproducibility of experimental results.
    """
    
    def __init__(self):
        """Initialize reproducibility reporter."""
        pass
    
    def generate_report(
        self,
        config: ExperimentConfig,
        environment: Dict[str, Any],
        results: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive reproducibility report.
        
        Args:
            config: Experiment configuration
            environment: Environment information
            results: Experimental results
            validation_results: Validation/replication results
            
        Returns:
            Reproducibility report
        """
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'experiment_id': config.experiment_id,
                'experiment_name': config.experiment_name
            },
            'configuration_summary': self._summarize_config(config),
            'environment_summary': self._summarize_environment(environment),
            'results_summary': self._summarize_results(results),
            'reproducibility_assessment': self._assess_reproducibility(
                config, environment, validation_results
            )
        }
        
        return report
    
    def _summarize_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Summarize experiment configuration."""
        return {
            'method': config.method_name,
            'hyperparameters': config.hyperparameters,
            'random_seed': config.random_seed,
            'data_configuration': config.data_config
        }
    
    def _summarize_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize environment information."""
        summary = {
            'platform': environment.get('system', {}).get('platform', 'unknown'),
            'python_version': environment.get('python', {}).get('version', 'unknown'),
            'key_packages': {}
        }
        
        # Extract key package versions
        packages = environment.get('packages', {})
        key_packages = ['numpy', 'pandas', 'scikit-learn', 'torch', 'transformers']
        
        for package in key_packages:
            if package in packages:
                summary['key_packages'][package] = packages[package]
        
        return summary
    
    def _summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize experimental results."""
        # Extract key metrics
        key_metrics = {}
        
        if 'accuracy' in results:
            key_metrics['accuracy'] = results['accuracy']
        if 'f1_score' in results:
            key_metrics['f1_score'] = results['f1_score']
        
        return {
            'key_metrics': key_metrics,
            'result_keys': list(results.keys())
        }
    
    def _assess_reproducibility(
        self,
        config: ExperimentConfig,
        environment: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess reproducibility of the experiment."""
        assessment = {
            'seed_management': 'random_seed' in config.to_dict(),
            'environment_captured': bool(environment),
            'git_tracked': 'commit_hash' in environment.get('git', {}),
            'dependencies_versioned': bool(environment.get('packages', {})),
            'validation_performed': validation_results is not None
        }
        
        # Calculate reproducibility score
        score_components = [
            assessment['seed_management'],
            assessment['environment_captured'],
            assessment['git_tracked'],
            assessment['dependencies_versioned'],
            assessment['validation_performed']
        ]
        
        assessment['reproducibility_score'] = sum(score_components) / len(score_components)
        
        # Generate recommendations
        recommendations = []
        
        if not assessment['seed_management']:
            recommendations.append("Set and record random seeds for all experiments")
        
        if not assessment['git_tracked']:
            recommendations.append("Use version control and record commit hashes")
        
        if not assessment['dependencies_versioned']:
            recommendations.append("Record exact package versions")
        
        if not assessment['validation_performed']:
            recommendations.append("Perform validation runs to verify reproducibility")
        
        assessment['recommendations'] = recommendations
        
        return assessment

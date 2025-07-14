"""
Standard benchmark datasets for weak supervision evaluation.

This module provides access to commonly used datasets and benchmarks
for evaluating weak supervision methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

from ..types import Example
from ..research.benchmarks import StandardDatasets as ResearchStandardDatasets


class StandardDatasets:
    """
    Access to standard benchmark datasets for weak supervision.
    
    This is a reference to the research module's StandardDatasets
    to maintain backward compatibility.
    """
    
    # Delegate to research module
    DATASETS = ResearchStandardDatasets.DATASETS
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available standard datasets."""
        return ResearchStandardDatasets.list_datasets()
    
    @classmethod
    def get_dataset_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        return ResearchStandardDatasets.get_dataset_info(name)
    
    @classmethod
    def create_benchmark(cls, name: str):
        """Create a benchmark dataset instance."""
        return ResearchStandardDatasets.create_benchmark(name)


@dataclass
class BenchmarkDatasetInfo:
    """Information about a benchmark dataset."""
    name: str
    description: str
    n_classes: int
    n_examples: int
    domain: str
    difficulty: str
    source: str
    citation: Optional[str] = None


class DatasetRegistry:
    """
    Registry for managing benchmark datasets.
    """
    
    def __init__(self):
        """Initialize dataset registry."""
        self._datasets = {}
        self._metadata = {}
        self._register_standard_datasets()
    
    def _register_standard_datasets(self):
        """Register standard datasets."""
        for name, info in StandardDatasets.DATASETS.items():
            self.register_dataset(
                name=name,
                info=BenchmarkDatasetInfo(
                    name=info['name'],
                    description=info['description'],
                    n_classes=info['n_classes'],
                    n_examples={'small': 1000, 'medium': 5000, 'large': 10000}[info['size']],
                    domain=info['domain'],
                    difficulty=info['size'],
                    source='synthetic'
                )
            )
    
    def register_dataset(self, name: str, info: BenchmarkDatasetInfo):
        """Register a new dataset."""
        self._metadata[name] = info
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets."""
        return list(self._metadata.keys())
    
    def get_dataset_info(self, name: str) -> BenchmarkDatasetInfo:
        """Get dataset information."""
        if name not in self._metadata:
            raise ValueError(f"Dataset {name} not found. Available: {self.list_datasets()}")
        return self._metadata[name]
    
    def get_datasets_by_domain(self, domain: str) -> List[str]:
        """Get datasets filtered by domain."""
        return [
            name for name, info in self._metadata.items()
            if info.domain == domain
        ]
    
    def get_datasets_by_difficulty(self, difficulty: str) -> List[str]:
        """Get datasets filtered by difficulty."""
        return [
            name for name, info in self._metadata.items()
            if info.difficulty == difficulty
        ]


# Global dataset registry
DATASET_REGISTRY = DatasetRegistry()


def load_benchmark_dataset(
    name: str,
    split: str = 'train',
    data_dir: Optional[str] = None
) -> Tuple[List[Example], np.ndarray]:
    """
    Load a benchmark dataset.
    
    Args:
        name: Dataset name
        split: Data split ('train', 'val', 'test')
        data_dir: Optional data directory
        
    Returns:
        Tuple of (examples, labels)
    """
    if name not in DATASET_REGISTRY.list_datasets():
        raise ValueError(f"Dataset {name} not found. Available: {DATASET_REGISTRY.list_datasets()}")
    
    # For now, use the StandardDatasets approach
    benchmark = StandardDatasets.create_benchmark(name)
    examples, labels = benchmark.load_data()
    
    # Simple split logic (in practice, this would load actual splits)
    n_examples = len(examples)
    if split == 'train':
        # Use first 70% for training
        end_idx = int(0.7 * n_examples)
        return examples[:end_idx], labels[:end_idx]
    elif split == 'val':
        # Use next 15% for validation
        start_idx = int(0.7 * n_examples)
        end_idx = int(0.85 * n_examples)
        return examples[start_idx:end_idx], labels[start_idx:end_idx]
    elif split == 'test':
        # Use last 15% for testing
        start_idx = int(0.85 * n_examples)
        return examples[start_idx:], labels[start_idx:]
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")


def create_synthetic_dataset(
    n_examples: int = 1000,
    n_classes: int = 2,
    domain: str = 'text',
    difficulty: str = 'medium',
    noise_level: float = 0.1
) -> Tuple[List[Example], np.ndarray]:
    """
    Create a synthetic dataset for testing.
    
    Args:
        n_examples: Number of examples to generate
        n_classes: Number of classes
        domain: Domain type ('text', 'medical', 'finance', etc.)
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        noise_level: Amount of noise to add (0.0 to 1.0)
        
    Returns:
        Tuple of (examples, labels)
    """
    # Generate synthetic examples
    examples = []
    for i in range(n_examples):
        # Generate domain-specific content
        if domain == 'text':
            content = f"This is example {i} with {domain} content. "
            content += "Some meaningful text here. " * np.random.randint(1, 5)
        elif domain == 'medical':
            symptoms = ['fever', 'cough', 'fatigue', 'headache', 'nausea']
            content = f"Patient {i}: " + ", ".join(np.random.choice(symptoms, np.random.randint(1, 4)))
        elif domain == 'finance':
            content = f"Transaction {i}: Amount ${np.random.uniform(10, 1000):.2f}, "
            content += f"Type: {'credit' if np.random.random() > 0.5 else 'debit'}"
        else:
            content = f"Generic example {i} for domain {domain}"
        
        example = Example(
            id=f"synthetic_{domain}_{i}",
            data=content,
            metadata={
                'domain': domain,
                'difficulty': difficulty,
                'synthetic': True
            }
        )
        examples.append(example)
    
    # Generate labels based on difficulty
    if difficulty == 'easy':
        # Clear patterns
        labels = np.array([i % n_classes for i in range(n_examples)])
    elif difficulty == 'medium':
        # Some randomness
        labels = np.random.randint(0, n_classes, n_examples)
        # But add some pattern
        for i in range(0, n_examples, n_classes * 2):
            end_idx = min(i + n_classes, n_examples)
            labels[i:end_idx] = np.arange(end_idx - i) % n_classes
    else:  # hard
        # Mostly random
        labels = np.random.randint(0, n_classes, n_examples)
    
    # Add noise
    if noise_level > 0:
        n_noisy = int(noise_level * n_examples)
        noisy_indices = np.random.choice(n_examples, n_noisy, replace=False)
        for idx in noisy_indices:
            # Flip to random label
            possible_labels = [l for l in range(n_classes) if l != labels[idx]]
            if possible_labels:
                labels[idx] = np.random.choice(possible_labels)
    
    return examples, labels


def get_dataset_statistics(examples: List[Example], labels: np.ndarray) -> Dict[str, Any]:
    """
    Get statistics about a dataset.
    
    Args:
        examples: List of examples
        labels: Array of labels
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'n_examples': len(examples),
        'n_classes': len(np.unique(labels)),
        'class_distribution': {},
        'avg_text_length': 0,
        'domains': set(),
    }
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        stats['class_distribution'][int(label)] = int(count)
    
    # Text statistics
    text_lengths = []
    domains = set()
    
    for example in examples:
        if isinstance(example.data, str):
            text_lengths.append(len(example.data))
        
        if example.metadata and 'domain' in example.metadata:
            domains.add(example.metadata['domain'])
    
    if text_lengths:
        stats['avg_text_length'] = np.mean(text_lengths)
        stats['min_text_length'] = np.min(text_lengths)
        stats['max_text_length'] = np.max(text_lengths)
    
    stats['domains'] = list(domains)
    
    return stats
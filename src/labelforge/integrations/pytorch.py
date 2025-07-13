"""
PyTorch integration for LabelForge.

This module provides utilities for exporting LabelForge models and data
to PyTorch format for downstream training and deployment.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch not available. PyTorch integration disabled.")

from ..types import Example, LFOutput
from ..label_model import LabelModel


class LabelForgeDataset(Dataset):
    """
    PyTorch Dataset wrapper for LabelForge data.
    
    Provides a PyTorch-compatible dataset interface for examples,
    labels, and probabilities from weak supervision.
    """
    
    def __init__(
        self,
        examples: List[Example],
        labels: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        include_text: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize PyTorch dataset.
        
        Args:
            examples: List of Example objects
            labels: Hard labels (optional)
            probabilities: Soft labels/probabilities (optional)
            include_text: Whether to include text in the dataset
            transform: Optional transform function for examples
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch is required for PyTorchExporter")
        
        self.examples = examples
        self.labels = labels
        self.probabilities = probabilities
        self.include_text = include_text
        self.transform = transform
        
        # Validate inputs
        if labels is not None:
            assert len(labels) == len(examples), "Labels and examples must have same length"
        
        if probabilities is not None:
            assert len(probabilities) == len(examples), "Probabilities and examples must have same length"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        item = {}
        
        if self.include_text:
            item['text'] = self.examples[idx].text
        
        # Add example metadata if available
        if hasattr(self.examples[idx], 'metadata') and self.examples[idx].metadata:
            item['metadata'] = self.examples[idx].metadata
        
        # Add labels if available
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Add probabilities if available
        if self.probabilities is not None:
            item['probabilities'] = torch.tensor(self.probabilities[idx], dtype=torch.float32)
        
        # Apply transform if provided
        if self.transform:
            item = self.transform(item)
        
        return item


class PyTorchExporter:
    """
    Export LabelForge models and data to PyTorch format.
    
    Provides utilities for converting weak supervision outputs to
    PyTorch datasets and models for downstream training.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize PyTorch exporter.
        
        Args:
            label_model: Trained LabelModel instance
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch is required for PyTorchExporter")
        
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'mu')
    
    def to_pytorch_dataset(
        self,
        examples: List[Example],
        lf_output: Optional[LFOutput] = None,
        use_probabilities: bool = True,
        include_text: bool = True,
        transform: Optional[callable] = None
    ) -> LabelForgeDataset:
        """
        Convert examples to PyTorch dataset.
        
        Args:
            examples: List of Example objects
            lf_output: LFOutput for generating labels/probabilities
            use_probabilities: Whether to include soft labels
            include_text: Whether to include text in dataset
            transform: Optional transform function
            
        Returns:
            PyTorch dataset with examples and labels
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before export")
        
        labels = None
        probabilities = None
        
        if lf_output is not None:
            # Generate predictions
            labels = self.label_model.predict(lf_output)
            
            if use_probabilities:
                probabilities = self.label_model.predict_proba(lf_output)
        
        return LabelForgeDataset(
            examples=examples,
            labels=labels,
            probabilities=probabilities,
            include_text=include_text,
            transform=transform
        )
    
    def to_dataloader(
        self,
        examples: List[Example],
        lf_output: Optional[LFOutput] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from examples.
        
        Args:
            examples: List of Example objects
            lf_output: LFOutput for generating labels
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            PyTorch DataLoader
        """
        dataset = self.to_pytorch_dataset(examples, lf_output)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def export_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Export label model parameters as PyTorch tensors.
        
        Returns:
            Dictionary with model parameters as tensors
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before export")
        
        weights = {}
        
        if hasattr(self.label_model, 'mu'):
            weights['mu'] = torch.tensor(self.label_model.mu, dtype=torch.float32)
        
        if hasattr(self.label_model, 'balance'):
            weights['balance'] = torch.tensor(self.label_model.balance, dtype=torch.float32)
        
        return weights
    
    def create_pytorch_classifier(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1
    ) -> torch.nn.Module:
        """
        Create a PyTorch classifier initialized with weak supervision.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            
        Returns:
            PyTorch classifier module
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before creating classifier")
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_dims[-1], self.label_model.cardinality))
        
        return torch.nn.Sequential(*layers)
    
    def export_training_data(
        self,
        examples: List[Example],
        lf_output: LFOutput,
        confidence_threshold: float = 0.8,
        export_format: str = "tensors"
    ) -> Dict[str, Any]:
        """
        Export high-confidence examples for training downstream models.
        
        Args:
            examples: List of Example objects
            lf_output: LFOutput with labeling function votes
            confidence_threshold: Minimum confidence for inclusion
            export_format: Format for export ("tensors", "numpy", "dict")
            
        Returns:
            Dictionary with filtered examples and labels
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before export")
        
        # Get predictions and probabilities
        predictions = self.label_model.predict(lf_output)
        probabilities = self.label_model.predict_proba(lf_output)
        
        # Filter by confidence
        max_probs = np.max(probabilities, axis=1)
        high_conf_mask = max_probs >= confidence_threshold
        
        filtered_examples = [examples[i] for i in range(len(examples)) if high_conf_mask[i]]
        filtered_labels = predictions[high_conf_mask]
        filtered_probs = probabilities[high_conf_mask]
        
        export_data = {
            'examples': filtered_examples,
            'n_examples': len(filtered_examples),
            'confidence_threshold': confidence_threshold,
            'coverage_rate': np.mean(high_conf_mask)
        }
        
        if export_format == "tensors":
            export_data['labels'] = torch.tensor(filtered_labels, dtype=torch.long)
            export_data['probabilities'] = torch.tensor(filtered_probs, dtype=torch.float32)
        elif export_format == "numpy":
            export_data['labels'] = filtered_labels
            export_data['probabilities'] = filtered_probs
        else:  # dict format
            export_data['labels'] = filtered_labels.tolist()
            export_data['probabilities'] = filtered_probs.tolist()
        
        return export_data


def collate_labelforge_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for LabelForge datasets.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched data dictionary
    """
    if not HAS_PYTORCH:
        raise ImportError("PyTorch is required for batch collation")
    
    batched = {}
    
    # Handle text data
    if 'text' in batch[0]:
        batched['text'] = [item['text'] for item in batch]
    
    # Handle tensor data
    for key in ['label', 'probabilities']:
        if key in batch[0]:
            batched[key] = torch.stack([item[key] for item in batch])
    
    # Handle metadata
    if 'metadata' in batch[0]:
        batched['metadata'] = [item['metadata'] for item in batch]
    
    return batched

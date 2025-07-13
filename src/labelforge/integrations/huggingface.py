"""
Hugging Face integration for LabelForge.

This module provides utilities for exporting LabelForge data to
Hugging Face datasets and models for NLP tasks.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import warnings

try:
    from datasets import Dataset, DatasetDict
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    warnings.warn("Hugging Face datasets not available. HuggingFace integration disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Hugging Face transformers not available. Model functionality disabled.")

from ..types import Example, LFOutput
from ..label_model import LabelModel


class HuggingFaceExporter:
    """
    Export LabelForge data to Hugging Face format.
    
    Provides utilities for converting weak supervision outputs to
    Hugging Face datasets and preparing them for transformer training.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize Hugging Face exporter.
        
        Args:
            label_model: Trained LabelModel instance
        """
        if not HAS_DATASETS:
            raise ImportError("Hugging Face datasets is required for HuggingFaceExporter")
        
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'mu')
    
    def to_hf_dataset(
        self,
        examples: List[Example],
        lf_output: Optional[LFOutput] = None,
        include_probabilities: bool = True,
        confidence_threshold: Optional[float] = None,
        split_ratios: Optional[Dict[str, float]] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Convert examples to Hugging Face dataset.
        
        Args:
            examples: List of Example objects
            lf_output: LFOutput for generating labels
            include_probabilities: Whether to include soft labels
            confidence_threshold: Filter examples by confidence (optional)
            split_ratios: Dictionary with train/val/test split ratios
            
        Returns:
            Hugging Face Dataset or DatasetDict if splits provided
        """
        if not self.is_fitted and lf_output is not None:
            raise ValueError("Label model must be fitted before export")
        
        # Prepare data dictionary
        data_dict = {
            'text': [example.text for example in examples],
            'id': list(range(len(examples)))
        }
        
        # Add metadata if available
        if examples and hasattr(examples[0], 'metadata') and examples[0].metadata:
            metadata_keys = set()
            for example in examples:
                if example.metadata:
                    metadata_keys.update(example.metadata.keys())
            
            for key in metadata_keys:
                data_dict[key] = [
                    example.metadata.get(key) if example.metadata else None 
                    for example in examples
                ]
        
        # Generate labels and probabilities if LF output provided
        if lf_output is not None:
            labels = self.label_model.predict(lf_output)
            probabilities = self.label_model.predict_proba(lf_output)
            
            # Apply confidence filtering if specified
            if confidence_threshold is not None:
                max_probs = np.max(probabilities, axis=1)
                high_conf_mask = max_probs >= confidence_threshold
                
                # Filter all data
                for key in data_dict:
                    if isinstance(data_dict[key], list):
                        data_dict[key] = [data_dict[key][i] for i in range(len(data_dict[key])) if high_conf_mask[i]]
                
                labels = labels[high_conf_mask]
                probabilities = probabilities[high_conf_mask]
            
            data_dict['label'] = labels.tolist()
            
            if include_probabilities:
                # Add class probabilities
                for i in range(probabilities.shape[1]):
                    data_dict[f'prob_class_{i}'] = probabilities[:, i].tolist()
                
                # Add max probability as confidence score
                data_dict['confidence'] = np.max(probabilities, axis=1).tolist()
        
        # Create dataset
        dataset = Dataset.from_dict(data_dict)
        
        # Create splits if requested
        if split_ratios is not None:
            return self._create_splits(dataset, split_ratios)
        
        return dataset
    
    def _create_splits(self, dataset: Dataset, split_ratios: Dict[str, float]) -> DatasetDict:
        """Create train/validation/test splits."""
        # Validate split ratios
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        dataset_dict = {}
        remaining_dataset = dataset
        
        # Sort splits to ensure consistent ordering
        sorted_splits = sorted(split_ratios.items())
        
        for i, (split_name, ratio) in enumerate(sorted_splits):
            if i == len(sorted_splits) - 1:
                # Last split gets all remaining data
                dataset_dict[split_name] = remaining_dataset
            else:
                # Calculate split size
                split_size = int(len(remaining_dataset) * ratio / sum(r for _, r in sorted_splits[i:]))
                
                # Create split
                split_dataset = remaining_dataset.select(range(split_size))
                dataset_dict[split_name] = split_dataset
                
                # Update remaining dataset
                remaining_indices = list(range(split_size, len(remaining_dataset)))
                remaining_dataset = remaining_dataset.select(remaining_indices)
        
        return DatasetDict(dataset_dict)
    
    def prepare_for_training(
        self,
        dataset: Dataset,
        model_name: str = "bert-base-uncased",
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 512,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """
        Prepare dataset for transformer training.
        
        Args:
            dataset: Hugging Face dataset
            model_name: Pre-trained model name
            text_column: Name of text column
            label_column: Name of label column
            max_length: Maximum sequence length
            tokenizer_kwargs: Additional tokenizer arguments
            
        Returns:
            Tokenized dataset ready for training
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("Hugging Face transformers required for training preparation")
        
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            """Tokenize text examples."""
            return tokenizer(
                examples[text_column],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors=None  # Return lists, not tensors
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column] if text_column in dataset.column_names else []
        )
        
        # Rename label column if necessary
        if label_column in tokenized_dataset.column_names and label_column != "labels":
            tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        
        return tokenized_dataset
    
    def create_model_for_classification(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a Hugging Face model for sequence classification.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of classification labels
            model_kwargs: Additional model arguments
            
        Returns:
            Hugging Face model for sequence classification
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("Hugging Face transformers required for model creation")
        
        if model_kwargs is None:
            model_kwargs = {}
        
        if num_labels is None:
            num_labels = self.label_model.cardinality
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            **model_kwargs
        )
        
        return model
    
    def export_training_config(
        self,
        output_dir: str = "./results",
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate training configuration for Hugging Face Trainer.
        
        Args:
            output_dir: Output directory for model and logs
            learning_rate: Learning rate for training
            batch_size: Training batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            **kwargs: Additional training arguments
            
        Returns:
            Training configuration dictionary
        """
        config = {
            "output_dir": output_dir,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "logging_dir": f"{output_dir}/logs",
            "logging_steps": 100,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_accuracy",
            "greater_is_better": True,
        }
        
        # Add any additional arguments
        config.update(kwargs)
        
        return config
    
    def export_for_inference(
        self,
        examples: List[Example],
        lf_output: LFOutput,
        confidence_threshold: float = 0.9,
        format: str = "jsonl"
    ) -> List[Dict[str, Any]]:
        """
        Export high-confidence predictions for inference or evaluation.
        
        Args:
            examples: List of Example objects
            lf_output: LFOutput with labeling function votes
            confidence_threshold: Minimum confidence for inclusion
            format: Output format ("jsonl", "csv", "dict")
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before export")
        
        # Get predictions and probabilities
        predictions = self.label_model.predict(lf_output)
        probabilities = self.label_model.predict_proba(lf_output)
        
        # Filter by confidence
        max_probs = np.max(probabilities, axis=1)
        high_conf_mask = max_probs >= confidence_threshold
        
        export_data = []
        
        for i, (example, pred, prob) in enumerate(zip(examples, predictions, probabilities)):
            if high_conf_mask[i]:
                item = {
                    "id": i,
                    "text": example.text,
                    "predicted_label": int(pred),
                    "confidence": float(max_probs[i]),
                    "probabilities": prob.tolist()
                }
                
                # Add metadata if available
                if hasattr(example, 'metadata') and example.metadata:
                    item["metadata"] = example.metadata
                
                export_data.append(item)
        
        return export_data

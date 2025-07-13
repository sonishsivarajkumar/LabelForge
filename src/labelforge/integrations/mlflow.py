"""
MLflow integration for LabelForge.

This module provides experiment tracking capabilities using MLflow,
enabling automatic logging of weak supervision experiments and model metrics.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
import warnings
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    warnings.warn("MLflow not available. Experiment tracking disabled.")

from ..types import Example, LFOutput
from ..label_model import LabelModel


class MLflowTracker:
    """
    MLflow experiment tracking for weak supervision experiments.
    
    Provides automatic logging of experiments, model parameters,
    metrics, and artifacts for reproducible research.
    """
    
    def __init__(
        self,
        experiment_name: str = "labelforge_experiments",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location for storing artifacts
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow is required for experiment tracking")
        
        self.experiment_name = experiment_name
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                self.experiment_id = experiment_id
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            warnings.warn(f"Failed to create/get MLflow experiment: {e}")
            self.experiment_id = None
        
        self.active_run = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> Any:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            nested: Whether this is a nested run
            
        Returns:
            MLflow run object
        """
        if self.experiment_id:
            mlflow.set_experiment(experiment_id=self.experiment_id)
        
        self.active_run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        
        return self.active_run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_experiment_config(
        self,
        examples: List[Example],
        lf_names: List[str],
        model_params: Dict[str, Any],
        additional_config: Optional[Dict[str, Any]] = None
    ):
        """
        Log experiment configuration.
        
        Args:
            examples: List of training examples
            lf_names: Names of labeling functions
            model_params: Label model parameters
            additional_config: Additional configuration to log
        """
        # Dataset statistics
        mlflow.log_param("n_examples", len(examples))
        mlflow.log_param("n_labeling_functions", len(lf_names))
        mlflow.log_param("lf_names", json.dumps(lf_names))
        
        # Model parameters
        for key, value in model_params.items():
            mlflow.log_param(f"model_{key}", value)
        
        # Additional configuration
        if additional_config:
            for key, value in additional_config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, json.dumps(str(value)))
    
    def log_lf_statistics(self, lf_output: LFOutput, examples: List[Example]):
        """
        Log labeling function statistics.
        
        Args:
            lf_output: LFOutput with labeling function votes
            examples: List of examples
        """
        votes = lf_output.votes
        n_examples, n_lfs = votes.shape
        
        # Coverage statistics
        lf_coverage = np.mean(votes != -1, axis=0)
        total_coverage = np.mean(np.any(votes != -1, axis=1))
        
        mlflow.log_metric("total_coverage", total_coverage)
        mlflow.log_metric("mean_lf_coverage", np.mean(lf_coverage))
        mlflow.log_metric("std_lf_coverage", np.std(lf_coverage))
        mlflow.log_metric("min_lf_coverage", np.min(lf_coverage))
        mlflow.log_metric("max_lf_coverage", np.max(lf_coverage))
        
        # Agreement statistics
        agreements = []
        for i in range(n_lfs):
            for j in range(i + 1, n_lfs):
                both_vote = (votes[:, i] != -1) & (votes[:, j] != -1)
                if np.sum(both_vote) > 0:
                    agreement = np.mean(votes[both_vote, i] == votes[both_vote, j])
                    agreements.append(agreement)
        
        if agreements:
            mlflow.log_metric("mean_pairwise_agreement", np.mean(agreements))
            mlflow.log_metric("std_pairwise_agreement", np.std(agreements))
        
        # Conflict statistics
        conflict_count = 0
        for i in range(n_examples):
            example_votes = votes[i][votes[i] != -1]
            if len(example_votes) > 1 and len(np.unique(example_votes)) > 1:
                conflict_count += 1
        
        conflict_rate = conflict_count / n_examples
        mlflow.log_metric("conflict_rate", conflict_rate)
        mlflow.log_metric("n_conflict_examples", conflict_count)
        
        # Log individual LF coverage
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        for i, (name, coverage) in enumerate(zip(lf_names, lf_coverage)):
            mlflow.log_metric(f"lf_coverage_{name}", coverage)
    
    def log_model_training(
        self,
        label_model: LabelModel,
        lf_output: LFOutput,
        training_time: Optional[float] = None,
        convergence_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log model training information.
        
        Args:
            label_model: Trained label model
            lf_output: LFOutput used for training
            training_time: Training time in seconds
            convergence_info: Convergence information
        """
        # Model parameters
        if hasattr(label_model, 'cardinality'):
            mlflow.log_param("model_cardinality", label_model.cardinality)
        
        # Training time
        if training_time is not None:
            mlflow.log_metric("training_time_seconds", training_time)
        
        # Convergence information
        if convergence_info:
            for key, value in convergence_info.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"convergence_{key}", value)
        
        # Log model as artifact
        try:
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(label_model, f)
                mlflow.log_artifact(f.name, "model")
                Path(f.name).unlink()  # Clean up temp file
        except Exception as e:
            warnings.warn(f"Failed to log model artifact: {e}")
    
    def log_predictions(
        self,
        label_model: LabelModel,
        lf_output: LFOutput,
        examples: List[Example],
        true_labels: Optional[np.ndarray] = None
    ):
        """
        Log model predictions and evaluation metrics.
        
        Args:
            label_model: Trained label model
            lf_output: LFOutput for predictions
            examples: List of examples
            true_labels: True labels for supervised evaluation
        """
        # Generate predictions
        predictions = label_model.predict(lf_output)
        probabilities = label_model.predict_proba(lf_output)
        
        # Prediction statistics
        max_probs = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        mlflow.log_metric("mean_prediction_confidence", np.mean(max_probs))
        mlflow.log_metric("std_prediction_confidence", np.std(max_probs))
        mlflow.log_metric("mean_prediction_entropy", np.mean(entropy))
        mlflow.log_metric("std_prediction_entropy", np.std(entropy))
        
        # High/low confidence examples
        high_conf_threshold = 0.8
        low_conf_threshold = 0.6
        
        high_conf_count = np.sum(max_probs > high_conf_threshold)
        low_conf_count = np.sum(max_probs < low_conf_threshold)
        
        mlflow.log_metric("high_confidence_examples", high_conf_count)
        mlflow.log_metric("low_confidence_examples", low_conf_count)
        mlflow.log_metric("high_confidence_rate", high_conf_count / len(predictions))
        mlflow.log_metric("low_confidence_rate", low_conf_count / len(predictions))
        
        # Class distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_preds, pred_counts):
            mlflow.log_metric(f"predicted_class_{pred}_count", count)
            mlflow.log_metric(f"predicted_class_{pred}_rate", count / len(predictions))
        
        # Supervised evaluation if true labels available
        if true_labels is not None:
            self._log_supervised_metrics(predictions, probabilities, true_labels)
    
    def _log_supervised_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ):
        """Log supervised evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, confusion_matrix
        )
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        mlflow.log_metric("accuracy", accuracy)
        
        # Handle binary vs multi-class
        if len(np.unique(true_labels)) == 2:
            f1 = f1_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
        else:
            f1 = f1_score(true_labels, predictions, average='weighted')
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Per-class metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"class_{label}_{metric_name}", value)
        
        # Log confusion matrix as artifact
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import tempfile
            
            cm = confusion_matrix(true_labels, predictions)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                plt.savefig(f.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(f.name, "plots")
                Path(f.name).unlink()
            
            plt.close()
            
        except Exception as e:
            warnings.warn(f"Failed to log confusion matrix: {e}")
    
    def log_artifacts(self, artifacts: Dict[str, Union[str, Path]]):
        """
        Log artifacts to MLflow.
        
        Args:
            artifacts: Dictionary mapping artifact names to file paths
        """
        for name, path in artifacts.items():
            try:
                mlflow.log_artifact(str(path), name)
            except Exception as e:
                warnings.warn(f"Failed to log artifact {name}: {e}")
    
    def create_experiment_summary(
        self,
        experiment_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a summary of the experiment results.
        
        Args:
            experiment_results: Dictionary with experiment results
            save_path: Optional path to save summary
            
        Returns:
            Experiment summary as string
        """
        summary_lines = [
            "# LabelForge Experiment Summary",
            f"**Experiment:** {self.experiment_name}",
            f"**Run ID:** {self.active_run.info.run_id if self.active_run else 'N/A'}",
            "",
            "## Configuration",
        ]
        
        # Add configuration details
        for key, value in experiment_results.get('config', {}).items():
            summary_lines.append(f"- **{key}:** {value}")
        
        summary_lines.extend([
            "",
            "## Results",
        ])
        
        # Add results
        for key, value in experiment_results.get('results', {}).items():
            if isinstance(value, float):
                summary_lines.append(f"- **{key}:** {value:.4f}")
            else:
                summary_lines.append(f"- **{key}:** {value}")
        
        summary = "\\n".join(summary_lines)
        
        # Save to file if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
            mlflow.log_artifact(save_path, "summaries")
        
        return summary

"""
Weights & Biases integration for LabelForge.

This module provides experiment tracking and visualization capabilities
using Weights & Biases for weak supervision experiments.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
import warnings

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("Weights & Biases not available. WandB integration disabled.")

from ..types import Example, LFOutput
from ..label_model import LabelModel


class WandBTracker:
    """
    Weights & Biases experiment tracking for weak supervision.
    
    Provides real-time logging, visualization, and comparison
    of weak supervision experiments.
    """
    
    def __init__(
        self,
        project_name: str = "labelforge",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize WandB tracker.
        
        Args:
            project_name: Name of the WandB project
            entity: WandB entity (username or team)
            config: Initial configuration dictionary
            name: Run name
            tags: List of tags for the run
        """
        if not HAS_WANDB:
            raise ImportError("Weights & Biases is required for WandBTracker")
        
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
        # Initialize run
        self.start_run(config=config, name=name, tags=tags)
    
    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None
    ):
        """
        Start a new WandB run.
        
        Args:
            config: Configuration dictionary
            name: Run name
            tags: List of tags
            group: Run group for organizing related runs
            job_type: Job type (e.g., "train", "eval")
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=config or {},
            name=name,
            tags=tags,
            group=group,
            job_type=job_type,
            reinit=True
        )
    
    def finish_run(self):
        """Finish the current WandB run."""
        if self.run:
            wandb.finish()
            self.run = None
    
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
            additional_config: Additional configuration
        """
        config_update = {
            "dataset": {
                "n_examples": len(examples),
                "n_labeling_functions": len(lf_names),
                "lf_names": lf_names,
            },
            "model": model_params,
        }
        
        if additional_config:
            config_update.update(additional_config)
        
        wandb.config.update(config_update)
    
    def log_lf_statistics(self, lf_output: LFOutput, examples: List[Example]):
        """
        Log labeling function statistics.
        
        Args:
            lf_output: LFOutput with labeling function votes
            examples: List of examples
        """
        votes = lf_output.votes
        n_examples, n_lfs = votes.shape
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        # Coverage statistics
        lf_coverage = np.mean(votes != -1, axis=0)
        total_coverage = np.mean(np.any(votes != -1, axis=1))
        
        # Log overall statistics
        wandb.log({
            "lf_stats/total_coverage": total_coverage,
            "lf_stats/mean_lf_coverage": np.mean(lf_coverage),
            "lf_stats/std_lf_coverage": np.std(lf_coverage),
            "lf_stats/min_lf_coverage": np.min(lf_coverage),
            "lf_stats/max_lf_coverage": np.max(lf_coverage),
        })
        
        # Log individual LF coverage
        for name, coverage in zip(lf_names, lf_coverage):
            wandb.log({f"lf_coverage/{name}": coverage})
        
        # Agreement statistics
        agreements = []
        for i in range(n_lfs):
            for j in range(i + 1, n_lfs):
                both_vote = (votes[:, i] != -1) & (votes[:, j] != -1)
                if np.sum(both_vote) > 0:
                    agreement = np.mean(votes[both_vote, i] == votes[both_vote, j])
                    agreements.append(agreement)
                    
                    # Log pairwise agreement
                    wandb.log({
                        f"lf_agreement/{lf_names[i]}_vs_{lf_names[j]}": agreement
                    })
        
        if agreements:
            wandb.log({
                "lf_stats/mean_pairwise_agreement": np.mean(agreements),
                "lf_stats/std_pairwise_agreement": np.std(agreements),
            })
        
        # Conflict statistics
        conflict_count = 0
        conflict_examples = []
        
        for i in range(n_examples):
            example_votes = votes[i][votes[i] != -1]
            if len(example_votes) > 1 and len(np.unique(example_votes)) > 1:
                conflict_count += 1
                if len(conflict_examples) < 10:  # Store first 10 conflicts
                    conflict_examples.append({
                        "text": examples[i].text[:200],  # Truncate for display
                        "votes": example_votes.tolist(),
                        "n_conflicts": len(np.unique(example_votes)) - 1
                    })
        
        conflict_rate = conflict_count / n_examples
        
        wandb.log({
            "lf_stats/conflict_rate": conflict_rate,
            "lf_stats/n_conflict_examples": conflict_count,
        })
        
        # Log conflict examples as a table
        if conflict_examples:
            conflict_table = wandb.Table(
                columns=["text", "votes", "n_conflicts"],
                data=[[ex["text"], str(ex["votes"]), ex["n_conflicts"]] 
                      for ex in conflict_examples]
            )
            wandb.log({"lf_conflicts": conflict_table})
    
    def log_training_progress(
        self,
        iteration: int,
        loss: float,
        parameter_change: float,
        convergence_criterion: float
    ):
        """
        Log training progress metrics.
        
        Args:
            iteration: Current iteration
            loss: Current loss value
            parameter_change: Parameter change from previous iteration
            convergence_criterion: Convergence criterion value
        """
        wandb.log({
            "training/iteration": iteration,
            "training/loss": loss,
            "training/parameter_change": parameter_change,
            "training/convergence_criterion": convergence_criterion,
        })
    
    def log_model_predictions(
        self,
        label_model: LabelModel,
        lf_output: LFOutput,
        examples: List[Example],
        true_labels: Optional[np.ndarray] = None,
        step: Optional[int] = None
    ):
        """
        Log model predictions and evaluation metrics.
        
        Args:
            label_model: Trained label model
            lf_output: LFOutput for predictions
            examples: List of examples
            true_labels: True labels for supervised evaluation
            step: Optional step for logging
        """
        # Generate predictions
        predictions = label_model.predict(lf_output)
        probabilities = label_model.predict_proba(lf_output)
        
        # Prediction statistics
        max_probs = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        log_dict = {
            "predictions/mean_confidence": np.mean(max_probs),
            "predictions/std_confidence": np.std(max_probs),
            "predictions/mean_entropy": np.mean(entropy),
            "predictions/std_entropy": np.std(entropy),
        }
        
        # High/low confidence statistics
        high_conf_threshold = 0.8
        low_conf_threshold = 0.6
        
        high_conf_count = np.sum(max_probs > high_conf_threshold)
        low_conf_count = np.sum(max_probs < low_conf_threshold)
        
        log_dict.update({
            "predictions/high_confidence_examples": high_conf_count,
            "predictions/low_confidence_examples": low_conf_count,
            "predictions/high_confidence_rate": high_conf_count / len(predictions),
            "predictions/low_confidence_rate": low_conf_count / len(predictions),
        })
        
        # Class distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_preds, pred_counts):
            log_dict[f"predictions/class_{pred}_count"] = count
            log_dict[f"predictions/class_{pred}_rate"] = count / len(predictions)
        
        # Supervised evaluation if available
        if true_labels is not None:
            supervised_metrics = self._calculate_supervised_metrics(
                predictions, probabilities, true_labels
            )
            log_dict.update(supervised_metrics)
        
        wandb.log(log_dict, step=step)
        
        # Log prediction confidence histogram
        confidence_hist = wandb.Histogram(max_probs)
        wandb.log({"predictions/confidence_distribution": confidence_hist}, step=step)
        
        # Log entropy histogram
        entropy_hist = wandb.Histogram(entropy)
        wandb.log({"predictions/entropy_distribution": entropy_hist}, step=step)
    
    def _calculate_supervised_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate supervised evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix
        )
        
        metrics = {}
        
        # Basic metrics
        metrics["evaluation/accuracy"] = accuracy_score(true_labels, predictions)
        
        # Handle binary vs multi-class
        if len(np.unique(true_labels)) == 2:
            metrics["evaluation/f1_score"] = f1_score(true_labels, predictions)
            metrics["evaluation/precision"] = precision_score(true_labels, predictions)
            metrics["evaluation/recall"] = recall_score(true_labels, predictions)
        else:
            metrics["evaluation/f1_score"] = f1_score(true_labels, predictions, average='weighted')
            metrics["evaluation/precision"] = precision_score(true_labels, predictions, average='weighted')
            metrics["evaluation/recall"] = recall_score(true_labels, predictions, average='weighted')
        
        return metrics
    
    def log_lf_importance(self, importance_df, method: str = "permutation"):
        """
        Log labeling function importance analysis.
        
        Args:
            importance_df: DataFrame with LF importance scores
            method: Importance calculation method
        """
        # Log as table
        importance_table = wandb.Table(dataframe=importance_df)
        wandb.log({f"lf_importance/{method}": importance_table})
        
        # Log individual importance scores
        for _, row in importance_df.iterrows():
            wandb.log({
                f"lf_importance/{method}/{row['lf_name']}": row['importance_score']
            })
    
    def log_uncertainty_analysis(
        self,
        uncertainty_results: Dict[str, Any],
        method: str = "bootstrap"
    ):
        """
        Log uncertainty quantification results.
        
        Args:
            uncertainty_results: Results from uncertainty analysis
            method: Uncertainty estimation method
        """
        probabilities = uncertainty_results['probabilities']
        lower_bounds = uncertainty_results['lower_bounds']
        upper_bounds = uncertainty_results['upper_bounds']
        predictions = uncertainty_results['predictions']
        
        # Calculate uncertainty metrics
        max_probs = np.max(probabilities, axis=1)
        uncertainty_width = upper_bounds[np.arange(len(predictions)), predictions] - \
                           lower_bounds[np.arange(len(predictions)), predictions]
        
        wandb.log({
            f"uncertainty/{method}/mean_confidence": np.mean(max_probs),
            f"uncertainty/{method}/mean_width": np.mean(uncertainty_width),
            f"uncertainty/{method}/std_width": np.std(uncertainty_width),
            f"uncertainty/{method}/max_width": np.max(uncertainty_width),
        })
        
        # Log uncertainty distribution
        uncertainty_hist = wandb.Histogram(uncertainty_width)
        wandb.log({f"uncertainty/{method}/width_distribution": uncertainty_hist})
    
    def log_cross_validation_results(self, cv_results: Dict[str, Any]):
        """
        Log cross-validation results.
        
        Args:
            cv_results: Results from cross-validation
        """
        # Log aggregated metrics
        for key, value in cv_results.items():
            if key.endswith('_mean') or key.endswith('_std'):
                wandb.log({f"cross_validation/{key}": value})
        
        # Log fold-by-fold results if available
        if 'fold_results' in cv_results:
            fold_data = []
            for i, fold_result in enumerate(cv_results['fold_results']):
                fold_row = {"fold": i + 1}
                
                if 'weak_supervision_metrics' in fold_result:
                    ws_metrics = fold_result['weak_supervision_metrics']
                    fold_row.update({
                        "coverage": ws_metrics.get('total_coverage', 0),
                        "conflict_rate": ws_metrics.get('conflict_rate', 0),
                        "agreement": ws_metrics.get('avg_pairwise_agreement', 0),
                    })
                
                fold_data.append(fold_row)
            
            if fold_data:
                cv_table = wandb.Table(
                    columns=list(fold_data[0].keys()),
                    data=[list(row.values()) for row in fold_data]
                )
                wandb.log({"cross_validation/fold_results": cv_table})
    
    def create_summary_dashboard(self):
        """Create a summary dashboard with key metrics and visualizations."""
        # This would create custom WandB panels and charts
        # Implementation depends on specific dashboard requirements
        pass

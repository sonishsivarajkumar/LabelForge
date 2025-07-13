"""
Advanced evaluation framework for weak supervision models.

This module provides comprehensive evaluation tools including cross-validation,
advanced metrics, and comparison utilities for weak supervision models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib not available. Plotting functionality disabled.")

from ..types import LFOutput, Example
from ..label_model import LabelModel


class AdvancedEvaluator:
    """
    Advanced evaluation tools for weak supervision models.
    
    Provides comprehensive evaluation metrics beyond basic accuracy,
    including weak supervision specific metrics and analysis tools.
    """
    
    def __init__(self):
        """Initialize advanced evaluator."""
        pass
    
    def evaluate_comprehensive(
        self,
        label_model: LabelModel,
        lf_output: LFOutput,
        true_labels: Optional[np.ndarray] = None,
        examples: Optional[List[Example]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a label model.
        
        Args:
            label_model: Trained LabelModel instance
            lf_output: LFOutput containing labeling function votes
            true_labels: True labels for supervised evaluation (optional)
            examples: Original examples for additional analysis (optional)
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {}
        
        # Get model predictions
        predictions = label_model.predict(lf_output)
        probabilities = label_model.predict_proba(lf_output)
        
        # Basic model statistics
        results['model_stats'] = self._calculate_model_stats(
            predictions, probabilities, lf_output
        )
        
        # Weak supervision specific metrics
        results['weak_supervision_metrics'] = self._calculate_ws_metrics(
            lf_output, predictions, probabilities
        )
        
        # Supervised evaluation if true labels available
        if true_labels is not None:
            results['supervised_metrics'] = self._calculate_supervised_metrics(
                predictions, probabilities, true_labels
            )
        
        # Coverage and conflict analysis
        results['coverage_analysis'] = self._analyze_coverage_patterns(lf_output)
        
        return results
    
    def _calculate_model_stats(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        lf_output: LFOutput
    ) -> Dict[str, Any]:
        """Calculate basic model statistics."""
        n_examples = len(predictions)
        n_classes = probabilities.shape[1]
        
        # Prediction confidence
        max_probs = np.max(probabilities, axis=1)
        prediction_entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        # Class distribution
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        class_distribution = dict(zip(unique_preds.astype(int), pred_counts))
        
        return {
            'n_examples': n_examples,
            'n_classes': n_classes,
            'class_distribution': class_distribution,
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'mean_entropy': np.mean(prediction_entropy),
            'high_confidence_examples': np.sum(max_probs > 0.9),
            'low_confidence_examples': np.sum(max_probs < 0.6)
        }
    
    def _calculate_ws_metrics(
        self,
        lf_output: LFOutput,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate weak supervision specific metrics."""
        votes = lf_output.votes
        n_lfs = votes.shape[1]
        
        # Coverage metrics
        total_coverage = np.mean(np.any(votes != -1, axis=1))
        per_lf_coverage = np.mean(votes != -1, axis=0)
        
        # Conflict metrics
        conflicts_per_example = []
        for i in range(len(votes)):
            example_votes = votes[i][votes[i] != -1]
            if len(example_votes) > 1:
                conflicts_per_example.append(len(np.unique(example_votes)) - 1)
            else:
                conflicts_per_example.append(0)
        
        conflict_rate = np.mean(np.array(conflicts_per_example) > 0)
        avg_conflicts = np.mean(conflicts_per_example)
        
        # Agreement metrics
        pairwise_agreements = []
        for i in range(n_lfs):
            for j in range(i + 1, n_lfs):
                # Find examples where both LFs vote
                both_vote = (votes[:, i] != -1) & (votes[:, j] != -1)
                if np.sum(both_vote) > 0:
                    agreement = np.mean(votes[both_vote, i] == votes[both_vote, j])
                    pairwise_agreements.append(agreement)
        
        avg_agreement = np.mean(pairwise_agreements) if pairwise_agreements else 0
        
        return {
            'total_coverage': total_coverage,
            'mean_lf_coverage': np.mean(per_lf_coverage),
            'coverage_std': np.std(per_lf_coverage),
            'conflict_rate': conflict_rate,
            'avg_conflicts_per_example': avg_conflicts,
            'avg_pairwise_agreement': avg_agreement,
            'n_labeling_functions': n_lfs
        }
    
    def _calculate_supervised_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate supervised evaluation metrics."""
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle multi-class vs binary
        if len(np.unique(true_labels)) == 2:
            f1 = f1_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            
            # ROC AUC for binary classification
            if probabilities.shape[1] == 2:
                auc = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                auc = None
        else:
            f1 = f1_score(true_labels, predictions, average='weighted')
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            auc = None  # Multi-class AUC is more complex
        
        # Per-class metrics
        unique_labels = np.unique(true_labels)
        per_class_metrics = {}
        
        for label in unique_labels:
            label_mask = (true_labels == label)
            pred_mask = (predictions == label)
            
            if np.sum(label_mask) > 0:
                class_precision = np.sum(label_mask & pred_mask) / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0
                class_recall = np.sum(label_mask & pred_mask) / np.sum(label_mask)
                class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
                
                per_class_metrics[int(label)] = {
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1,
                    'support': np.sum(label_mask)
                }
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc_score': auc,
            'per_class_metrics': per_class_metrics
        }
    
    def _analyze_coverage_patterns(self, lf_output: LFOutput) -> Dict[str, Any]:
        """Analyze coverage patterns in the labeling functions."""
        votes = lf_output.votes
        n_examples, n_lfs = votes.shape
        
        # Coverage overlap analysis
        coverage_matrix = (votes != -1).astype(int)
        
        # Examples covered by k labeling functions
        coverage_counts = np.sum(coverage_matrix, axis=1)
        coverage_distribution = {}
        for k in range(n_lfs + 1):
            coverage_distribution[k] = np.sum(coverage_counts == k)
        
        # LF overlap patterns
        overlap_matrix = np.zeros((n_lfs, n_lfs))
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i != j:
                    overlap = np.sum(coverage_matrix[:, i] & coverage_matrix[:, j])
                    overlap_matrix[i, j] = overlap / n_examples
        
        return {
            'coverage_distribution': coverage_distribution,
            'uncovered_examples': coverage_distribution.get(0, 0),
            'fully_covered_examples': np.sum(coverage_counts == n_lfs),
            'mean_coverage_per_example': np.mean(coverage_counts),
            'overlap_matrix': overlap_matrix,
            'max_overlap': np.max(overlap_matrix),
            'min_overlap': np.min(overlap_matrix[overlap_matrix > 0]) if np.any(overlap_matrix > 0) else 0
        }
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, Any]],
        metric: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Compare multiple models based on evaluation results.
        
        Args:
            models_results: Dictionary mapping model names to evaluation results
            metric: Primary metric for comparison
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {'model_name': model_name}
            
            # Extract key metrics
            if 'supervised_metrics' in results:
                supervised = results['supervised_metrics']
                row.update({
                    'accuracy': supervised.get('accuracy', np.nan),
                    'f1_score': supervised.get('f1_score', np.nan),
                    'precision': supervised.get('precision', np.nan),
                    'recall': supervised.get('recall', np.nan)
                })
            
            if 'weak_supervision_metrics' in results:
                ws_metrics = results['weak_supervision_metrics']
                row.update({
                    'coverage': ws_metrics.get('total_coverage', np.nan),
                    'conflict_rate': ws_metrics.get('conflict_rate', np.nan),
                    'agreement': ws_metrics.get('avg_pairwise_agreement', np.nan)
                })
            
            if 'model_stats' in results:
                stats = results['model_stats']
                row.update({
                    'mean_confidence': stats.get('mean_confidence', np.nan),
                    'mean_entropy': stats.get('mean_entropy', np.nan)
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric if available
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df


class CrossValidator:
    """
    Cross-validation framework for weak supervision models.
    
    Provides tools for cross-validation with proper handling of weak supervision
    constraints and labeling function dependencies.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: Optional[int] = None):
        """
        Initialize cross-validator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def cross_validate_ws(
        self,
        lf_output: LFOutput,
        examples: List[Example],
        model_params: Optional[Dict[str, Any]] = None,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for weak supervision model.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            examples: List of examples
            model_params: Parameters for LabelModel initialization
            true_labels: True labels for evaluation (optional)
            
        Returns:
            Dictionary with cross-validation results
        """
        if model_params is None:
            model_params = {}
        
        n_examples = len(examples)
        votes = lf_output.votes
        
        # Create folds
        if true_labels is not None:
            # Stratified split if labels available
            kfold = StratifiedKFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
            fold_splits = list(kfold.split(votes, true_labels))
        else:
            # Regular k-fold split
            kfold = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            fold_splits = list(kfold.split(votes))
        
        fold_results = []
        evaluator = AdvancedEvaluator()
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            # Split data
            train_votes = votes[train_idx]
            val_votes = votes[val_idx]
            
            train_lf_output = LFOutput(
                votes=train_votes,
                lf_names=lf_output.lf_names
            )
            val_lf_output = LFOutput(
                votes=val_votes,
                lf_names=lf_output.lf_names
            )
            
            # Train model on fold
            model = LabelModel(**model_params)
            model.fit(train_lf_output, verbose=False)
            
            # Evaluate on validation set
            val_true_labels = true_labels[val_idx] if true_labels is not None else None
            val_examples = [examples[i] for i in val_idx]
            
            fold_result = evaluator.evaluate_comprehensive(
                model, val_lf_output, val_true_labels, val_examples
            )
            fold_result['fold_idx'] = fold_idx
            fold_result['train_size'] = len(train_idx)
            fold_result['val_size'] = len(val_idx)
            
            fold_results.append(fold_result)
        
        # Aggregate results across folds
        return self._aggregate_cv_results(fold_results)
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        aggregated = {
            'n_folds': len(fold_results),
            'fold_results': fold_results
        }
        
        # Aggregate supervised metrics if available
        if 'supervised_metrics' in fold_results[0]:
            supervised_metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            for metric in supervised_metrics:
                values = [
                    fold['supervised_metrics'].get(metric, np.nan) 
                    for fold in fold_results
                ]
                aggregated[f'{metric}_mean'] = np.nanmean(values)
                aggregated[f'{metric}_std'] = np.nanstd(values)
        
        # Aggregate weak supervision metrics
        if 'weak_supervision_metrics' in fold_results[0]:
            ws_metrics = ['total_coverage', 'conflict_rate', 'avg_pairwise_agreement']
            for metric in ws_metrics:
                values = [
                    fold['weak_supervision_metrics'].get(metric, np.nan)
                    for fold in fold_results
                ]
                aggregated[f'{metric}_mean'] = np.nanmean(values)
                aggregated[f'{metric}_std'] = np.nanstd(values)
        
        return aggregated
    
    def plot_cv_results(
        self, 
        cv_results: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot cross-validation results.
        
        Args:
            cv_results: Results from cross_validate_ws
            metrics: List of metrics to plot
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'total_coverage', 'conflict_rate']
        
        # Filter available metrics
        available_metrics = [
            metric for metric in metrics 
            if f'{metric}_mean' in cv_results
        ]
        
        if not available_metrics:
            print("No plottable metrics found in CV results")
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            mean_val = cv_results[f'{metric}_mean']
            std_val = cv_results[f'{metric}_std']
            
            # Extract individual fold values
            fold_values = [
                fold['supervised_metrics'].get(metric, np.nan) if 'supervised_metrics' in fold
                else fold['weak_supervision_metrics'].get(metric, np.nan)
                for fold in cv_results['fold_results']
            ]
            fold_values = [v for v in fold_values if not np.isnan(v)]
            
            # Box plot
            axes[i].boxplot(fold_values, labels=[metric])
            axes[i].set_title(f'{metric.replace("_", " ").title()}\\nMean: {mean_val:.3f} ± {std_val:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

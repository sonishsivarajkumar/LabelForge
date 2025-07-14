"""
Benchmark metrics and evaluation utilities.

This module provides comprehensive metrics for evaluating weak supervision
methods and comparing performance across different approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
import warnings

from ..types import Example


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    description: str
    higher_is_better: bool = True


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metric results."""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    matthews_corr: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None
    memory_usage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'matthews_corr': self.matthews_corr,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report,
            'memory_usage': self.memory_usage
        }


class WeakSupervisionMetrics:
    """
    Specialized metrics for weak supervision evaluation.
    
    Provides metrics that are particularly relevant for assessing
    the quality of weak supervision approaches.
    """
    
    @staticmethod
    def label_model_coverage(weak_labels: np.ndarray) -> float:
        """
        Calculate the coverage of labeling functions.
        
        Args:
            weak_labels: Weak labels matrix (n_examples x n_lfs)
            
        Returns:
            Coverage percentage (0-1)
        """
        n_examples, n_lfs = weak_labels.shape
        
        # Count examples with at least one non-abstain label
        covered_examples = np.sum(np.any(weak_labels != -1, axis=1))
        
        return covered_examples / n_examples
    
    @staticmethod
    def label_model_conflict(weak_labels: np.ndarray) -> float:
        """
        Calculate the conflict rate among labeling functions.
        
        Args:
            weak_labels: Weak labels matrix (n_examples x n_lfs)
            
        Returns:
            Conflict rate (0-1)
        """
        n_examples, n_lfs = weak_labels.shape
        conflicts = 0
        total_comparisons = 0
        
        for i in range(n_examples):
            # Get non-abstain labels for this example
            labels = weak_labels[i][weak_labels[i] != -1]
            
            if len(labels) > 1:
                # Check for conflicts
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    conflicts += 1
                total_comparisons += 1
        
        return conflicts / total_comparisons if total_comparisons > 0 else 0.0
    
    @staticmethod
    def labeling_function_accuracy(
        weak_labels: np.ndarray,
        true_labels: np.ndarray,
        lf_index: int
    ) -> float:
        """
        Calculate accuracy for a specific labeling function.
        
        Args:
            weak_labels: Weak labels matrix
            true_labels: True labels
            lf_index: Index of the labeling function
            
        Returns:
            Accuracy of the labeling function
        """
        lf_labels = weak_labels[:, lf_index]
        
        # Only consider examples where LF didn't abstain
        mask = lf_labels != -1
        
        if np.sum(mask) == 0:
            return 0.0
        
        return accuracy_score(true_labels[mask], lf_labels[mask])
    
    @staticmethod
    def labeling_function_coverage(weak_labels: np.ndarray, lf_index: int) -> float:
        """
        Calculate coverage for a specific labeling function.
        
        Args:
            weak_labels: Weak labels matrix
            lf_index: Index of the labeling function
            
        Returns:
            Coverage of the labeling function
        """
        lf_labels = weak_labels[:, lf_index]
        return np.sum(lf_labels != -1) / len(lf_labels)
    
    @staticmethod
    def agreement_matrix(weak_labels: np.ndarray) -> np.ndarray:
        """
        Calculate agreement matrix between labeling functions.
        
        Args:
            weak_labels: Weak labels matrix
            
        Returns:
            Agreement matrix (n_lfs x n_lfs)
        """
        n_examples, n_lfs = weak_labels.shape
        agreement = np.zeros((n_lfs, n_lfs))
        
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i == j:
                    agreement[i, j] = 1.0
                else:
                    # Find examples where both LFs don't abstain
                    mask = (weak_labels[:, i] != -1) & (weak_labels[:, j] != -1)
                    
                    if np.sum(mask) > 0:
                        # Calculate agreement on these examples
                        agreements = weak_labels[mask, i] == weak_labels[mask, j]
                        agreement[i, j] = np.mean(agreements)
                    else:
                        agreement[i, j] = 0.0
        
        return agreement


class MetricCalculator:
    """
    Calculate comprehensive metrics for model evaluation.
    """
    
    def __init__(self, average: str = 'weighted'):
        """
        Initialize metric calculator.
        
        Args:
            average: Averaging strategy for multi-class metrics
        """
        self.average = average
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> BenchmarkMetrics:
        """
        Calculate all standard metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            class_names: Class names (optional)
            
        Returns:
            BenchmarkMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=self.average, zero_division=0)
        precision = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        try:
            if class_names:
                report = classification_report(
                    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
                )
            else:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception:
            report = None
        
        # Probability-based metrics (if available)
        auc_roc = None
        auc_pr = None
        if y_prob is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    auc_roc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                    auc_pr = average_precision_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                else:
                    # Multi-class classification
                    auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=self.average)
            except Exception as e:
                warnings.warn(f"Could not calculate AUC metrics: {e}")
        
        # Matthews correlation coefficient
        matthews_corr = None
        try:
            matthews_corr = matthews_corrcoef(y_true, y_pred)
        except Exception:
            pass
        
        return BenchmarkMetrics(
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            matthews_corr=matthews_corr,
            confusion_matrix=cm,
            classification_report=report
        )
    
    def calculate_weak_supervision_metrics(
        self,
        weak_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate weak supervision specific metrics.
        
        Args:
            weak_labels: Weak labels matrix
            true_labels: True labels (optional)
            
        Returns:
            Dictionary of metrics
        """
        ws_metrics = WeakSupervisionMetrics()
        
        metrics = {
            'coverage': ws_metrics.label_model_coverage(weak_labels),
            'conflict_rate': ws_metrics.label_model_conflict(weak_labels)
        }
        
        # Per-LF metrics
        n_lfs = weak_labels.shape[1]
        lf_coverages = []
        lf_accuracies = []
        
        for lf_idx in range(n_lfs):
            coverage = ws_metrics.labeling_function_coverage(weak_labels, lf_idx)
            lf_coverages.append(coverage)
            
            if true_labels is not None:
                accuracy = ws_metrics.labeling_function_accuracy(weak_labels, true_labels, lf_idx)
                lf_accuracies.append(accuracy)
        
        metrics['avg_lf_coverage'] = np.mean(lf_coverages)
        metrics['min_lf_coverage'] = np.min(lf_coverages)
        metrics['max_lf_coverage'] = np.max(lf_coverages)
        
        if lf_accuracies:
            metrics['avg_lf_accuracy'] = np.mean(lf_accuracies)
            metrics['min_lf_accuracy'] = np.min(lf_accuracies)
            metrics['max_lf_accuracy'] = np.max(lf_accuracies)
        
        return metrics


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    weak_labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate all available metrics for a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        weak_labels: Weak labels matrix (optional)
        class_names: Class names (optional)
        
    Returns:
        Dictionary of all metrics
    """
    calculator = MetricCalculator()
    
    # Standard metrics
    standard_metrics = calculator.calculate_all_metrics(
        y_true, y_pred, y_prob, class_names
    )
    
    results = {
        'standard_metrics': standard_metrics.to_dict()
    }
    
    # Weak supervision metrics
    if weak_labels is not None:
        ws_metrics = calculator.calculate_weak_supervision_metrics(
            weak_labels, y_true
        )
        results['weak_supervision_metrics'] = ws_metrics
    
    return results


def compare_model_performance(
    models_results: Dict[str, BenchmarkMetrics],
    metric: str = 'f1_score'
) -> pd.DataFrame:
    """
    Compare performance across multiple models.
    
    Args:
        models_results: Dictionary mapping model names to BenchmarkMetrics
        metric: Primary metric for ranking
        
    Returns:
        Comparison DataFrame sorted by the specified metric
    """
    comparison_data = []
    
    for model_name, metrics in models_results.items():
        row = {
            'model': model_name,
            'accuracy': metrics.accuracy,
            'f1_score': metrics.f1_score,
            'precision': metrics.precision,
            'recall': metrics.recall
        }
        
        if metrics.auc_roc is not None:
            row['auc_roc'] = metrics.auc_roc
        
        if metrics.auc_pr is not None:
            row['auc_pr'] = metrics.auc_pr
        
        if metrics.matthews_corr is not None:
            row['matthews_corr'] = metrics.matthews_corr
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by specified metric (descending)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    
    return df


def metric_significance_test(
    metric_values_1: List[float],
    metric_values_2: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test statistical significance between two sets of metric values.
    
    Args:
        metric_values_1: Metric values from first model
        metric_values_2: Metric values from second model
        alpha: Significance level
        
    Returns:
        Test results
    """
    from scipy import stats
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(metric_values_1, metric_values_2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(metric_values_1) - 1) * np.var(metric_values_1, ddof=1) +
         (len(metric_values_2) - 1) * np.var(metric_values_2, ddof=1)) /
        (len(metric_values_1) + len(metric_values_2) - 2)
    )
    
    cohen_d = (np.mean(metric_values_1) - np.mean(metric_values_2)) / pooled_std
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohen_d,
        'mean_1': np.mean(metric_values_1),
        'mean_2': np.mean(metric_values_2),
        'std_1': np.std(metric_values_1),
        'std_2': np.std(metric_values_2)
    }


# Alias for backward compatibility  
ModelPerformanceMetrics = BenchmarkMetrics


def create_performance_report(
    model_name: str,
    metrics: BenchmarkMetrics,
    weak_labels: Optional[np.ndarray] = None,
    output_format: str = 'dict'
) -> Union[Dict[str, Any], str]:
    """
    Create a comprehensive performance report.
    
    Args:
        model_name: Name of the model
        metrics: Benchmark metrics
        weak_labels: Optional weak labels for WS metrics
        output_format: Output format ('dict', 'json', 'markdown')
        
    Returns:
        Performance report in specified format
    """
    report = {
        'model_name': model_name,
        'standard_metrics': metrics.to_dict(),
        'summary': {
            'overall_score': (metrics.accuracy + metrics.f1_score) / 2,
            'strengths': [],
            'weaknesses': []
        }
    }
    
    # Add analysis
    if metrics.f1_score > 0.8:
        report['summary']['strengths'].append('High F1 score indicating good balance of precision and recall')
    elif metrics.f1_score < 0.6:
        report['summary']['weaknesses'].append('Low F1 score suggesting poor classification performance')
    
    if metrics.precision > 0.85:
        report['summary']['strengths'].append('High precision with few false positives')
    elif metrics.precision < 0.7:
        report['summary']['weaknesses'].append('Low precision with many false positives')
    
    if metrics.recall > 0.85:
        report['summary']['strengths'].append('High recall capturing most positive cases')
    elif metrics.recall < 0.7:
        report['summary']['weaknesses'].append('Low recall missing many positive cases')
    
    # Add weak supervision metrics if available
    if weak_labels is not None:
        calculator = MetricCalculator()
        ws_metrics = calculator.calculate_weak_supervision_metrics(weak_labels)
        report['weak_supervision_metrics'] = ws_metrics
        
        if ws_metrics['coverage'] > 0.8:
            report['summary']['strengths'].append('High labeling function coverage')
        elif ws_metrics['coverage'] < 0.5:
            report['summary']['weaknesses'].append('Low labeling function coverage')
    
    if output_format == 'json':
        import json
        return json.dumps(report, indent=2, default=str)
    elif output_format == 'markdown':
        return _format_report_markdown(report)
    else:
        return report


def _format_report_markdown(report: Dict[str, Any]) -> str:
    """Format report as markdown."""
    md = f"# Performance Report: {report['model_name']}\n\n"
    
    # Standard metrics
    md += "## Standard Metrics\n\n"
    metrics = report['standard_metrics']
    md += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
    md += f"- **F1 Score**: {metrics['f1_score']:.3f}\n"
    md += f"- **Precision**: {metrics['precision']:.3f}\n"
    md += f"- **Recall**: {metrics['recall']:.3f}\n"
    
    if metrics.get('auc_roc'):
        md += f"- **AUC-ROC**: {metrics['auc_roc']:.3f}\n"
    
    # Summary
    md += "\n## Summary\n\n"
    md += f"**Overall Score**: {report['summary']['overall_score']:.3f}\n\n"
    
    if report['summary']['strengths']:
        md += "### Strengths\n"
        for strength in report['summary']['strengths']:
            md += f"- {strength}\n"
        md += "\n"
    
    if report['summary']['weaknesses']:
        md += "### Areas for Improvement\n"
        for weakness in report['summary']['weaknesses']:
            md += f"- {weakness}\n"
        md += "\n"
    
    # Weak supervision metrics
    if 'weak_supervision_metrics' in report:
        md += "## Weak Supervision Metrics\n\n"
        ws_metrics = report['weak_supervision_metrics']
        md += f"- **Coverage**: {ws_metrics['coverage']:.3f}\n"
        md += f"- **Conflict Rate**: {ws_metrics['conflict_rate']:.3f}\n"
        md += f"- **Average LF Accuracy**: {ws_metrics.get('avg_lf_accuracy', 'N/A')}\n"
    
    return md


# Additional utility function for backward compatibility
def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    **kwargs
) -> BenchmarkMetrics:
    """
    Evaluate model performance with standard metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities (optional)
        **kwargs: Additional arguments
        
    Returns:
        BenchmarkMetrics object
    """
    calculator = MetricCalculator()
    return calculator.calculate_all_metrics(y_true, y_pred, y_prob)
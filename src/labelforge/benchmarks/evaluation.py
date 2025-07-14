"""
Evaluation utilities for benchmarking weak supervision methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score

from ..types import Example, LFOutput
from ..label_model import LabelModel
from .metrics import BenchmarkMetrics, MetricCalculator, evaluate_model_performance


@dataclass
class CrossValidationResult:
    """Results from cross-validation evaluation."""
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    fold_results: List[BenchmarkMetrics]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean_accuracy': self.mean_accuracy,
            'std_accuracy': self.std_accuracy,
            'mean_f1': self.mean_f1,
            'std_f1': self.std_f1,
            'fold_results': [result.to_dict() for result in self.fold_results]
        }


class CrossValidator:
    """
    Cross-validation for weak supervision models.
    """
    
    def __init__(self, n_folds: int = 5, stratified: bool = True, random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of folds
            stratified: Whether to use stratified sampling
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state
        
        if stratified:
            self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        else:
            self.cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    def cross_validate(
        self,
        examples: List[Example],
        lf_output: LFOutput,
        true_labels: np.ndarray,
        model_class=LabelModel,
        model_kwargs: Optional[Dict] = None
    ) -> CrossValidationResult:
        """
        Perform cross-validation.
        
        Args:
            examples: List of examples
            lf_output: Labeling function output
            true_labels: True labels
            model_class: Model class to use
            model_kwargs: Model initialization arguments
            
        Returns:
            Cross-validation results
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        fold_results = []
        accuracies = []
        f1_scores = []
        
        # Get weak labels matrix
        weak_labels = lf_output.labels
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(weak_labels, true_labels)):
            # Split data
            train_weak_labels = weak_labels[train_idx]
            test_weak_labels = weak_labels[test_idx]
            train_true_labels = true_labels[train_idx]
            test_true_labels = true_labels[test_idx]
            
            # Create LFOutput for training
            train_lf_output = LFOutput(
                labels=train_weak_labels,
                example_ids=[examples[i].id for i in train_idx],
                lf_names=lf_output.lf_names
            )
            
            # Train model
            model = model_class(**model_kwargs)
            model.fit(train_lf_output)
            
            # Predict on test set
            test_lf_output = LFOutput(
                labels=test_weak_labels,
                example_ids=[examples[i].id for i in test_idx],
                lf_names=lf_output.lf_names
            )
            
            predictions = model.predict(test_lf_output)
            probabilities = model.predict_proba(test_lf_output)
            
            # Calculate metrics
            calculator = MetricCalculator()
            metrics = calculator.calculate_all_metrics(
                test_true_labels, predictions, probabilities
            )
            
            fold_results.append(metrics)
            accuracies.append(metrics.accuracy)
            f1_scores.append(metrics.f1_score)
        
        return CrossValidationResult(
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_f1=np.mean(f1_scores),
            std_f1=np.std(f1_scores),
            fold_results=fold_results
        )


class BenchmarkEvaluator:
    """
    Comprehensive benchmark evaluation.
    """
    
    def __init__(self):
        """Initialize benchmark evaluator."""
        self.results = {}
        self.calculator = MetricCalculator()
    
    def evaluate_model(
        self,
        model_name: str,
        model,
        test_examples: List[Example],
        test_lf_output: LFOutput,
        test_true_labels: np.ndarray
    ) -> BenchmarkMetrics:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            model: Trained model
            test_examples: Test examples
            test_lf_output: Test LF output
            test_true_labels: True test labels
            
        Returns:
            Benchmark metrics
        """
        predictions = model.predict(test_lf_output)
        probabilities = model.predict_proba(test_lf_output)
        
        metrics = self.calculator.calculate_all_metrics(
            test_true_labels, predictions, probabilities
        )
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self, metric: str = 'f1_score') -> pd.DataFrame:
        """
        Compare evaluated models.
        
        Args:
            metric: Primary metric for comparison
            
        Returns:
            Comparison DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        from .metrics import compare_model_performance
        return compare_model_performance(self.results, metric)


class ModelComparison:
    """
    Statistical comparison of models.
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.model_results = {}
    
    def add_model_results(
        self,
        model_name: str,
        cv_results: CrossValidationResult
    ):
        """Add cross-validation results for a model."""
        self.model_results[model_name] = cv_results
    
    def statistical_comparison(
        self,
        model1: str,
        model2: str,
        metric: str = 'accuracy',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare two models statistically.
        
        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
            alpha: Significance level
            
        Returns:
            Statistical test results
        """
        if model1 not in self.model_results or model2 not in self.model_results:
            raise ValueError("Both models must have results added")
        
        results1 = self.model_results[model1]
        results2 = self.model_results[model2]
        
        # Extract metric values from fold results
        if metric == 'accuracy':
            values1 = [fold.accuracy for fold in results1.fold_results]
            values2 = [fold.accuracy for fold in results2.fold_results]
        elif metric == 'f1_score':
            values1 = [fold.f1_score for fold in results1.fold_results]
            values2 = [fold.f1_score for fold in results2.fold_results]
        else:
            raise ValueError(f"Metric {metric} not supported")
        
        from .metrics import metric_significance_test
        return metric_significance_test(values1, values2, alpha)
    
    def generate_report(self) -> str:
        """Generate a comparison report."""
        report = "Model Comparison Report\n"
        report += "=" * 25 + "\n\n"
        
        for model_name, results in self.model_results.items():
            report += f"{model_name}:\n"
            report += f"  Mean Accuracy: {results.mean_accuracy:.3f} ± {results.std_accuracy:.3f}\n"
            report += f"  Mean F1 Score: {results.mean_f1:.3f} ± {results.std_f1:.3f}\n"
            report += "\n"
        
        return report


def evaluate_model(
    model,
    examples: List[Example],
    lf_output: LFOutput,
    true_labels: np.ndarray
) -> BenchmarkMetrics:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        examples: Test examples
        lf_output: LF output
        true_labels: True labels
        
    Returns:
        Benchmark metrics
    """
    predictions = model.predict(lf_output)
    probabilities = model.predict_proba(lf_output)
    
    calculator = MetricCalculator()
    return calculator.calculate_all_metrics(true_labels, predictions, probabilities)


def cross_validate_model(
    model_class,
    examples: List[Example],
    lf_output: LFOutput,
    true_labels: np.ndarray,
    n_folds: int = 5,
    model_kwargs: Optional[Dict] = None
) -> CrossValidationResult:
    """
    Cross-validate a model.
    
    Args:
        model_class: Model class
        examples: Examples
        lf_output: LF output
        true_labels: True labels
        n_folds: Number of folds
        model_kwargs: Model initialization arguments
        
    Returns:
        Cross-validation results
    """
    cv = CrossValidator(n_folds=n_folds)
    return cv.cross_validate(examples, lf_output, true_labels, model_class, model_kwargs)
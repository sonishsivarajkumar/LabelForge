"""
Advanced evaluation framework for weak supervision research.

This module provides statistical testing, cross-validation, and evaluation
protocols specifically designed for weak supervision scenarios.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    from scipy import stats
    from scipy.stats import bootstrap
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Statistical testing disabled.")

try:
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. Cross-validation disabled.")

from ..types import Example, LFOutput
from ..label_model import LabelModel


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    is_significant: Optional[bool] = None
    alpha: float = 0.05
    
    def __post_init__(self):
        if self.is_significant is None:
            self.is_significant = self.p_value < self.alpha


@dataclass
class CrossValidationResult:
    """Result of cross-validation evaluation."""
    method_name: str
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    metric_name: str = "accuracy"
    
    def __str__(self):
        return (f"{self.method_name}: {self.mean_score:.3f} ± {self.std_score:.3f} "
                f"({self.metric_name})")


class StatisticalTester:
    """
    Statistical significance testing for weak supervision experiments.
    
    Provides various statistical tests to compare method performance
    and determine statistical significance.
    """
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level
            n_bootstrap: Number of bootstrap samples
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy is required for statistical testing")
        
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
    
    def paired_t_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Perform paired t-test between two sets of scores.
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score arrays must have same length")
        
        statistic, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        diff = scores_a - scores_b
        pooled_std = np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2)
        effect_size = np.mean(diff) / pooled_std if pooled_std > 0 else 0
        
        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            alpha=self.alpha
        )
    
    def wilcoxon_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric).
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            alternative: Alternative hypothesis
            
        Returns:
            Statistical test result
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Score arrays must have same length")
        
        # Remove pairs where both scores are equal
        diff = scores_a - scores_b
        non_zero_mask = diff != 0
        
        if np.sum(non_zero_mask) == 0:
            # All differences are zero
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0,
                p_value=1.0,
                alpha=self.alpha
            )
        
        statistic, p_value = stats.wilcoxon(
            scores_a[non_zero_mask], 
            scores_b[non_zero_mask],
            alternative=alternative
        )
        
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha
        )
    
    def mcnemar_test(
        self,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        true_labels: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform McNemar's test for comparing classifier performance.
        
        Args:
            predictions_a: Predictions from method A
            predictions_b: Predictions from method B
            true_labels: Ground truth labels
            
        Returns:
            Statistical test result
        """
        # Create contingency table
        correct_a = (predictions_a == true_labels)
        correct_b = (predictions_b == true_labels)
        
        # McNemar's test focuses on disagreements
        both_correct = np.sum(correct_a & correct_b)
        both_wrong = np.sum(~correct_a & ~correct_b)
        a_correct_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_correct = np.sum(~correct_a & correct_b)
        
        # McNemar's statistic
        if a_correct_b_wrong + a_wrong_b_correct == 0:
            # No disagreements
            return StatisticalTestResult(
                test_name="McNemar's test",
                statistic=0,
                p_value=1.0,
                alpha=self.alpha
            )
        
        statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1)**2 / (a_correct_b_wrong + a_wrong_b_correct)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return StatisticalTestResult(
            test_name="McNemar's test",
            statistic=statistic,
            p_value=p_value,
            alpha=self.alpha
        )
    
    def bootstrap_confidence_interval(
        self,
        scores: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'percentile'
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for scores.
        
        Args:
            scores: Performance scores
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Bootstrap method ('percentile', 'bca')
            
        Returns:
            Confidence interval (lower, upper)
        """
        if len(scores) == 0:
            return (0.0, 0.0)
        
        # Simple percentile bootstrap
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    def compare_multiple_methods(
        self,
        method_scores: Dict[str, np.ndarray],
        test_type: str = 'friedman'
    ) -> Dict[str, Any]:
        """
        Compare multiple methods using appropriate statistical test.
        
        Args:
            method_scores: Dictionary mapping method names to score arrays
            test_type: Type of test ('friedman', 'anova')
            
        Returns:
            Dictionary with test results and post-hoc comparisons
        """
        method_names = list(method_scores.keys())
        score_arrays = [method_scores[name] for name in method_names]
        
        if test_type == 'friedman':
            statistic, p_value = stats.friedmanchisquare(*score_arrays)
            test_name = "Friedman test"
        elif test_type == 'anova':
            statistic, p_value = stats.f_oneway(*score_arrays)
            test_name = "One-way ANOVA"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        result = {
            'overall_test': StatisticalTestResult(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                alpha=self.alpha
            ),
            'pairwise_comparisons': {}
        }
        
        # Post-hoc pairwise comparisons if overall test is significant
        if p_value < self.alpha:
            for i, method_a in enumerate(method_names):
                for j, method_b in enumerate(method_names[i+1:], i+1):
                    if test_type == 'friedman':
                        # Use Wilcoxon for post-hoc
                        pairwise_result = self.wilcoxon_test(
                            method_scores[method_a],
                            method_scores[method_b]
                        )
                    else:
                        # Use t-test for post-hoc
                        pairwise_result = self.paired_t_test(
                            method_scores[method_a],
                            method_scores[method_b]
                        )
                    
                    result['pairwise_comparisons'][f"{method_a}_vs_{method_b}"] = pairwise_result
        
        return result


class CrossValidationEvaluator:
    """
    Cross-validation evaluation specifically designed for weak supervision.
    
    Handles the unique challenges of evaluating weak supervision methods,
    including proper handling of unlabeled data and labeling function dependencies.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validation evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn is required for cross-validation")
        
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def stratified_cv(
        self,
        examples: List[Example],
        lf_output: LFOutput,
        true_labels: np.ndarray,
        model_class: type = LabelModel,
        model_params: Optional[Dict[str, Any]] = None,
        scoring: str = 'accuracy'
    ) -> CrossValidationResult:
        """
        Perform stratified cross-validation for weak supervision.
        
        Args:
            examples: List of examples
            lf_output: Labeling function output
            true_labels: Ground truth labels for evaluation
            model_class: Label model class to use
            model_params: Parameters for label model
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if model_params is None:
            model_params = {}
        
        # Create stratified folds
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        cv_scores = []
        
        for train_idx, val_idx in skf.split(examples, true_labels):
            # Split data
            train_votes = lf_output.votes[train_idx]
            val_votes = lf_output.votes[val_idx]
            
            train_example_ids = [lf_output.example_ids[i] for i in train_idx]
            val_example_ids = [lf_output.example_ids[i] for i in val_idx]
            
            train_lf_output = LFOutput(votes=train_votes, example_ids=train_example_ids, lf_names=lf_output.lf_names)
            val_lf_output = LFOutput(votes=val_votes, example_ids=val_example_ids, lf_names=lf_output.lf_names)
            
            val_labels = true_labels[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(train_lf_output)
            
            # Make predictions
            predictions = model.predict(val_lf_output)
            
            # Calculate score
            if scoring == 'accuracy':
                score = accuracy_score(val_labels, predictions)
            elif scoring == 'f1':
                score = f1_score(val_labels, predictions, average='weighted')
            elif scoring == 'precision':
                score = precision_score(val_labels, predictions, average='weighted')
            elif scoring == 'recall':
                score = recall_score(val_labels, predictions, average='weighted')
            else:
                raise ValueError(f"Unknown scoring metric: {scoring}")
            
            cv_scores.append(score)
        
        cv_scores = np.array(cv_scores)
        
        # Calculate confidence interval
        tester = StatisticalTester()
        ci = tester.bootstrap_confidence_interval(cv_scores)
        
        return CrossValidationResult(
            method_name=model_class.__name__,
            cv_scores=cv_scores,
            mean_score=np.mean(cv_scores),
            std_score=np.std(cv_scores),
            confidence_interval=ci,
            metric_name=scoring
        )
    
    def time_series_cv(
        self,
        examples: List[Example],
        lf_output: LFOutput,
        true_labels: np.ndarray,
        model_class: type = LabelModel,
        model_params: Optional[Dict[str, Any]] = None,
        scoring: str = 'accuracy'
    ) -> CrossValidationResult:
        """
        Perform time series cross-validation.
        
        Useful when examples have temporal ordering that must be preserved.
        
        Args:
            examples: List of examples (assumed to be in temporal order)
            lf_output: Labeling function output
            true_labels: Ground truth labels
            model_class: Label model class
            model_params: Model parameters
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if model_params is None:
            model_params = {}
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(examples):
            # Split data maintaining temporal order
            train_votes = lf_output.votes[train_idx]
            val_votes = lf_output.votes[val_idx]
            
            train_example_ids = [lf_output.example_ids[i] for i in train_idx]
            val_example_ids = [lf_output.example_ids[i] for i in val_idx]
            
            train_lf_output = LFOutput(votes=train_votes, example_ids=train_example_ids, lf_names=lf_output.lf_names)
            val_lf_output = LFOutput(votes=val_votes, example_ids=val_example_ids, lf_names=lf_output.lf_names)
            
            val_labels = true_labels[val_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(train_lf_output)
            
            # Make predictions
            predictions = model.predict(val_lf_output)
            
            # Calculate score
            if scoring == 'accuracy':
                score = accuracy_score(val_labels, predictions)
            elif scoring == 'f1':
                score = f1_score(val_labels, predictions, average='weighted')
            else:
                raise ValueError(f"Unknown scoring metric: {scoring}")
            
            cv_scores.append(score)
        
        cv_scores = np.array(cv_scores)
        
        # Calculate confidence interval
        tester = StatisticalTester()
        ci = tester.bootstrap_confidence_interval(cv_scores)
        
        return CrossValidationResult(
            method_name=f"{model_class.__name__}_TimeSeries",
            cv_scores=cv_scores,
            mean_score=np.mean(cv_scores),
            std_score=np.std(cv_scores),
            confidence_interval=ci,
            metric_name=scoring
        )


class EvaluationProtocol:
    """
    Standardized evaluation protocol for weak supervision research.
    
    Provides consistent evaluation procedures that can be used across
    different studies and papers for fair comparison.
    """
    
    def __init__(
        self,
        protocol_name: str = "standard",
        cv_folds: int = 5,
        n_bootstrap: int = 1000,
        test_split: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize evaluation protocol.
        
        Args:
            protocol_name: Name of the evaluation protocol
            cv_folds: Number of cross-validation folds
            n_bootstrap: Number of bootstrap samples
            test_split: Test set split ratio
            random_state: Random state for reproducibility
        """
        self.protocol_name = protocol_name
        self.cv_folds = cv_folds
        self.n_bootstrap = n_bootstrap
        self.test_split = test_split
        self.random_state = random_state
        
        # Initialize components
        self.cv_evaluator = CrossValidationEvaluator(cv_folds, random_state)
        self.statistical_tester = StatisticalTester(n_bootstrap=n_bootstrap)
    
    def full_evaluation(
        self,
        methods: Dict[str, Callable],
        examples: List[Example],
        lf_output: LFOutput,
        true_labels: np.ndarray,
        metrics: List[str] = ['accuracy', 'f1']
    ) -> Dict[str, Any]:
        """
        Perform full evaluation protocol.
        
        Args:
            methods: Dictionary of method names to model creation functions
            examples: List of examples
            lf_output: Labeling function output
            true_labels: Ground truth labels
            metrics: List of metrics to evaluate
            
        Returns:
            Complete evaluation results
        """
        results = {
            'protocol_name': self.protocol_name,
            'cv_results': {},
            'statistical_tests': {},
            'summary_statistics': {}
        }
        
        # Perform cross-validation for each method and metric
        for metric in metrics:
            results['cv_results'][metric] = {}
            method_scores = {}
            
            for method_name, method_func in methods.items():
                # Create model
                model = method_func()
                
                # Perform cross-validation
                cv_result = self.cv_evaluator.stratified_cv(
                    examples=examples,
                    lf_output=lf_output,
                    true_labels=true_labels,
                    model_class=type(model),
                    model_params={},  # Assume default params
                    scoring=metric
                )
                
                results['cv_results'][metric][method_name] = cv_result
                method_scores[method_name] = cv_result.cv_scores
            
            # Statistical comparison
            if len(methods) > 1:
                comparison_result = self.statistical_tester.compare_multiple_methods(
                    method_scores
                )
                results['statistical_tests'][metric] = comparison_result
        
        # Generate summary statistics
        results['summary_statistics'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results."""
        summary = {
            'best_methods': {},
            'significant_differences': {},
            'performance_ranking': {}
        }
        
        for metric, cv_results in results['cv_results'].items():
            # Find best method for this metric
            best_method = max(cv_results.keys(), 
                            key=lambda x: cv_results[x].mean_score)
            summary['best_methods'][metric] = {
                'method': best_method,
                'score': cv_results[best_method].mean_score,
                'std': cv_results[best_method].std_score
            }
            
            # Rank methods
            method_ranking = sorted(cv_results.keys(),
                                  key=lambda x: cv_results[x].mean_score,
                                  reverse=True)
            summary['performance_ranking'][metric] = method_ranking
            
            # Check for significant differences
            if metric in results['statistical_tests']:
                stat_result = results['statistical_tests'][metric]
                summary['significant_differences'][metric] = {
                    'overall_significant': stat_result['overall_test'].is_significant,
                    'significant_pairs': [
                        pair for pair, test in stat_result['pairwise_comparisons'].items()
                        if test.is_significant
                    ]
                }
        
        return summary


class ReproducibilityChecker:
    """
    Tools for ensuring reproducible weak supervision experiments.
    
    Helps track experimental conditions and validate result consistency.
    """
    
    def __init__(self):
        """Initialize reproducibility checker."""
        self.experiment_records = []
    
    def record_experiment(
        self,
        experiment_name: str,
        method_name: str,
        hyperparameters: Dict[str, Any],
        results: Dict[str, float],
        data_hash: Optional[str] = None,
        environment_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Record an experiment for reproducibility tracking.
        
        Args:
            experiment_name: Name of the experiment
            method_name: Name of the method
            hyperparameters: Method hyperparameters
            results: Experimental results
            data_hash: Hash of the dataset used
            environment_info: Environment information
            
        Returns:
            Unique experiment ID
        """
        import time
        import hashlib
        
        experiment_id = hashlib.md5(
            f"{experiment_name}_{method_name}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        record = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'method_name': method_name,
            'hyperparameters': hyperparameters,
            'results': results,
            'data_hash': data_hash,
            'environment_info': environment_info or {},
            'timestamp': time.time()
        }
        
        self.experiment_records.append(record)
        return experiment_id
    
    def validate_reproduction(
        self,
        original_experiment_id: str,
        new_results: Dict[str, float],
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Validate that new results reproduce original experiment.
        
        Args:
            original_experiment_id: ID of original experiment
            new_results: New experimental results
            tolerance: Tolerance for considering results equivalent
            
        Returns:
            Validation report
        """
        # Find original experiment
        original_exp = None
        for record in self.experiment_records:
            if record['experiment_id'] == original_experiment_id:
                original_exp = record
                break
        
        if original_exp is None:
            return {'status': 'error', 'message': 'Original experiment not found'}
        
        # Compare results
        original_results = original_exp['results']
        
        validation_report = {
            'status': 'success',
            'reproducible': True,
            'metric_comparisons': {},
            'overall_difference': 0.0
        }
        
        total_diff = 0.0
        n_metrics = 0
        
        for metric, original_value in original_results.items():
            if metric in new_results:
                new_value = new_results[metric]
                diff = abs(original_value - new_value)
                is_close = diff <= tolerance
                
                validation_report['metric_comparisons'][metric] = {
                    'original': original_value,
                    'new': new_value,
                    'difference': diff,
                    'within_tolerance': is_close
                }
                
                if not is_close:
                    validation_report['reproducible'] = False
                
                total_diff += diff
                n_metrics += 1
        
        if n_metrics > 0:
            validation_report['overall_difference'] = total_diff / n_metrics
        
        return validation_report

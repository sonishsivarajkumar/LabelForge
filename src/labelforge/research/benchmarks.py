"""
Benchmarking suite for weak supervision methods.

This module provides standardized benchmarks, datasets, and evaluation
protocols for comparing weak supervision approaches.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
from abc import ABC, abstractmethod

from ..types import Example, LFOutput
from ..label_model import LabelModel
from ..lf import LabelingFunction


@dataclass
class BenchmarkResult:
    """Container for benchmark evaluation results."""
    method_name: str
    dataset_name: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc_score: Optional[float] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    n_examples: Optional[int] = None
    n_features: Optional[int] = None
    additional_metrics: Optional[Dict[str, float]] = None


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def load_data(self) -> Tuple[List[Example], np.ndarray]:
        """Load the dataset examples and ground truth labels."""
        pass
    
    @abstractmethod
    def get_labeling_functions(self) -> List[LabelingFunction]:
        """Get the labeling functions for this dataset."""
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> List[str]:
        """Get the standard evaluation metrics for this dataset."""
        pass


class WRENCHBenchmark(BenchmarkDataset):
    """
    Integration with WRENCH benchmark datasets for weak supervision.
    
    Provides access to standardized datasets from the WRENCH paper
    for fair comparison of weak supervision methods.
    """
    
    AVAILABLE_DATASETS = [
        "youtube", "sms", "imdb", "yelp", "amazon", "agnews",
        "trec", "chemprot", "census", "commercial", "tennis"
    ]
    
    def __init__(self, dataset_name: str, data_path: Optional[str] = None):
        """
        Initialize WRENCH benchmark dataset.
        
        Args:
            dataset_name: Name of the WRENCH dataset
            data_path: Path to dataset files (optional)
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available. "
                           f"Choose from: {self.AVAILABLE_DATASETS}")
        
        super().__init__(
            name=f"WRENCH-{dataset_name}",
            description=f"WRENCH benchmark dataset: {dataset_name}"
        )
        
        self.dataset_name = dataset_name
        self.data_path = data_path or f"./data/wrench/{dataset_name}"
        self._examples = None
        self._labels = None
        self._lfs = None
    
    def load_data(self) -> Tuple[List[Example], np.ndarray]:
        """Load WRENCH dataset."""
        if self._examples is None or self._labels is None:
            self._load_wrench_data()
        return self._examples, self._labels
    
    def _load_wrench_data(self):
        """Load data from WRENCH format files."""
        data_path = Path(self.data_path)
        
        # Load training data
        train_file = data_path / "train.json"
        if not train_file.exists():
            raise FileNotFoundError(f"WRENCH data not found at {train_file}")
        
        with open(train_file, 'r') as f:
            train_data = [json.loads(line) for line in f]
        
        # Load test data
        test_file = data_path / "test.json"
        test_data = []
        if test_file.exists():
            with open(test_file, 'r') as f:
                test_data = [json.loads(line) for line in f]
        
        # Combine train and test
        all_data = train_data + test_data
        
        # Convert to Examples and labels
        self._examples = []
        labels = []
        
        for item in all_data:
            # Extract text
            text = item.get('text', item.get('data', ''))
            
            # Create example
            example = Example(
                text=text,
                example_id=item.get('id', len(self._examples)),
                metadata={'source': 'wrench', 'dataset': self.dataset_name}
            )
            self._examples.append(example)
            
            # Extract label
            label = item.get('label', -1)
            labels.append(label)
        
        self._labels = np.array(labels)
    
    def get_labeling_functions(self) -> List[LabelingFunction]:
        """Get labeling functions for WRENCH dataset."""
        if self._lfs is None:
            self._create_default_lfs()
        return self._lfs
    
    def _create_default_lfs(self):
        """Create default labeling functions based on dataset type."""
        from ..templates import KeywordLF, RegexLF, SentimentLF
        
        self._lfs = []
        
        # Create dataset-specific LFs
        if self.dataset_name in ['youtube', 'sms', 'imdb', 'yelp', 'amazon']:
            # Sentiment/review datasets
            self._lfs.extend([
                SentimentLF("positive_sentiment", positive_threshold=0.6),
                SentimentLF("negative_sentiment", negative_threshold=0.4),
                KeywordLF("positive_keywords", 
                         keywords=['good', 'great', 'excellent', 'amazing', 'love'],
                         label=1),
                KeywordLF("negative_keywords",
                         keywords=['bad', 'terrible', 'awful', 'hate', 'worst'],
                         label=0)
            ])
        
        elif self.dataset_name in ['agnews', 'trec']:
            # Classification datasets
            self._lfs.extend([
                KeywordLF("sports_keywords",
                         keywords=['sports', 'game', 'team', 'player', 'score'],
                         label=0),
                KeywordLF("politics_keywords", 
                         keywords=['politics', 'government', 'election', 'president'],
                         label=1),
                KeywordLF("tech_keywords",
                         keywords=['technology', 'computer', 'software', 'internet'],
                         label=2)
            ])
        
        # Add length-based LFs
        def short_text_lf(example):
            return 0 if len(example.text.split()) < 10 else -1
        
        def long_text_lf(example):
            return 1 if len(example.text.split()) > 50 else -1
        
        self._lfs.extend([
            LabelingFunction("short_text", short_text_lf),
            LabelingFunction("long_text", long_text_lf)
        ])
    
    def get_evaluation_metrics(self) -> List[str]:
        """Get standard evaluation metrics."""
        return ['accuracy', 'f1_score', 'precision', 'recall', 'auc_score']


class SyntheticDataGenerator:
    """
    Generate synthetic datasets for testing weak supervision methods.
    
    Useful for controlled experiments and method validation.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize synthetic data generator."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_classification_dataset(
        self,
        n_examples: int = 1000,
        n_classes: int = 2,
        n_features: int = 10,
        noise_level: float = 0.1,
        class_balance: Optional[List[float]] = None
    ) -> Tuple[List[Example], np.ndarray, np.ndarray]:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_examples: Number of examples to generate
            n_classes: Number of classes
            n_features: Number of features per example
            noise_level: Amount of noise to add
            class_balance: Class distribution (optional)
            
        Returns:
            Tuple of (examples, features, labels)
        """
        # Generate class labels
        if class_balance is None:
            class_balance = [1.0/n_classes] * n_classes
        
        labels = np.random.choice(
            n_classes, 
            size=n_examples, 
            p=class_balance
        )
        
        # Generate features based on class
        features = np.zeros((n_examples, n_features))
        
        for i in range(n_examples):
            # Class-dependent feature generation
            class_mean = labels[i] * 2.0  # Separate classes
            features[i] = np.random.normal(
                class_mean, 
                1.0 + noise_level, 
                n_features
            )
        
        # Create text examples from features
        examples = []
        for i in range(n_examples):
            # Convert features to text representation
            feature_words = [f"feature_{j}_{features[i,j]:.2f}" 
                           for j in range(min(5, n_features))]
            text = " ".join(feature_words)
            
            example = Example(
                text=text,
                example_id=i,
                metadata={
                    'synthetic': True,
                    'true_label': int(labels[i]),
                    'features': features[i].tolist()
                }
            )
            examples.append(example)
        
        return examples, features, labels
    
    def generate_weak_labels(
        self,
        true_labels: np.ndarray,
        n_lfs: int = 5,
        accuracy_range: Tuple[float, float] = (0.6, 0.9),
        coverage_range: Tuple[float, float] = (0.3, 0.8)
    ) -> np.ndarray:
        """
        Generate weak labels from true labels.
        
        Args:
            true_labels: Ground truth labels
            n_lfs: Number of labeling functions to simulate
            accuracy_range: Range of LF accuracies
            coverage_range: Range of LF coverage rates
            
        Returns:
            Weak label matrix (n_examples x n_lfs)
        """
        n_examples = len(true_labels)
        n_classes = len(np.unique(true_labels))
        
        weak_labels = np.full((n_examples, n_lfs), -1)  # -1 = abstain
        
        for lf_idx in range(n_lfs):
            # Sample accuracy and coverage for this LF
            accuracy = np.random.uniform(*accuracy_range)
            coverage = np.random.uniform(*coverage_range)
            
            # Determine which examples this LF will label
            labeled_mask = np.random.random(n_examples) < coverage
            labeled_indices = np.where(labeled_mask)[0]
            
            for idx in labeled_indices:
                true_label = true_labels[idx]
                
                # Generate label based on accuracy
                if np.random.random() < accuracy:
                    # Correct label
                    weak_labels[idx, lf_idx] = true_label
                else:
                    # Random incorrect label
                    possible_labels = [l for l in range(n_classes) if l != true_label]
                    weak_labels[idx, lf_idx] = np.random.choice(possible_labels)
        
        return weak_labels


class StandardDatasets:
    """
    Registry of standard weak supervision datasets.
    
    Provides easy access to commonly used benchmarking datasets
    and their metadata.
    """
    
    # Standard dataset configurations
    DATASETS = {
        'spam': {
            'name': 'Spam Detection',
            'description': 'Email spam classification dataset',
            'n_classes': 2,
            'size': 'medium',
            'domain': 'text'
        },
        'sentiment': {
            'name': 'Sentiment Analysis',
            'description': 'Movie review sentiment classification',
            'n_classes': 2,
            'size': 'large',
            'domain': 'text'
        },
        'medical': {
            'name': 'Medical Diagnosis',
            'description': 'Medical condition classification',
            'n_classes': 5,
            'size': 'small',
            'domain': 'medical'
        },
        'finance': {
            'name': 'Financial Fraud',
            'description': 'Financial transaction fraud detection',
            'n_classes': 2,
            'size': 'large',
            'domain': 'finance'
        }
    }
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available standard datasets."""
        return list(cls.DATASETS.keys())
    
    @classmethod
    def get_dataset_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if name not in cls.DATASETS:
            raise ValueError(f"Dataset {name} not found. Available: {cls.list_datasets()}")
        return cls.DATASETS[name]
    
    @classmethod
    def create_benchmark(cls, name: str) -> BenchmarkDataset:
        """Create a benchmark dataset instance."""
        if name not in cls.DATASETS:
            raise ValueError(f"Dataset {name} not found. Available: {cls.list_datasets()}")
        
        info = cls.DATASETS[name]
        
        # For now, return a synthetic dataset based on the configuration
        # In a real implementation, this would load actual datasets
        class StandardBenchmark(BenchmarkDataset):
            def __init__(self, dataset_info):
                super().__init__(
                    name=dataset_info['name'],
                    description=dataset_info['description']
                )
                self.dataset_info = dataset_info
                self._examples = None
                self._labels = None
            
            def load_data(self) -> Tuple[List[Example], np.ndarray]:
                if self._examples is None:
                    # Generate synthetic data based on dataset configuration
                    n_samples = {'small': 1000, 'medium': 5000, 'large': 10000}[
                        self.dataset_info['size']
                    ]
                    
                    # Create synthetic examples
                    self._examples = [
                        Example(
                            id=f"{name}_{i}",
                            data=f"Sample {i} for {self.dataset_info['domain']} domain",
                            metadata={'domain': self.dataset_info['domain']}
                        )
                        for i in range(n_samples)
                    ]
                    
                    # Generate synthetic labels
                    self._labels = np.random.randint(
                        0, self.dataset_info['n_classes'], n_samples
                    )
                
                return self._examples, self._labels
            
            def generate_weak_labels(
                self, 
                labeling_functions: List[LabelingFunction]
            ) -> np.ndarray:
                examples, true_labels = self.load_data()
                n_examples = len(examples)
                n_lfs = len(labeling_functions)
                
                # Generate synthetic weak labels with some noise
                weak_labels = np.full((n_examples, n_lfs), -1, dtype=int)
                
                for lf_idx, lf in enumerate(labeling_functions):
                    # Simulate LF accuracy based on dataset difficulty
                    accuracy = 0.7 if self.dataset_info['size'] == 'large' else 0.8
                    coverage = np.random.uniform(0.6, 0.9)  # Random coverage
                    
                    for ex_idx in range(n_examples):
                        if np.random.random() < coverage:  # LF fires
                            if np.random.random() < accuracy:  # Correct prediction
                                weak_labels[ex_idx, lf_idx] = true_labels[ex_idx]
                            else:  # Incorrect prediction
                                # Random incorrect label
                                possible_labels = [
                                    l for l in range(self.dataset_info['n_classes']) 
                                    if l != true_labels[ex_idx]
                                ]
                                if possible_labels:
                                    weak_labels[ex_idx, lf_idx] = np.random.choice(possible_labels)
                
                return weak_labels
        
        return StandardBenchmark(info)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for weak supervision methods.
    
    Provides standardized evaluation protocols and comparison tools.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_benchmark(
        self,
        methods: Dict[str, Callable],
        datasets: List[BenchmarkDataset],
        n_trials: int = 5,
        save_results: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across methods and datasets.
        
        Args:
            methods: Dictionary of method names to callable methods
            datasets: List of benchmark datasets
            n_trials: Number of trials per method-dataset combination
            save_results: Whether to save results to disk
            
        Returns:
            List of benchmark results
        """
        all_results = []
        
        for dataset in datasets:
            print(f"\nEvaluating on dataset: {dataset.name}")
            
            # Load dataset
            examples, true_labels = dataset.load_data()
            labeling_functions = dataset.get_labeling_functions()
            
            # Apply labeling functions
            lf_output = self._apply_labeling_functions(examples, labeling_functions)
            
            for method_name, method_func in methods.items():
                print(f"  Running method: {method_name}")
                
                trial_results = []
                
                for trial in range(n_trials):
                    print(f"    Trial {trial + 1}/{n_trials}")
                    
                    # Run method
                    start_time = time.time()
                    predictions = method_func(lf_output, true_labels)
                    end_time = time.time()
                    
                    # Evaluate predictions
                    result = self._evaluate_predictions(
                        predictions=predictions,
                        true_labels=true_labels,
                        method_name=f"{method_name}_trial_{trial}",
                        dataset_name=dataset.name,
                        training_time=end_time - start_time,
                        n_examples=len(examples)
                    )
                    
                    trial_results.append(result)
                
                # Aggregate trial results
                avg_result = self._aggregate_results(trial_results, method_name, dataset.name)
                all_results.append(avg_result)
        
        if save_results:
            self._save_results(all_results)
        
        self.results.extend(all_results)
        return all_results
    
    def _apply_labeling_functions(
        self, 
        examples: List[Example], 
        lfs: List[LabelingFunction]
    ) -> LFOutput:
        """Apply labeling functions to examples."""
        votes = np.full((len(examples), len(lfs)), -1)
        
        for i, example in enumerate(examples):
            for j, lf in enumerate(lfs):
                try:
                    vote = lf(example)
                    votes[i, j] = vote if vote is not None else -1
                except Exception:
                    votes[i, j] = -1  # Abstain on error
        
        return LFOutput(
            votes=votes,
            example_ids=[ex.example_id for ex in examples]
        )
    
    def _evaluate_predictions(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        method_name: str,
        dataset_name: str,
        training_time: float,
        n_examples: int
    ) -> BenchmarkResult:
        """Evaluate predictions against ground truth."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            roc_auc_score = None
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        
        auc = None
        if roc_auc_score and len(np.unique(true_labels)) == 2:
            try:
                auc = roc_auc_score(true_labels, predictions)
            except Exception:
                pass
        
        return BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            accuracy=accuracy,
            f1_score=f1,
            precision=precision,
            recall=recall,
            auc_score=auc,
            training_time=training_time,
            n_examples=n_examples
        )
    
    def _aggregate_results(
        self,
        trial_results: List[BenchmarkResult],
        method_name: str,
        dataset_name: str
    ) -> BenchmarkResult:
        """Aggregate results across trials."""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_score', 'training_time']
        
        aggregated = {}
        for metric in metrics:
            values = [getattr(r, metric) for r in trial_results if getattr(r, metric) is not None]
            if values:
                aggregated[metric] = np.mean(values)
            else:
                aggregated[metric] = None
        
        return BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            accuracy=aggregated['accuracy'],
            f1_score=aggregated['f1_score'], 
            precision=aggregated['precision'],
            recall=aggregated['recall'],
            auc_score=aggregated['auc_score'],
            training_time=aggregated['training_time'],
            n_examples=trial_results[0].n_examples
        )
    
    def _save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to file."""
        results_data = []
        for result in results:
            result_dict = {
                'method_name': result.method_name,
                'dataset_name': result.dataset_name,
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'auc_score': result.auc_score,
                'training_time': result.training_time,
                'n_examples': result.n_examples
            }
            results_data.append(result_dict)
        
        # Save as JSON
        output_file = self.output_dir / f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_data)
        csv_file = output_file.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {output_file} and {csv_file}")
    
    def compare_methods(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        metric: str = 'accuracy',
        save_plot: bool = True
    ) -> pd.DataFrame:
        """
        Compare methods across datasets using specified metric.
        
        Args:
            results: List of results (uses self.results if None)
            metric: Metric to compare
            save_plot: Whether to save comparison plot
            
        Returns:
            Comparison DataFrame
        """
        if results is None:
            results = self.results
        
        # Create comparison table
        data = []
        for result in results:
            data.append({
                'Method': result.method_name,
                'Dataset': result.dataset_name,
                'Metric': getattr(result, metric)
            })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='Method', columns='Dataset', values='Metric')
        
        # Create plot if matplotlib available
        if save_plot:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
                plt.title(f'Method Comparison - {metric.title()}')
                plt.tight_layout()
                
                plot_file = self.output_dir / f"method_comparison_{metric}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Comparison plot saved to {plot_file}")
            except ImportError:
                print("Matplotlib/Seaborn not available. Skipping plot generation.")
        
        return pivot_df


def compare_methods(
    methods: Dict[str, Callable],
    datasets: List[BenchmarkDataset],
    metrics: List[str] = ['accuracy', 'f1_score'],
    n_trials: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to compare multiple methods.
    
    Args:
        methods: Dictionary of method names to callable methods
        datasets: List of benchmark datasets  
        metrics: List of metrics to compare
        n_trials: Number of trials per method-dataset combination
        
    Returns:
        Dictionary mapping metrics to comparison DataFrames
    """
    suite = BenchmarkSuite()
    results = suite.run_benchmark(methods, datasets, n_trials)
    
    comparisons = {}
    for metric in metrics:
        comparisons[metric] = suite.compare_methods(results, metric)
    
    return comparisons

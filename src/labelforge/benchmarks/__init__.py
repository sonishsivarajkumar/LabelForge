"""
LabelForge Benchmarking Suite

This module provides standardized benchmarks and evaluation tools
for weak supervision methods.
"""

from .datasets import (
    StandardDatasets,
    BenchmarkDatasetInfo,
    DatasetRegistry,
    DATASET_REGISTRY,
    load_benchmark_dataset,
    create_synthetic_dataset,
    get_dataset_statistics
)

from .metrics import (
    WeakSupervisionMetrics,
    BenchmarkMetrics,
    MetricCalculator,
    calculate_all_metrics,
    compare_model_performance,
    evaluate_model_performance,
    ModelPerformanceMetrics
)

from .evaluation import (
    CrossValidator,
    BenchmarkEvaluator,
    ModelComparison,
    evaluate_model,
    cross_validate_model
)

from .reproducibility import (
    ReproducibilityManager,
    ExperimentConfig,
    set_random_seed,
    capture_environment,
    save_experiment_config
)

__all__ = [
    # Datasets
    'StandardDatasets',
    'BenchmarkDatasetInfo', 
    'DatasetRegistry',
    'DATASET_REGISTRY',
    'load_benchmark_dataset',
    'create_synthetic_dataset',
    'get_dataset_statistics',
    
    # Metrics
    'WeakSupervisionMetrics',
    'BenchmarkMetrics',
    'MetricCalculator',
    'calculate_all_metrics',
    'compare_model_performance',
    'evaluate_model_performance',
    'ModelPerformanceMetrics',
    
    # Evaluation
    'CrossValidator',
    'BenchmarkEvaluator',
    'ModelComparison',
    'evaluate_model',
    'cross_validate_model',
    
    # Reproducibility
    'ReproducibilityManager',
    'ExperimentConfig',
    'set_random_seed',
    'capture_environment',
    'save_experiment_config'
]
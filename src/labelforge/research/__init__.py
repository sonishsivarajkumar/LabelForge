"""
LabelForge Research Utilities

This module provides academic research tools including benchmarking,
statistical testing, publication utilities, and reproducibility features.
"""

from .benchmarks import (
    BenchmarkSuite,
    WRENCHBenchmark,
    StandardDatasets,
    SyntheticDataGenerator,
    BenchmarkResult,
    BenchmarkDataset,
    compare_methods
)

from .evaluation import (
    StatisticalTester,
    StatisticalTestResult,
    CrossValidationResult,
    CrossValidationEvaluator,
    EvaluationProtocol,
    ReproducibilityChecker
)

from .publication import (
    LaTeXExporter,
    AcademicPlotter,
    ResultSummarizer,
    ExperimentTracker,
    CitationFormatter,
    TableGenerator
)

from .reproducibility import (
    ExperimentConfig,
    EnvironmentCapture,
    SeedManager,
    DatasetVersioner,
    ResultArchiver,
    ReproducibilityReport
)

__all__ = [
    # Benchmarking
    "BenchmarkSuite",
    "WRENCHBenchmark", 
    "StandardDatasets",
    "SyntheticDataGenerator",
    "BenchmarkResults",
    "compare_methods",
    
    # Evaluation
    "StatisticalTester",
    "SignificanceTest",
    "BootstrapCI", 
    "CrossValidationEvaluator",
    "EvaluationProtocol",
    "ReproducibilityChecker",
    
    # Publication
    "LaTeXExporter",
    "AcademicPlotter",
    "ResultSummarizer",
    "ExperimentTracker",
    "CitationFormatter", 
    "TableGenerator",
    
    # Reproducibility
    "ExperimentConfig",
    "EnvironmentCapture",
    "SeedManager",
    "DatasetVersioner",
    "ResultArchiver",
    "ReproducibilityReport"
]

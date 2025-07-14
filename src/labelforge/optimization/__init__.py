"""
Performance optimization tools for LabelForge.

This module provides utilities for optimizing weak supervision workflows,
including parallel processing, memory management, and performance monitoring.
"""

from .parallel import (
    ParallelLFApplicator,
    DistributedLabelModel,
    MultiProcessingPool,
    parallel_apply_lfs,
    chunk_examples
)

from .memory import (
    MemoryEfficientDataset,
    StreamingLFApplicator,
    LazyEvaluator,
    CacheManager,
    memory_monitor
)

from .profiling import (
    PerformanceProfiler,
    RuntimeProfiler,
    MemoryProfiler,
    BottleneckDetector,
    profile_function,
    benchmark_performance,
    profile_lf_application
)

# Add PerformanceMonitor as an alias for now
PerformanceMonitor = RuntimeProfiler

from .acceleration import (
    GPUAccelerator,
    CUDALabelModel,
    VectorizedOperations,
    OptimizedMatrixOps,
    batch_operations
)

__all__ = [
    # Parallel processing
    "ParallelLFApplicator",
    "DistributedLabelModel", 
    "MultiProcessingPool",
    "parallel_apply_lfs",
    "chunk_examples",
    
    # Memory optimization
    "MemoryEfficientDataset",
    "StreamingLFApplicator",
    "LazyEvaluator",
    "CacheManager",
    "memory_monitor",
    
    # Profiling
    "PerformanceProfiler",
    "RuntimeProfiler",
    "MemoryProfiler",
    "BottleneckDetector",
    "profile_function",
    
    # GPU acceleration
    "GPUAccelerator",
    "CUDALabelModel",
    "VectorizedOperations",
    "OptimizedMatrixOps",
    "batch_operations"
]

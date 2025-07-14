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
    ProfileResult,
    BottleneckInfo,
    RuntimeProfiler,
    profile_function,
    benchmark_performance,
    profile_lf_application
)

# Add aliases for compatibility
PerformanceProfiler = RuntimeProfiler
MemoryProfiler = RuntimeProfiler
BottleneckDetector = RuntimeProfiler
PerformanceMonitor = RuntimeProfiler
ParallelProcessor = ParallelLFApplicator

# Placeholder classes for acceleration (TODO: implement)
class GPUAccelerator:
    """Placeholder for GPU acceleration."""
    pass

class CUDALabelModel:
    """Placeholder for CUDA-accelerated LabelModel."""
    pass

class VectorizedOperations:
    """Placeholder for vectorized operations."""
    pass

class OptimizedMatrixOps:
    """Placeholder for optimized matrix operations."""
    pass

def batch_operations(*args, **kwargs):
    """Placeholder for batch operations."""
    pass

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

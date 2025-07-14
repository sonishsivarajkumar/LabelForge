"""
Memory optimization utilities for LabelForge.

This module provides tools for managing memory usage when working with
large datasets and complex labeling function pipelines.
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Iterator, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile
import warnings
from contextlib import contextmanager

from ..types import Example, LFOutput
from ..lf import LabelingFunction


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: float  # Total system memory in GB
    available_memory: float  # Available memory in GB
    used_memory: float  # Used memory in GB
    process_memory: float  # Current process memory in GB
    memory_percent: float  # Memory usage percentage


class MemoryEfficientDataset:
    """
    Memory-efficient dataset that can handle large datasets by streaming
    examples from disk and managing memory usage.
    """
    
    def __init__(
        self,
        examples: Union[List[Example], str, Path],
        chunk_size: int = 1000,
        cache_size: int = 10000,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize memory-efficient dataset.
        
        Args:
            examples: List of examples or path to serialized examples
            chunk_size: Number of examples to process at once
            cache_size: Maximum number of examples to keep in memory
            temp_dir: Temporary directory for spill-over storage
        """
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load or store examples
        if isinstance(examples, (str, Path)):
            self.data_path = Path(examples)
            self._examples = None
            self._load_from_disk()
        else:
            self._examples = examples
            self.data_path = None
            self._maybe_spill_to_disk()
        
        self._cache = {}
        self._cache_order = []
    
    def _load_from_disk(self):
        """Load examples from disk."""
        if self.data_path and self.data_path.exists():
            with open(self.data_path, 'rb') as f:
                self._examples = pickle.load(f)
    
    def _maybe_spill_to_disk(self):
        """Spill examples to disk if memory usage is high."""
        memory_stats = get_memory_stats()
        
        # If memory usage > 80%, spill to disk
        if memory_stats.memory_percent > 80 and self._examples:
            if not self.data_path:
                self.data_path = self.temp_dir / f"examples_{id(self)}.pkl"
            
            with open(self.data_path, 'wb') as f:
                pickle.dump(self._examples, f)
            
            # Keep only a sample in memory
            if len(self._examples) > self.cache_size:
                self._cache = {
                    i: ex for i, ex in enumerate(self._examples[:self.cache_size])
                }
                self._examples = None
                gc.collect()
    
    def __len__(self) -> int:
        """Get number of examples."""
        if self._examples:
            return len(self._examples)
        elif self.data_path and self.data_path.exists():
            # Load just to count
            with open(self.data_path, 'rb') as f:
                examples = pickle.load(f)
                return len(examples)
        return 0
    
    def __getitem__(self, idx: int) -> Example:
        """Get example by index."""
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        # Load from memory
        if self._examples:
            example = self._examples[idx]
        else:
            # Load from disk
            if not self.data_path or not self.data_path.exists():
                raise IndexError(f"Index {idx} out of range")
            
            with open(self.data_path, 'rb') as f:
                examples = pickle.load(f)
                if idx >= len(examples):
                    raise IndexError(f"Index {idx} out of range")
                example = examples[idx]
        
        # Add to cache
        self._update_cache(idx, example)
        return example
    
    def _update_cache(self, idx: int, example: Example):
        """Update cache with LRU eviction."""
        if idx in self._cache:
            # Move to end (most recent)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
        else:
            # Add new item
            if len(self._cache) >= self.cache_size:
                # Evict oldest
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            
            self._cache[idx] = example
            self._cache_order.append(idx)
    
    def iter_chunks(self) -> Iterator[List[Example]]:
        """Iterate over examples in chunks."""
        total_examples = len(self)
        
        for start_idx in range(0, total_examples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_examples)
            chunk = [self[i] for i in range(start_idx, end_idx)]
            yield chunk
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.data_path and self.data_path.exists() and self.temp_dir in self.data_path.parents:
            self.data_path.unlink()


class StreamingLFApplicator:
    """
    Memory-efficient labeling function applicator that processes
    examples in streams to minimize memory usage.
    """
    
    def __init__(
        self,
        labeling_functions: List[LabelingFunction],
        chunk_size: int = 1000,
        n_jobs: int = 1
    ):
        """
        Initialize streaming applicator.
        
        Args:
            labeling_functions: List of labeling functions to apply
            chunk_size: Number of examples to process at once
            n_jobs: Number of parallel jobs
        """
        self.labeling_functions = labeling_functions
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
    
    def apply_streaming(
        self,
        dataset: MemoryEfficientDataset,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Apply labeling functions to dataset in streaming fashion.
        
        Args:
            dataset: Memory-efficient dataset
            progress_callback: Optional progress callback function
            
        Returns:
            Weak labels matrix
        """
        total_examples = len(dataset)
        n_lfs = len(self.labeling_functions)
        
        # Pre-allocate result matrix
        weak_labels = np.full((total_examples, n_lfs), -1, dtype=int)
        
        processed = 0
        
        for chunk_idx, chunk in enumerate(dataset.iter_chunks()):
            chunk_size = len(chunk)
            start_idx = chunk_idx * self.chunk_size
            end_idx = start_idx + chunk_size
            
            # Apply LFs to chunk
            chunk_labels = self._apply_to_chunk(chunk)
            
            # Store results
            weak_labels[start_idx:end_idx] = chunk_labels
            
            processed += chunk_size
            
            if progress_callback:
                progress_callback(processed, total_examples)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        return weak_labels
    
    def _apply_to_chunk(self, chunk: List[Example]) -> np.ndarray:
        """Apply labeling functions to a chunk of examples."""
        chunk_size = len(chunk)
        n_lfs = len(self.labeling_functions)
        chunk_labels = np.full((chunk_size, n_lfs), -1, dtype=int)
        
        for lf_idx, lf in enumerate(self.labeling_functions):
            for ex_idx, example in enumerate(chunk):
                try:
                    result = lf.apply(example)
                    if result != -1:  # LF fired
                        chunk_labels[ex_idx, lf_idx] = result
                except Exception as e:
                    warnings.warn(f"LF {lf.name} failed on example {example.id}: {e}")
        
        return chunk_labels


class LazyEvaluator:
    """
    Lazy evaluation system that computes results only when needed
    and caches them for future use.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """Initialize lazy evaluator."""
        self.max_cache_size = max_cache_size
        self._cache = {}
        self._cache_order = []
    
    def lazy_apply(
        self,
        func: Callable,
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Lazily apply function with caching.
        
        Args:
            func: Function to apply
            *args: Function arguments
            cache_key: Optional cache key (auto-generated if None)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        if cache_key is None:
            # Generate cache key from function and arguments
            cache_key = self._generate_cache_key(func, args, kwargs)
        
        # Check cache
        if cache_key in self._cache:
            # Move to end (most recent)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key]
        
        # Compute result
        result = func(*args, **kwargs)
        
        # Add to cache
        self._update_cache(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        import hashlib
        
        # Create a string representation
        key_str = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
        
        # Hash for consistent length
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: Any):
        """Update cache with LRU eviction."""
        if len(self._cache) >= self.max_cache_size:
            # Evict oldest
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        self._cache_order.append(key)
    
    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_order.clear()


class CacheManager:
    """
    Centralized cache management for LabelForge operations.
    """
    
    def __init__(self, max_memory_mb: int = 500):
        """
        Initialize cache manager.
        
        Args:
            max_memory_mb: Maximum memory to use for caching in MB
        """
        self.max_memory_mb = max_memory_mb
        self._caches = {}
        self._memory_usage = 0
    
    def create_cache(self, name: str, max_size: int = 1000) -> LazyEvaluator:
        """Create a named cache."""
        if name in self._caches:
            return self._caches[name]
        
        cache = LazyEvaluator(max_cache_size=max_size)
        self._caches[name] = cache
        return cache
    
    def clear_cache(self, name: Optional[str] = None):
        """Clear cache(s)."""
        if name:
            if name in self._caches:
                self._caches[name].clear_cache()
        else:
            for cache in self._caches.values():
                cache.clear_cache()
            gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'total_mb': self.max_memory_mb,
            'used_mb': self._memory_usage,
            'available_mb': self.max_memory_mb - self._memory_usage
        }


def get_memory_stats() -> MemoryStats:
    """Get current memory usage statistics."""
    memory = psutil.virtual_memory()
    process = psutil.Process()
    
    return MemoryStats(
        total_memory=memory.total / (1024**3),  # GB
        available_memory=memory.available / (1024**3),  # GB
        used_memory=memory.used / (1024**3),  # GB
        process_memory=process.memory_info().rss / (1024**3),  # GB
        memory_percent=memory.percent
    )


@contextmanager
def memory_monitor(threshold_percent: float = 90.0):
    """
    Context manager for monitoring memory usage.
    
    Args:
        threshold_percent: Memory usage threshold to warn at
    """
    initial_stats = get_memory_stats()
    
    try:
        yield initial_stats
    finally:
        final_stats = get_memory_stats()
        
        if final_stats.memory_percent > threshold_percent:
            warnings.warn(
                f"Memory usage high: {final_stats.memory_percent:.1f}% "
                f"({final_stats.used_memory:.1f}GB used)"
            )
        
        # Log memory change
        memory_change = final_stats.process_memory - initial_stats.process_memory
        if abs(memory_change) > 0.1:  # > 100MB change
            print(f"Memory change: {memory_change:+.1f}GB")


def optimize_memory_usage():
    """Optimize memory usage by running garbage collection and clearing caches."""
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory stats
    stats = get_memory_stats()
    
    print(f"Garbage collected {collected} objects")
    print(f"Memory usage: {stats.memory_percent:.1f}% ({stats.used_memory:.1f}GB)")
    
    return stats


def memory_efficient_chunk_size(
    total_examples: int,
    memory_limit_gb: float = 2.0,
    bytes_per_example: int = 1024
) -> int:
    """
    Calculate optimal chunk size based on memory constraints.
    
    Args:
        total_examples: Total number of examples
        memory_limit_gb: Memory limit in GB
        bytes_per_example: Estimated bytes per example
        
    Returns:
        Optimal chunk size
    """
    # Convert to bytes
    memory_limit_bytes = memory_limit_gb * (1024**3)
    
    # Calculate max examples that fit in memory
    max_examples = int(memory_limit_bytes / bytes_per_example)
    
    # Use smaller of max_examples or total_examples
    chunk_size = min(max_examples, total_examples)
    
    # Ensure minimum chunk size
    chunk_size = max(chunk_size, 100)
    
    return chunk_size

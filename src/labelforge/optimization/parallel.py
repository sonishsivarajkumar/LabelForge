"""
Parallel processing utilities for weak supervision.

This module provides tools for parallelizing labeling function application
and model training across multiple cores and distributed systems.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import time
import warnings
from functools import partial

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("Joblib not available. Some parallel features disabled.")

from ..types import Example, LFOutput
from ..lf import LabelingFunction
from ..label_model import LabelModel


class ParallelLFApplicator:
    """
    Apply labeling functions in parallel across multiple processes.
    
    Provides efficient parallel execution of labeling functions
    with automatic load balancing and error handling.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = 'multiprocessing',
        chunk_size: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize parallel LF applicator.
        
        Args:
            n_jobs: Number of parallel jobs (-1 uses all available cores)
            backend: Parallel backend ('multiprocessing', 'threading', 'joblib')
            chunk_size: Size of example chunks for processing
            verbose: Whether to show progress information
        """
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.backend = backend
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        if backend == 'joblib' and not HAS_JOBLIB:
            warnings.warn("Joblib not available, falling back to multiprocessing")
            self.backend = 'multiprocessing'
    
    def apply_lfs(
        self,
        examples: List[Example],
        labeling_functions: List[LabelingFunction],
        show_progress: bool = True
    ) -> LFOutput:
        """
        Apply labeling functions in parallel.
        
        Args:
            examples: List of examples to label
            labeling_functions: List of labeling functions
            show_progress: Whether to show progress bar
            
        Returns:
            LFOutput with voting matrix
        """
        n_examples = len(examples)
        n_lfs = len(labeling_functions)
        
        if self.verbose:
            print(f"Applying {n_lfs} labeling functions to {n_examples} examples "
                  f"using {self.n_jobs} workers")
        
        # Determine chunk size
        if self.chunk_size is None:
            # Aim for ~4 chunks per worker for good load balancing
            self.chunk_size = max(1, n_examples // (self.n_jobs * 4))
        
        # Create chunks
        example_chunks = list(chunk_examples(examples, self.chunk_size))
        
        if self.backend == 'joblib':
            return self._apply_lfs_joblib(example_chunks, labeling_functions)
        elif self.backend == 'multiprocessing':
            return self._apply_lfs_multiprocessing(example_chunks, labeling_functions)
        elif self.backend == 'threading':
            return self._apply_lfs_threading(example_chunks, labeling_functions)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _apply_lfs_joblib(
        self,
        example_chunks: List[List[Example]],
        labeling_functions: List[LabelingFunction]
    ) -> LFOutput:
        """Apply LFs using joblib backend."""
        if not HAS_JOBLIB:
            raise ImportError("Joblib is required for joblib backend")
        
        # Process chunks in parallel
        with Parallel(n_jobs=self.n_jobs, verbose=1 if self.verbose else 0) as parallel:
            chunk_results = parallel(
                delayed(_apply_lfs_to_chunk)(chunk, labeling_functions)
                for chunk in example_chunks
            )
        
        return self._combine_chunk_results(chunk_results, example_chunks, labeling_functions)
    
    def _apply_lfs_multiprocessing(
        self,
        example_chunks: List[List[Example]],
        labeling_functions: List[LabelingFunction]
    ) -> LFOutput:
        """Apply LFs using multiprocessing backend."""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(_apply_lfs_to_chunk, chunk, labeling_functions): i
                for i, chunk in enumerate(example_chunks)
            }
            
            # Collect results in order
            chunk_results = [None] * len(example_chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results[chunk_idx] = future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing chunk {chunk_idx}: {e}")
                    # Create empty result for failed chunk
                    chunk_size = len(example_chunks[chunk_idx])
                    chunk_results[chunk_idx] = np.full(
                        (chunk_size, len(labeling_functions)), -1
                    )
        
        return self._combine_chunk_results(chunk_results, example_chunks, labeling_functions)
    
    def _apply_lfs_threading(
        self,
        example_chunks: List[List[Example]],
        labeling_functions: List[LabelingFunction]
    ) -> LFOutput:
        """Apply LFs using threading backend."""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(_apply_lfs_to_chunk, chunk, labeling_functions): i
                for i, chunk in enumerate(example_chunks)
            }
            
            # Collect results in order
            chunk_results = [None] * len(example_chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results[chunk_idx] = future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing chunk {chunk_idx}: {e}")
                    # Create empty result for failed chunk
                    chunk_size = len(example_chunks[chunk_idx])
                    chunk_results[chunk_idx] = np.full(
                        (chunk_size, len(labeling_functions)), -1
                    )
        
        return self._combine_chunk_results(chunk_results, example_chunks, labeling_functions)
    
    def _combine_chunk_results(
        self,
        chunk_results: List[np.ndarray],
        example_chunks: List[List[Example]],
        labeling_functions: List[LabelingFunction]
    ) -> LFOutput:
        """Combine results from parallel chunks."""
        # Combine vote matrices
        votes = np.vstack(chunk_results)
        
        # Combine example IDs
        example_ids = []
        for chunk in example_chunks:
            example_ids.extend([ex.example_id for ex in chunk])
        
        # Extract LF names
        lf_names = [lf.name for lf in labeling_functions]
        
        return LFOutput(votes=votes, example_ids=example_ids, lf_names=lf_names)


class DistributedLabelModel:
    """
    Distributed training for label models across multiple workers.
    
    Enables training on larger datasets that don't fit on a single machine.
    """
    
    def __init__(
        self,
        base_model: LabelModel,
        n_workers: int = 2,
        aggregation_method: str = 'averaging'
    ):
        """
        Initialize distributed label model.
        
        Args:
            base_model: Base label model to distribute
            n_workers: Number of worker processes
            aggregation_method: Method for aggregating worker results
        """
        self.base_model = base_model
        self.n_workers = n_workers
        self.aggregation_method = aggregation_method
        self.worker_models = []
    
    def fit(
        self,
        lf_output: LFOutput,
        y_dev: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'DistributedLabelModel':
        """
        Fit distributed label model.
        
        Args:
            lf_output: LF output for training
            y_dev: Development labels for validation
            **kwargs: Additional arguments for base model
            
        Returns:
            Self
        """
        # Split data across workers
        worker_data = self._split_data(lf_output, self.n_workers)
        
        # Train workers in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            for worker_lf_output in worker_data:
                future = executor.submit(
                    _train_worker_model,
                    self.base_model,
                    worker_lf_output,
                    y_dev,
                    kwargs
                )
                futures.append(future)
            
            # Collect trained models
            self.worker_models = []
            for future in as_completed(futures):
                try:
                    worker_model = future.result()
                    self.worker_models.append(worker_model)
                except Exception as e:
                    warnings.warn(f"Worker model training failed: {e}")
        
        # Aggregate worker models
        self._aggregate_models()
        
        return self
    
    def predict(self, lf_output: LFOutput) -> np.ndarray:
        """
        Make predictions using distributed model.
        
        Args:
            lf_output: LF output for prediction
            
        Returns:
            Predicted labels
        """
        if not self.worker_models:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all workers
        worker_predictions = []
        for model in self.worker_models:
            try:
                pred = model.predict(lf_output)
                worker_predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Worker prediction failed: {e}")
        
        if not worker_predictions:
            raise RuntimeError("All worker predictions failed")
        
        # Aggregate predictions
        worker_predictions = np.array(worker_predictions)
        
        if self.aggregation_method == 'averaging':
            # Average probabilities then argmax
            worker_probs = np.array([
                model.predict_proba(lf_output) 
                for model in self.worker_models
            ])
            avg_probs = np.mean(worker_probs, axis=0)
            return np.argmax(avg_probs, axis=1)
        
        elif self.aggregation_method == 'voting':
            # Majority voting
            return np.array([
                np.bincount(worker_predictions[:, i]).argmax()
                for i in range(worker_predictions.shape[1])
            ])
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def predict_proba(self, lf_output: LFOutput) -> np.ndarray:
        """
        Predict class probabilities using distributed model.
        
        Args:
            lf_output: LF output for prediction
            
        Returns:
            Class probabilities
        """
        if not self.worker_models:
            raise ValueError("Model must be fitted before prediction")
        
        # Get probabilities from all workers
        worker_probs = []
        for model in self.worker_models:
            try:
                probs = model.predict_proba(lf_output)
                worker_probs.append(probs)
            except Exception as e:
                warnings.warn(f"Worker probability prediction failed: {e}")
        
        if not worker_probs:
            raise RuntimeError("All worker probability predictions failed")
        
        # Average probabilities
        worker_probs = np.array(worker_probs)
        return np.mean(worker_probs, axis=0)
    
    def _split_data(
        self,
        lf_output: LFOutput,
        n_splits: int
    ) -> List[LFOutput]:
        """Split LF output across workers."""
        n_examples = lf_output.votes.shape[0]
        chunk_size = n_examples // n_splits
        
        worker_data = []
        
        for i in range(n_splits):
            start_idx = i * chunk_size
            if i == n_splits - 1:
                # Last worker gets remaining examples
                end_idx = n_examples
            else:
                end_idx = (i + 1) * chunk_size
            
            worker_votes = lf_output.votes[start_idx:end_idx]
            worker_ids = lf_output.example_ids[start_idx:end_idx]
            
            worker_lf_output = LFOutput(
                votes=worker_votes,
                example_ids=worker_ids
            )
            worker_data.append(worker_lf_output)
        
        return worker_data
    
    def _aggregate_models(self):
        """Aggregate trained worker models."""
        if not self.worker_models:
            return
        
        # For now, just use the first model
        # In a more sophisticated implementation, we would
        # average the parameters
        self.base_model = self.worker_models[0]


class MultiProcessingPool:
    """
    Utility class for managing multiprocessing pools.
    
    Provides context management and resource cleanup.
    """
    
    def __init__(self, n_processes: Optional[int] = None):
        """
        Initialize multiprocessing pool.
        
        Args:
            n_processes: Number of processes (None uses all available)
        """
        self.n_processes = n_processes or cpu_count()
        self.pool = None
    
    def __enter__(self):
        """Enter context manager."""
        self.pool = Pool(self.n_processes)
        return self.pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.pool:
            self.pool.close()
            self.pool.join()


def chunk_examples(
    examples: List[Example],
    chunk_size: int
) -> Iterator[List[Example]]:
    """
    Split examples into chunks for parallel processing.
    
    Args:
        examples: List of examples to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of examples
    """
    for i in range(0, len(examples), chunk_size):
        yield examples[i:i + chunk_size]


def parallel_apply_lfs(
    examples: List[Example],
    labeling_functions: List[LabelingFunction],
    n_jobs: int = -1,
    backend: str = 'multiprocessing',
    verbose: bool = False
) -> LFOutput:
    """
    Convenience function for parallel LF application.
    
    Args:
        examples: List of examples
        labeling_functions: List of labeling functions
        n_jobs: Number of parallel jobs
        backend: Parallel backend
        verbose: Whether to show progress
        
    Returns:
        LF output with votes
    """
    applicator = ParallelLFApplicator(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose
    )
    
    return applicator.apply_lfs(examples, labeling_functions)


def _apply_lfs_to_chunk(
    chunk: List[Example],
    labeling_functions: List[LabelingFunction]
) -> np.ndarray:
    """
    Apply labeling functions to a chunk of examples.
    
    This function is designed to be called in a separate process.
    
    Args:
        chunk: Chunk of examples
        labeling_functions: List of labeling functions
        
    Returns:
        Vote matrix for the chunk
    """
    n_examples = len(chunk)
    n_lfs = len(labeling_functions)
    
    votes = np.full((n_examples, n_lfs), -1)
    
    for i, example in enumerate(chunk):
        for j, lf in enumerate(labeling_functions):
            try:
                vote = lf(example)
                votes[i, j] = vote if vote is not None else -1
            except Exception:
                # Abstain on any error
                votes[i, j] = -1
    
    return votes


def _train_worker_model(
    base_model: LabelModel,
    lf_output: LFOutput,
    y_dev: Optional[np.ndarray],
    fit_kwargs: Dict[str, Any]
) -> LabelModel:
    """
    Train a worker model on a subset of data.
    
    This function is designed to be called in a separate process.
    
    Args:
        base_model: Base model to copy and train
        lf_output: LF output for this worker
        y_dev: Development labels
        fit_kwargs: Keyword arguments for fit method
        
    Returns:
        Trained model
    """
    # Create a copy of the base model
    worker_model = LabelModel(
        cardinality=base_model.cardinality,
        device=base_model.device
    )
    
    # Train on worker data
    worker_model.fit(lf_output, y_dev=y_dev, **fit_kwargs)
    
    return worker_model

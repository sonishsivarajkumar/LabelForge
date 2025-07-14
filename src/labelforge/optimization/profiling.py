"""
Performance profiling tools for LabelForge.

This module provides utilities for profiling and optimizing the performance
of weak supervision workflows, including runtime analysis and bottleneck detection.
"""

import time
import psutil
import cProfile
import pstats
import io
import functools
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from contextlib import contextmanager

from ..types import Example, LFOutput
from ..lf import LabelingFunction


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_percent: float
    call_count: int
    stats: Dict[str, Any]


@dataclass
class BottleneckInfo:
    """Information about a performance bottleneck."""
    function_name: str
    total_time: float
    percentage_of_total: float
    call_count: int
    time_per_call: float
    description: str


class RuntimeProfiler:
    """
    Profile runtime performance of LabelForge operations.
    
    Provides detailed analysis of execution time, memory usage,
    and performance bottlenecks.
    """
    
    def __init__(self, name: str = "profiler"):
        """
        Initialize runtime profiler.
        
        Args:
            name: Name for this profiler instance
        """
        self.name = name
        self.results = []
        self.active_profiles = {}
        self.cumulative_stats = {}
        
    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.run_profiled(func, *args, **kwargs)
        return wrapper
    
    def run_profiled(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a function with profiling.
        
        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        func_name = func.__name__
        
        # Start profiling
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu = process.cpu_percent()
        
        # Create profiler
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Run function
            result = func(*args, **kwargs)
            
            # Stop profiling
            profiler.disable()
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            end_cpu = process.cpu_percent()
            
            # Extract stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats()
            stats_output = s.getvalue()
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=func_name,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_percent=end_cpu - start_cpu,
                call_count=1,
                stats={
                    'detailed_stats': stats_output,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
            
            self.results.append(profile_result)
            
            # Update cumulative stats
            if func_name not in self.cumulative_stats:
                self.cumulative_stats[func_name] = {
                    'total_time': 0,
                    'total_calls': 0,
                    'total_memory': 0,
                    'max_time': 0,
                    'min_time': float('inf')
                }
            
            stats = self.cumulative_stats[func_name]
            stats['total_time'] += profile_result.execution_time
            stats['total_calls'] += 1
            stats['total_memory'] += profile_result.memory_usage
            stats['max_time'] = max(stats['max_time'], profile_result.execution_time)
            stats['min_time'] = min(stats['min_time'], profile_result.execution_time)
            
            return result
            
        except Exception as e:
            profiler.disable()
            raise e
    
    @contextmanager
    def profile_context(self, context_name: str):
        """
        Context manager for profiling a block of code.
        
        Args:
            context_name: Name for this profiling context
        """
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield
        finally:
            profiler.disable()
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Extract stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats()
            stats_output = s.getvalue()
            
            # Create profile result
            profile_result = ProfileResult(
                function_name=context_name,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_percent=0,  # Not available for context
                call_count=1,
                stats={
                    'detailed_stats': stats_output,
                    'start_time': start_time,
                    'end_time': end_time
                }
            )
            
            self.results.append(profile_result)
    
    def get_bottlenecks(self, top_n: int = 10) -> List[BottleneckInfo]:
        """
        Identify performance bottlenecks.
        
        Args:
            top_n: Number of top bottlenecks to return
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        total_time = sum(result.execution_time for result in self.results)
        
        # Group by function name
        function_times = {}
        function_calls = {}
        
        for result in self.results:
            func_name = result.function_name
            if func_name not in function_times:
                function_times[func_name] = 0
                function_calls[func_name] = 0
            
            function_times[func_name] += result.execution_time
            function_calls[func_name] += result.call_count
        
        # Sort by total time
        sorted_functions = sorted(
            function_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        for func_name, func_time in sorted_functions:
            percentage = (func_time / total_time) * 100 if total_time > 0 else 0
            call_count = function_calls[func_name]
            time_per_call = func_time / call_count if call_count > 0 else 0
            
            description = f"Accounts for {percentage:.1f}% of total execution time"
            if time_per_call > 1.0:
                description += f", slow per-call performance ({time_per_call:.2f}s/call)"
            if call_count > 100:
                description += f", called frequently ({call_count} times)"
            
            bottleneck = BottleneckInfo(
                function_name=func_name,
                total_time=func_time,
                percentage_of_total=percentage,
                call_count=call_count,
                time_per_call=time_per_call,
                description=description
            )
            
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.results:
            return {'message': 'No profiling data available'}
        
        total_time = sum(result.execution_time for result in self.results)
        total_memory = sum(result.memory_usage for result in self.results)
        
        return {
            'total_functions_profiled': len(set(r.function_name for r in self.results)),
            'total_execution_time': total_time,
            'total_memory_usage': total_memory,
            'average_execution_time': total_time / len(self.results),
            'slowest_function': max(self.results, key=lambda x: x.execution_time).function_name,
            'most_memory_intensive': max(self.results, key=lambda x: x.memory_usage).function_name,
            'cumulative_stats': self.cumulative_stats
        }
    
    def export_results(self, output_path: str):
        """
        Export profiling results to file.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            # Export as JSON
            data = {
                'profiler_name': self.name,
                'summary': self.get_summary(),
                'bottlenecks': [
                    {
                        'function_name': b.function_name,
                        'total_time': b.total_time,
                        'percentage_of_total': b.percentage_of_total,
                        'call_count': b.call_count,
                        'time_per_call': b.time_per_call,
                        'description': b.description
                    }
                    for b in self.get_bottlenecks()
                ],
                'results': [
                    {
                        'function_name': r.function_name,
                        'execution_time': r.execution_time,
                        'memory_usage': r.memory_usage,
                        'cpu_percent': r.cpu_percent,
                        'call_count': r.call_count
                    }
                    for r in self.results
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif output_path.suffix == '.csv':
            # Export as CSV
            df_data = []
            for result in self.results:
                df_data.append({
                    'function_name': result.function_name,
                    'execution_time': result.execution_time,
                    'memory_usage': result.memory_usage,
                    'cpu_percent': result.cpu_percent,
                    'call_count': result.call_count
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


# Alias for backward compatibility
PerformanceProfiler = RuntimeProfiler


def profile_function(func: Callable) -> Callable:
    """
    Profile a labeling function.
    
    This is a convenience wrapper that creates a RuntimeProfiler instance
    and profiles the given labeling function.
    
    Args:
        func: The labeling function to profile
        
    Returns:
        Wrapped labeling function with profiling
    """
    profiler = RuntimeProfiler("labeling_function")
    
    return profiler.profile_function(func)
    

def profile_lf_application(
    labeling_functions: List[LabelingFunction],
    examples: List[Example],
    profiler: Optional[RuntimeProfiler] = None
) -> Tuple[LFOutput, RuntimeProfiler]:
    """
    Profile the application of labeling functions to examples.
    
    Args:
        labeling_functions: List of labeling functions
        examples: List of examples to label
        profiler: Optional existing profiler to use
        
    Returns:
        Tuple of (LF output, profiler with results)
    """
    if profiler is None:
        profiler = RuntimeProfiler("lf_application")
    
    from ..lf import apply_lfs
    
    # Profile the LF application
    with profiler.profile_context("apply_lfs"):
        lf_output = apply_lfs(labeling_functions, examples)
    
    return lf_output, profiler


def benchmark_performance(
    func: Callable,
    *args,
    n_runs: int = 5,
    warmup_runs: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark the performance of a function.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Benchmark runs
    times = []
    memory_usage = []
    
    for _ in range(n_runs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        times.append(end_time - start_time)
        memory_usage.append(end_memory - start_memory)
    
    return {
        'function_name': func.__name__,
        'n_runs': n_runs,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory': np.mean(memory_usage),
        'std_memory': np.std(memory_usage),
        'times': times,
        'memory_usage': memory_usage
    }


# Additional aliases for backward compatibility

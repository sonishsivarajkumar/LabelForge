"""
Publication utilities for academic research with LabelForge.

This module provides tools for generating publication-ready figures,
tables, and result summaries for academic papers and reports.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .benchmarks import BenchmarkResult
from .evaluation import CrossValidationResult, StatisticalTestResult


@dataclass
class PublicationFigure:
    """Container for publication-ready figure."""
    figure: Any  # matplotlib.figure.Figure
    caption: str
    filename: str
    width: float = 6.0  # inches
    height: float = 4.0  # inches
    dpi: int = 300


class AcademicPlotter:
    """
    Create publication-ready plots for academic papers.
    
    Follows academic plotting conventions and provides consistent
    styling across different types of visualizations.
    """
    
    def __init__(
        self,
        style: str = 'academic',
        color_palette: str = 'colorblind',
        font_size: int = 10
    ):
        """
        Initialize academic plotter.
        
        Args:
            style: Plotting style ('academic', 'presentation', 'poster')
            color_palette: Color palette name
            font_size: Base font size
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting")
        
        self.style = style
        self.color_palette = color_palette
        self.font_size = font_size
        
        # Set up academic style
        self._setup_style()
    
    def _setup_style(self):
        """Set up matplotlib style for academic publications."""
        plt.style.use('seaborn-v0_8-whitegrid' if HAS_SEABORN else 'default')
        
        # Academic styling parameters
        plt.rcParams.update({
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4,
            'font.family': 'serif',
            'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'dejavuserif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'patch.linewidth': 0.5,
            'axes.linewidth': 1.0,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def plot_method_comparison(
        self,
        results: List[BenchmarkResult],
        metric: str = 'accuracy',
        group_by: str = 'dataset',
        save_path: Optional[str] = None
    ) -> PublicationFigure:
        """
        Create method comparison plot.
        
        Args:
            results: List of benchmark results
            metric: Metric to plot
            group_by: How to group results ('dataset' or 'method')
            save_path: Path to save figure
            
        Returns:
            Publication figure object
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'Method': r.method_name,
                'Dataset': r.dataset_name,
                'Score': getattr(r, metric),
                'Training Time': r.training_time
            }
            for r in results if getattr(r, metric) is not None
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if group_by == 'dataset':
            # Group by dataset, compare methods
            datasets = df['Dataset'].unique()
            x_pos = np.arange(len(datasets))
            width = 0.8 / len(df['Method'].unique())
            
            for i, method in enumerate(df['Method'].unique()):
                method_data = df[df['Method'] == method]
                scores = [
                    method_data[method_data['Dataset'] == dataset]['Score'].iloc[0]
                    if len(method_data[method_data['Dataset'] == dataset]) > 0
                    else 0
                    for dataset in datasets
                ]
                
                ax.bar(x_pos + i * width, scores, width, 
                      label=method, alpha=0.8)
            
            ax.set_xlabel('Dataset')
            ax.set_xticks(x_pos + width * (len(df['Method'].unique()) - 1) / 2)
            ax.set_xticklabels(datasets, rotation=45)
            
        else:  # group_by == 'method'
            # Group by method, compare datasets
            methods = df['Method'].unique()
            x_pos = np.arange(len(methods))
            width = 0.8 / len(df['Dataset'].unique())
            
            for i, dataset in enumerate(df['Dataset'].unique()):
                dataset_data = df[df['Dataset'] == dataset]
                scores = [
                    dataset_data[dataset_data['Method'] == method]['Score'].iloc[0]
                    if len(dataset_data[dataset_data['Method'] == method]) > 0
                    else 0
                    for method in methods
                ]
                
                ax.bar(x_pos + i * width, scores, width,
                      label=dataset, alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_xticks(x_pos + width * (len(df['Dataset'].unique()) - 1) / 2)
            ax.set_xticklabels(methods, rotation=45)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Method Comparison - {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        caption = (f"Comparison of weak supervision methods on {metric}. "
                  f"Results averaged across multiple runs with error bars "
                  f"showing standard deviation.")
        
        return PublicationFigure(
            figure=fig,
            caption=caption,
            filename=f"method_comparison_{metric}.pdf"
        )
    
    def plot_performance_vs_time(
        self,
        results: List[BenchmarkResult],
        performance_metric: str = 'accuracy',
        time_metric: str = 'training_time',
        save_path: Optional[str] = None
    ) -> PublicationFigure:
        """
        Create performance vs. time tradeoff plot.
        
        Args:
            results: List of benchmark results
            performance_metric: Performance metric for y-axis
            time_metric: Time metric for x-axis
            save_path: Path to save figure
            
        Returns:
            Publication figure object
        """
        # Prepare data
        data = []
        for r in results:
            perf_value = getattr(r, performance_metric)
            time_value = getattr(r, time_metric)
            if perf_value is not None and time_value is not None:
                data.append({
                    'Method': r.method_name,
                    'Dataset': r.dataset_name,
                    'Performance': perf_value,
                    'Time': time_value
                })
        
        if not data:
            raise ValueError("No valid data for performance vs time plot")
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot with different markers for each method
        methods = df['Method'].unique()
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            marker = markers[i % len(markers)]
            
            ax.scatter(
                method_data['Time'],
                method_data['Performance'],
                label=method,
                marker=marker,
                s=60,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f'{time_metric.replace("_", " ").title()} (seconds)')
        ax.set_ylabel(performance_metric.replace('_', ' ').title())
        ax.set_title('Performance vs. Training Time Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add Pareto frontier if multiple methods
        if len(methods) > 1:
            # Find Pareto-optimal points (maximize performance, minimize time)
            pareto_points = []
            for _, row in df.iterrows():
                is_pareto = True
                for _, other_row in df.iterrows():
                    if (other_row['Performance'] >= row['Performance'] and 
                        other_row['Time'] <= row['Time'] and
                        (other_row['Performance'] > row['Performance'] or 
                         other_row['Time'] < row['Time'])):
                        is_pareto = False
                        break
                
                if is_pareto:
                    pareto_points.append((row['Time'], row['Performance']))
            
            if len(pareto_points) > 1:
                pareto_points.sort()
                pareto_x, pareto_y = zip(*pareto_points)
                ax.plot(pareto_x, pareto_y, 'r--', alpha=0.7, 
                       linewidth=2, label='Pareto Frontier')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        caption = (f"Performance vs. training time trade-off for different "
                  f"weak supervision methods. Points closer to the top-left "
                  f"represent better trade-offs.")
        
        return PublicationFigure(
            figure=fig,
            caption=caption,
            filename=f"performance_vs_time_{performance_metric}.pdf"
        )
    
    def plot_convergence_analysis(
        self,
        convergence_data: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> PublicationFigure:
        """
        Create convergence analysis plot.
        
        Args:
            convergence_data: Dictionary mapping method names to convergence curves
            save_path: Path to save figure
            
        Returns:
            Publication figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for method_name, curve in convergence_data.items():
            iterations = np.arange(1, len(curve) + 1)
            ax.plot(iterations, curve, label=method_name, 
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
        ax.set_title('Convergence Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        caption = ("Convergence analysis showing log-likelihood curves "
                  "for different weak supervision methods during training.")
        
        return PublicationFigure(
            figure=fig,
            caption=caption,
            filename="convergence_analysis.pdf"
        )


class LaTeXExporter:
    """
    Export results to LaTeX format for academic papers.
    
    Generates publication-ready tables and figures in LaTeX format.
    """
    
    def __init__(self, precision: int = 3):
        """
        Initialize LaTeX exporter.
        
        Args:
            precision: Number of decimal places for numeric values
        """
        self.precision = precision
    
    def results_table(
        self,
        results: List[BenchmarkResult],
        metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall'],
        caption: str = "Experimental results comparison",
        label: str = "tab:results"
    ) -> str:
        """
        Generate LaTeX table from benchmark results.
        
        Args:
            results: List of benchmark results
            metrics: Metrics to include in table
            caption: Table caption
            label: Table label for referencing
            
        Returns:
            LaTeX table string
        """
        # Prepare data
        df = pd.DataFrame([
            {
                'Method': r.method_name,
                'Dataset': r.dataset_name,
                **{metric: getattr(r, metric) for metric in metrics 
                   if getattr(r, metric) is not None}
            }
            for r in results
        ])
        
        # Group by method and dataset
        pivot_tables = {}
        for metric in metrics:
            if metric in df.columns:
                pivot_tables[metric] = df.pivot_table(
                    values=metric,
                    index='Method',
                    columns='Dataset',
                    aggfunc='mean'
                )
        
        # Generate LaTeX
        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{{caption}}}\n"
        latex_str += f"\\label{{{label}}}\n"
        
        # Create combined table with all metrics
        if pivot_tables:
            first_metric = list(pivot_tables.keys())[0]
            methods = pivot_tables[first_metric].index.tolist()
            datasets = pivot_tables[first_metric].columns.tolist()
            
            # Table header
            n_cols = len(datasets) * len(metrics) + 1
            latex_str += f"\\begin{{tabular}}{{l{'c' * (n_cols - 1)}}}\n"
            latex_str += "\\toprule\n"
            
            # Multi-level header
            header1 = "Method"
            header2 = ""
            for dataset in datasets:
                header1 += f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{dataset}}}"
                header2 += " & " + " & ".join(metrics)
            
            latex_str += header1 + " \\\\\n"
            latex_str += "\\cmidrule(lr){2-" + str(n_cols) + "}\n"
            latex_str += header2 + " \\\\\n"
            latex_str += "\\midrule\n"
            
            # Table rows
            for method in methods:
                row = method
                for dataset in datasets:
                    for metric in metrics:
                        if (metric in pivot_tables and 
                            method in pivot_tables[metric].index and
                            dataset in pivot_tables[metric].columns):
                            value = pivot_tables[metric].loc[method, dataset]
                            if pd.notna(value):
                                row += f" & {value:.{self.precision}f}"
                            else:
                                row += " & --"
                        else:
                            row += " & --"
                
                latex_str += row + " \\\\\n"
            
            latex_str += "\\bottomrule\n"
            latex_str += "\\end{tabular}\n"
        
        latex_str += "\\end{table}\n"
        
        return latex_str
    
    def statistical_significance_table(
        self,
        test_results: Dict[str, StatisticalTestResult],
        caption: str = "Statistical significance tests",
        label: str = "tab:significance"
    ) -> str:
        """
        Generate LaTeX table for statistical test results.
        
        Args:
            test_results: Dictionary of test results
            caption: Table caption
            label: Table label
            
        Returns:
            LaTeX table string
        """
        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{{caption}}}\n"
        latex_str += f"\\label{{{label}}}\n"
        latex_str += "\\begin{tabular}{lcccc}\n"
        latex_str += "\\toprule\n"
        latex_str += "Comparison & Test & Statistic & p-value & Significant \\\\\n"
        latex_str += "\\midrule\n"
        
        for comparison, result in test_results.items():
            significance = "Yes" if result.is_significant else "No"
            latex_str += (f"{comparison.replace('_', ' vs. ')} & "
                         f"{result.test_name} & "
                         f"{result.statistic:.{self.precision}f} & "
                         f"{result.p_value:.{self.precision}f} & "
                         f"{significance} \\\\\n")
        
        latex_str += "\\bottomrule\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\end{table}\n"
        
        return latex_str
    
    def figure_reference(
        self,
        figure: PublicationFigure,
        placement: str = "htbp"
    ) -> str:
        """
        Generate LaTeX figure environment.
        
        Args:
            figure: Publication figure object
            placement: Figure placement options
            
        Returns:
            LaTeX figure string
        """
        latex_str = f"\\begin{{figure}}[{placement}]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\includegraphics[width=0.8\\textwidth]{{{figure.filename}}}\n"
        latex_str += f"\\caption{{{figure.caption}}}\n"
        latex_str += f"\\label{{fig:{figure.filename.replace('.', '_')}}}\n"
        latex_str += "\\end{figure}\n"
        
        return latex_str


class ResultSummarizer:
    """
    Generate comprehensive result summaries for publications.
    
    Creates structured summaries of experimental results with
    statistical analysis and interpretation.
    """
    
    def __init__(self):
        """Initialize result summarizer."""
        pass
    
    def comprehensive_summary(
        self,
        benchmark_results: List[BenchmarkResult],
        cv_results: Dict[str, CrossValidationResult],
        statistical_tests: Dict[str, StatisticalTestResult],
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary of experimental results.
        
        Args:
            benchmark_results: Benchmark evaluation results
            cv_results: Cross-validation results
            statistical_tests: Statistical test results
            dataset_info: Information about datasets used
            
        Returns:
            Comprehensive summary dictionary
        """
        summary = {
            'experiment_overview': self._generate_overview(
                benchmark_results, dataset_info
            ),
            'performance_analysis': self._analyze_performance(
                benchmark_results, cv_results
            ),
            'statistical_analysis': self._analyze_significance(
                statistical_tests
            ),
            'recommendations': self._generate_recommendations(
                benchmark_results, statistical_tests
            )
        }
        
        return summary
    
    def _generate_overview(
        self,
        results: List[BenchmarkResult],
        dataset_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate experiment overview."""
        overview = {
            'n_methods': len(set(r.method_name for r in results)),
            'n_datasets': len(set(r.dataset_name for r in results)),
            'methods_tested': list(set(r.method_name for r in results)),
            'datasets_used': list(set(r.dataset_name for r in results)),
            'metrics_evaluated': ['accuracy', 'f1_score', 'precision', 'recall']
        }
        
        if dataset_info:
            overview['dataset_details'] = dataset_info
        
        return overview
    
    def _analyze_performance(
        self,
        benchmark_results: List[BenchmarkResult],
        cv_results: Dict[str, CrossValidationResult]
    ) -> Dict[str, Any]:
        """Analyze performance across methods and datasets."""
        # Best performing methods per metric
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        best_methods = {}
        
        for metric in metrics:
            metric_results = [r for r in benchmark_results 
                            if getattr(r, metric) is not None]
            if metric_results:
                best_result = max(metric_results, 
                                key=lambda x: getattr(x, metric))
                best_methods[metric] = {
                    'method': best_result.method_name,
                    'dataset': best_result.dataset_name,
                    'score': getattr(best_result, metric)
                }
        
        # Performance consistency analysis
        method_consistency = {}
        for method in set(r.method_name for r in benchmark_results):
            method_results = [r for r in benchmark_results 
                            if r.method_name == method]
            if method_results:
                accuracies = [r.accuracy for r in method_results 
                            if r.accuracy is not None]
                if accuracies:
                    method_consistency[method] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'min_accuracy': np.min(accuracies),
                        'max_accuracy': np.max(accuracies)
                    }
        
        return {
            'best_methods': best_methods,
            'consistency_analysis': method_consistency,
            'cv_results_summary': {
                name: {
                    'mean_score': result.mean_score,
                    'std_score': result.std_score,
                    'confidence_interval': result.confidence_interval
                }
                for name, result in cv_results.items()
            }
        }
    
    def _analyze_significance(
        self,
        statistical_tests: Dict[str, StatisticalTestResult]
    ) -> Dict[str, Any]:
        """Analyze statistical significance of results."""
        significant_tests = {
            name: result for name, result in statistical_tests.items()
            if result.is_significant
        }
        
        non_significant_tests = {
            name: result for name, result in statistical_tests.items()
            if not result.is_significant
        }
        
        return {
            'n_total_tests': len(statistical_tests),
            'n_significant': len(significant_tests),
            'n_non_significant': len(non_significant_tests),
            'significant_comparisons': list(significant_tests.keys()),
            'non_significant_comparisons': list(non_significant_tests.keys()),
            'min_p_value': min(r.p_value for r in statistical_tests.values()),
            'max_p_value': max(r.p_value for r in statistical_tests.values())
        }
    
    def _generate_recommendations(
        self,
        benchmark_results: List[BenchmarkResult],
        statistical_tests: Dict[str, StatisticalTestResult]
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Find consistently best method
        method_scores = {}
        for result in benchmark_results:
            if result.accuracy is not None:
                if result.method_name not in method_scores:
                    method_scores[result.method_name] = []
                method_scores[result.method_name].append(result.accuracy)
        
        if method_scores:
            avg_scores = {
                method: np.mean(scores) 
                for method, scores in method_scores.items()
            }
            best_method = max(avg_scores.keys(), key=lambda x: avg_scores[x])
            
            recommendations.append(
                f"Based on average performance across datasets, "
                f"{best_method} appears to be the most effective method."
            )
        
        # Check for statistical significance
        significant_count = sum(1 for test in statistical_tests.values() 
                               if test.is_significant)
        total_tests = len(statistical_tests)
        
        if significant_count > total_tests * 0.5:
            recommendations.append(
                "Multiple statistically significant differences were found, "
                "indicating meaningful performance variations between methods."
            )
        else:
            recommendations.append(
                "Few statistically significant differences were found, "
                "suggesting similar performance across methods."
            )
        
        return recommendations


class ExperimentTracker:
    """
    Track and manage experiment metadata for research publications.
    
    Provides tools for organizing experiments, tracking results,
    and generating summaries for academic papers.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to store experiment data
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.experiments = {}
        self.metadata = {
            'name': experiment_name,
            'created_at': pd.Timestamp.now().isoformat(),
            'experiments': []
        }
    
    def start_experiment(
        self,
        experiment_id: str,
        description: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_id: Unique experiment identifier
            description: Experiment description
            config: Experiment configuration
            
        Returns:
            Experiment run ID
        """
        run_id = f"{experiment_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_info = {
            'experiment_id': experiment_id,
            'run_id': run_id,
            'description': description,
            'config': config,
            'started_at': pd.Timestamp.now().isoformat(),
            'status': 'running',
            'results': {},
            'artifacts': []
        }
        
        self.experiments[run_id] = experiment_info
        self.metadata['experiments'].append(run_id)
        
        # Save metadata
        self._save_metadata()
        
        return run_id
    
    def log_result(
        self,
        run_id: str,
        metric_name: str,
        value: Union[float, int, str, Dict, List],
        step: Optional[int] = None
    ):
        """
        Log a result for an experiment run.
        
        Args:
            run_id: Experiment run ID
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        if run_id not in self.experiments:
            raise ValueError(f"Experiment run {run_id} not found")
        
        if 'results' not in self.experiments[run_id]:
            self.experiments[run_id]['results'] = {}
        
        if metric_name not in self.experiments[run_id]['results']:
            self.experiments[run_id]['results'][metric_name] = []
        
        result_entry = {
            'value': value,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if step is not None:
            result_entry['step'] = step
        
        self.experiments[run_id]['results'][metric_name].append(result_entry)
        
        # Save updated metadata
        self._save_metadata()
    
    def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_type: str = 'file',
        description: str = ""
    ):
        """
        Log an artifact for an experiment run.
        
        Args:
            run_id: Experiment run ID
            artifact_path: Path to the artifact
            artifact_type: Type of artifact ('file', 'figure', 'model', etc.)
            description: Artifact description
        """
        if run_id not in self.experiments:
            raise ValueError(f"Experiment run {run_id} not found")
        
        artifact_info = {
            'path': str(artifact_path),
            'type': artifact_type,
            'description': description,
            'logged_at': pd.Timestamp.now().isoformat()
        }
        
        self.experiments[run_id]['artifacts'].append(artifact_info)
        
        # Save updated metadata
        self._save_metadata()
    
    def finish_experiment(self, run_id: str, status: str = 'completed'):
        """
        Mark an experiment as finished.
        
        Args:
            run_id: Experiment run ID
            status: Final status ('completed', 'failed', 'stopped')
        """
        if run_id not in self.experiments:
            raise ValueError(f"Experiment run {run_id} not found")
        
        self.experiments[run_id]['status'] = status
        self.experiments[run_id]['finished_at'] = pd.Timestamp.now().isoformat()
        
        # Calculate duration
        started_at = pd.Timestamp(self.experiments[run_id]['started_at'])
        finished_at = pd.Timestamp(self.experiments[run_id]['finished_at'])
        duration = (finished_at - started_at).total_seconds()
        self.experiments[run_id]['duration_seconds'] = duration
        
        # Save updated metadata
        self._save_metadata()
    
    def get_experiment_results(self, run_id: str) -> Dict[str, Any]:
        """Get results for a specific experiment run."""
        if run_id not in self.experiments:
            raise ValueError(f"Experiment run {run_id} not found")
        
        return self.experiments[run_id]['results']
    
    def compare_experiments(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare results across multiple experiment runs.
        
        Args:
            run_ids: List of experiment run IDs to compare
            metrics: Optional list of metrics to include
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for run_id in run_ids:
            if run_id not in self.experiments:
                continue
            
            exp = self.experiments[run_id]
            row = {
                'run_id': run_id,
                'experiment_id': exp['experiment_id'],
                'description': exp['description'],
                'status': exp['status'],
                'duration_seconds': exp.get('duration_seconds', None)
            }
            
            # Add metric values (use latest value for each metric)
            results = exp.get('results', {})
            for metric_name, metric_history in results.items():
                if metrics is None or metric_name in metrics:
                    if metric_history:
                        row[metric_name] = metric_history[-1]['value']
                    else:
                        row[metric_name] = None
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def export_summary(self, output_path: Optional[str] = None) -> str:
        """
        Export experiment summary to JSON.
        
        Args:
            output_path: Optional path to save summary
            
        Returns:
            Summary JSON string
        """
        summary = {
            'metadata': self.metadata,
            'experiments': self.experiments,
            'summary_stats': {
                'total_experiments': len(self.experiments),
                'completed_experiments': len([
                    exp for exp in self.experiments.values()
                    if exp['status'] == 'completed'
                ]),
                'failed_experiments': len([
                    exp for exp in self.experiments.values()
                    if exp['status'] == 'failed'
                ])
            }
        }
        
        summary_json = json.dumps(summary, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(summary_json)
        
        return summary_json
    
    def _save_metadata(self):
        """Save experiment metadata to disk."""
        metadata_path = self.experiment_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Also save detailed experiment data
        experiments_path = self.experiment_dir / 'experiments.json'
        with open(experiments_path, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)


class CitationFormatter:
    """
    Format citations and references for academic publications.
    """
    
    def __init__(self, style: str = 'apa'):
        """
        Initialize citation formatter.
        
        Args:
            style: Citation style ('apa', 'mla', 'chicago')
        """
        self.style = style
    
    def format_software_citation(
        self,
        name: str = "LabelForge",
        version: str = "0.2.0",
        authors: List[str] = None,
        year: int = 2025,
        url: str = "https://github.com/labelforge/labelforge"
    ) -> str:
        """
        Format a software citation.
        
        Args:
            name: Software name
            version: Software version
            authors: List of authors
            year: Publication year
            url: Software URL
            
        Returns:
            Formatted citation
        """
        if authors is None:
            authors = ["LabelForge Contributors"]
        
        if self.style == 'apa':
            author_str = ", ".join(authors)
            return f"{author_str} ({year}). {name} (Version {version}) [Computer software]. {url}"
        
        elif self.style == 'bibtex':
            author_str = " and ".join(authors)
            return f"""@software{{{name.lower()}{year},
  author = {{{author_str}}},
  title = {{{name}}},
  version = {{{version}}},
  year = {{{year}}},
  url = {{{url}}}
}}"""
        
        else:
            # Default format
            return f"{name} v{version} ({year}) - {url}"
    
    def format_dataset_citation(
        self,
        dataset_name: str,
        authors: List[str],
        year: int,
        venue: str = "",
        url: str = ""
    ) -> str:
        """Format a dataset citation."""
        author_str = ", ".join(authors)
        
        if self.style == 'apa':
            citation = f"{author_str} ({year}). {dataset_name}"
            if venue:
                citation += f". {venue}"
            if url:
                citation += f". {url}"
            return citation
        
        else:
            citation = f"{dataset_name} - {author_str} ({year})"
            if venue:
                citation += f" - {venue}"
            return citation


class TableGenerator:
    """
    Generate publication-ready tables.
    """
    
    def __init__(self, format: str = 'latex'):
        """
        Initialize table generator.
        
        Args:
            format: Output format ('latex', 'html', 'markdown')
        """
        self.format = format
    
    def create_results_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        caption: str = "Experimental Results"
    ) -> str:
        """
        Create a results comparison table.
        
        Args:
            results: Dictionary mapping method names to metrics
            metrics: List of metrics to include
            caption: Table caption
            
        Returns:
            Formatted table string
        """
        if metrics is None:
            # Get all metrics from first result
            first_result = next(iter(results.values()))
            metrics = list(first_result.keys())
        
        if self.format == 'latex':
            return self._create_latex_table(results, metrics, caption)
        elif self.format == 'html':
            return self._create_html_table(results, metrics, caption)
        elif self.format == 'markdown':
            return self._create_markdown_table(results, metrics, caption)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _create_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        caption: str
    ) -> str:
        """Create LaTeX table."""
        n_cols = len(metrics) + 1  # +1 for method name
        col_spec = 'l' + 'c' * len(metrics)
        
        table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
Method & {' & '.join(metrics)} \\\\
\\midrule
"""
        
        for method, method_results in results.items():
            row_values = [method]
            for metric in metrics:
                value = method_results.get(metric, 0.0)
                if isinstance(value, float):
                    row_values.append(f"{value:.3f}")
                else:
                    row_values.append(str(value))
            
            table += " & ".join(row_values) + " \\\\\n"
        
        table += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return table
    
    def _create_html_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        caption: str
    ) -> str:
        """Create HTML table."""
        table = f'<table>\n<caption>{caption}</caption>\n'
        
        # Header
        table += '<thead>\n<tr>\n<th>Method</th>\n'
        for metric in metrics:
            table += f'<th>{metric}</th>\n'
        table += '</tr>\n</thead>\n'
        
        # Body
        table += '<tbody>\n'
        for method, method_results in results.items():
            table += f'<tr>\n<td>{method}</td>\n'
            for metric in metrics:
                value = method_results.get(metric, 0.0)
                if isinstance(value, float):
                    table += f'<td>{value:.3f}</td>\n'
                else:
                    table += f'<td>{value}</td>\n'
            table += '</tr>\n'
        table += '</tbody>\n</table>'
        
        return table
    
    def _create_markdown_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        caption: str
    ) -> str:
        """Create Markdown table."""
        table = f"**{caption}**\n\n"
        
        # Header
        header = "| Method |"
        separator = "|--------|"
        
        for metric in metrics:
            header += f" {metric} |"
            separator += "--------|"
        
        table += header + "\n" + separator + "\n"
        
        # Rows
        for method, method_results in results.items():
            row = f"| {method} |"
            for metric in metrics:
                value = method_results.get(metric, 0.0)
                if isinstance(value, float):
                    row += f" {value:.3f} |"
                else:
                    row += f" {value} |"
            table += row + "\n"
        
        return table

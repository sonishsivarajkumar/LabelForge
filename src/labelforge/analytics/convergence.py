"""
Convergence diagnostics for EM algorithm and model training.

This module provides tools for monitoring and analyzing the convergence
of the EM algorithm used in label model training.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib not available. Plotting functionality disabled.")

from ..label_model import LabelModel
from ..types import LFOutput


class ConvergenceTracker:
    """
    Track convergence metrics during model training.
    
    Provides tools for monitoring parameter changes, likelihood evolution,
    and convergence criteria during EM algorithm iterations.
    """
    
    def __init__(self):
        """Initialize convergence tracker."""
        self.history = {
            'iteration': [],
            'log_likelihood': [],
            'parameter_changes': [],
            'convergence_criteria': []
        }
        self.is_tracking = False
    
    def start_tracking(self) -> None:
        """Start tracking convergence metrics."""
        self.history = {
            'iteration': [],
            'log_likelihood': [],
            'parameter_changes': [],
            'convergence_criteria': []
        }
        self.is_tracking = True
    
    def stop_tracking(self) -> None:
        """Stop tracking convergence metrics."""
        self.is_tracking = False
    
    def record_iteration(
        self, 
        iteration: int,
        log_likelihood: float,
        parameter_change: float,
        convergence_criterion: float
    ) -> None:
        """
        Record metrics for a single iteration.
        
        Args:
            iteration: Current iteration number
            log_likelihood: Current log likelihood
            parameter_change: Change in parameters from previous iteration
            convergence_criterion: Current convergence criterion value
        """
        if self.is_tracking:
            self.history['iteration'].append(iteration)
            self.history['log_likelihood'].append(log_likelihood)
            self.history['parameter_changes'].append(parameter_change)
            self.history['convergence_criteria'].append(convergence_criterion)
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """
        Get summary of convergence behavior.
        
        Returns:
            Dictionary with convergence statistics
        """
        if not self.history['iteration']:
            return {'error': 'No tracking data available'}
        
        return {
            'total_iterations': len(self.history['iteration']),
            'final_log_likelihood': self.history['log_likelihood'][-1],
            'likelihood_improvement': (
                self.history['log_likelihood'][-1] - self.history['log_likelihood'][0]
                if len(self.history['log_likelihood']) > 1 else 0
            ),
            'final_parameter_change': self.history['parameter_changes'][-1],
            'converged': self.history['convergence_criteria'][-1] < 1e-6,
            'convergence_rate': self._estimate_convergence_rate()
        }
    
    def _estimate_convergence_rate(self) -> float:
        """Estimate the convergence rate from parameter changes."""
        changes = np.array(self.history['parameter_changes'])
        if len(changes) < 3:
            return np.nan
        
        # Fit exponential decay to parameter changes
        log_changes = np.log(changes[changes > 0])
        if len(log_changes) < 2:
            return np.nan
        
        # Simple linear regression on log scale
        x = np.arange(len(log_changes))
        slope = np.polyfit(x, log_changes, 1)[0]
        return -slope  # Convergence rate (positive value)
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """
        Plot convergence diagnostics.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        if not self.history['iteration']:
            raise ValueError("No tracking data available for plotting")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        iterations = self.history['iteration']
        
        # Log likelihood evolution
        ax1.plot(iterations, self.history['log_likelihood'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Log Likelihood')
        ax1.set_title('Log Likelihood Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Parameter changes
        ax2.semilogy(iterations, self.history['parameter_changes'], 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Change (log scale)')
        ax2.set_title('Parameter Changes')
        ax2.grid(True, alpha=0.3)
        
        # Convergence criteria
        ax3.semilogy(iterations, self.history['convergence_criteria'], 'g-', linewidth=2)
        ax3.axhline(y=1e-6, color='k', linestyle='--', alpha=0.5, label='Convergence threshold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Convergence Criterion (log scale)')
        ax3.set_title('Convergence Criteria')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Likelihood improvement rate
        if len(self.history['log_likelihood']) > 1:
            likelihood_diffs = np.diff(self.history['log_likelihood'])
            ax4.plot(iterations[1:], likelihood_diffs, 'm-', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Likelihood Improvement')
            ax4.set_title('Per-Iteration Likelihood Improvement')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class EMDiagnostics:
    """
    Advanced diagnostics for EM algorithm behavior.
    
    Provides detailed analysis of EM algorithm performance including
    parameter trajectory analysis and convergence diagnostics.
    """
    
    def __init__(self):
        """Initialize EM diagnostics."""
        self.parameter_history = []
        self.step_sizes = []
        self.gradient_norms = []
    
    def analyze_parameter_trajectory(
        self, 
        parameter_history: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Analyze the trajectory of parameters during training.
        
        Args:
            parameter_history: List of parameter dictionaries from each iteration
            
        Returns:
            Dictionary with trajectory analysis
        """
        if len(parameter_history) < 2:
            return {'error': 'Need at least 2 iterations for trajectory analysis'}
        
        # Analyze parameter stability
        stability_metrics = self._calculate_parameter_stability(parameter_history)
        
        # Analyze step sizes
        step_analysis = self._analyze_step_sizes(parameter_history)
        
        # Detect oscillations
        oscillation_analysis = self._detect_oscillations(parameter_history)
        
        return {
            'stability': stability_metrics,
            'step_sizes': step_analysis,
            'oscillations': oscillation_analysis,
            'total_iterations': len(parameter_history)
        }
    
    def _calculate_parameter_stability(
        self, 
        parameter_history: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Calculate stability metrics for parameters."""
        if len(parameter_history) < 3:
            return {'error': 'Need at least 3 iterations for stability analysis'}
        
        # Calculate variance in parameter changes
        param_names = parameter_history[0].keys()
        stability = {}
        
        for param_name in param_names:
            param_values = [params[param_name] for params in parameter_history]
            param_changes = [
                np.linalg.norm(param_values[i] - param_values[i-1])
                for i in range(1, len(param_values))
            ]
            
            stability[param_name] = {
                'mean_change': np.mean(param_changes),
                'std_change': np.std(param_changes),
                'final_change': param_changes[-1],
                'stability_ratio': np.std(param_changes) / (np.mean(param_changes) + 1e-8)
            }
        
        return stability
    
    def _analyze_step_sizes(
        self, 
        parameter_history: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Analyze step size patterns during training."""
        step_sizes = []
        
        for i in range(1, len(parameter_history)):
            total_change = 0
            for param_name in parameter_history[0].keys():
                change = np.linalg.norm(
                    parameter_history[i][param_name] - parameter_history[i-1][param_name]
                )
                total_change += change
            step_sizes.append(total_change)
        
        return {
            'step_sizes': step_sizes,
            'mean_step_size': np.mean(step_sizes),
            'step_size_trend': 'decreasing' if step_sizes[-1] < step_sizes[0] else 'increasing',
            'step_size_ratio': step_sizes[-1] / step_sizes[0] if step_sizes[0] > 0 else np.inf
        }
    
    def _detect_oscillations(
        self, 
        parameter_history: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Detect oscillatory behavior in parameter updates."""
        if len(parameter_history) < 4:
            return {'oscillations_detected': False, 'reason': 'Insufficient iterations'}
        
        # Simple oscillation detection based on direction changes
        param_names = parameter_history[0].keys()
        oscillation_scores = {}
        
        for param_name in param_names:
            param_values = [params[param_name] for params in parameter_history]
            
            # Calculate direction changes
            directions = []
            for i in range(2, len(param_values)):
                prev_change = param_values[i-1] - param_values[i-2]
                curr_change = param_values[i] - param_values[i-1]
                
                # Check if direction changed (dot product negative)
                if prev_change.size == curr_change.size:
                    dot_product = np.dot(prev_change.flatten(), curr_change.flatten())
                    directions.append(dot_product < 0)
            
            oscillation_score = np.mean(directions) if directions else 0
            oscillation_scores[param_name] = oscillation_score
        
        overall_oscillation = np.mean(list(oscillation_scores.values()))
        
        return {
            'oscillations_detected': overall_oscillation > 0.5,
            'oscillation_scores': oscillation_scores,
            'overall_oscillation_score': overall_oscillation
        }
    
    def diagnose_convergence_issues(
        self, 
        convergence_tracker: ConvergenceTracker
    ) -> Dict[str, Any]:
        """
        Diagnose potential issues with convergence.
        
        Args:
            convergence_tracker: ConvergenceTracker instance with history
            
        Returns:
            Dictionary with diagnostic information
        """
        if not convergence_tracker.history['iteration']:
            return {'error': 'No convergence history available'}
        
        issues = []
        recommendations = []
        
        # Check for slow convergence
        if len(convergence_tracker.history['iteration']) > 100:
            issues.append("Slow convergence: More than 100 iterations required")
            recommendations.append("Consider increasing convergence tolerance or using better initialization")
        
        # Check for stagnation
        recent_changes = convergence_tracker.history['parameter_changes'][-10:]
        if len(recent_changes) >= 10 and all(change < 1e-8 for change in recent_changes):
            issues.append("Parameter stagnation detected")
            recommendations.append("Model may have converged to local optimum")
        
        # Check for oscillation in likelihood
        if len(convergence_tracker.history['log_likelihood']) > 10:
            recent_likelihood = convergence_tracker.history['log_likelihood'][-10:]
            likelihood_diffs = np.diff(recent_likelihood)
            sign_changes = np.sum(np.diff(np.sign(likelihood_diffs)) != 0)
            
            if sign_changes > 5:
                issues.append("Likelihood oscillation detected")
                recommendations.append("Consider reducing learning rate or using more stable optimization")
        
        return {
            'issues_detected': issues,
            'recommendations': recommendations,
            'convergence_summary': convergence_tracker.get_convergence_summary()
        }

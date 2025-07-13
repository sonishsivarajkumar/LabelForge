"""
Uncertainty quantification for weak supervision models.

This module provides tools for quantifying and analyzing uncertainty in 
label model predictions, including confidence intervals, calibration analysis,
and epistemic uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from scipy import stats
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib not available. Plotting functionality disabled.")

from ..types import LFOutput
from ..label_model import LabelModel


class UncertaintyQuantifier:
    """
    Quantify uncertainty in label model predictions.
    
    Provides methods for estimating prediction uncertainty using various
    techniques including Monte Carlo dropout, bootstrap sampling, and
    confidence interval estimation.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize uncertainty quantifier.
        
        Args:
            label_model: Trained LabelModel instance
        """
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'mu')
        
    def predict_with_uncertainty(
        self, 
        lf_output: LFOutput,
        method: str = "bootstrap",
        n_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict labels with uncertainty estimates.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            method: Uncertainty estimation method ("bootstrap", "dropout", "ensemble")
            n_samples: Number of samples for uncertainty estimation
            confidence_level: Confidence level for intervals (0-1)
            
        Returns:
            Tuple of (predictions, probabilities, lower_bounds, upper_bounds)
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before uncertainty estimation")
            
        if method == "bootstrap":
            return self._bootstrap_uncertainty(lf_output, n_samples, confidence_level)
        elif method == "dropout":
            return self._dropout_uncertainty(lf_output, n_samples, confidence_level)
        elif method == "ensemble":
            return self._ensemble_uncertainty(lf_output, n_samples, confidence_level)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def _bootstrap_uncertainty(
        self, 
        lf_output: LFOutput, 
        n_samples: int, 
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate uncertainty using bootstrap sampling."""
        n_examples = lf_output.votes.shape[0]
        all_probs = []
        
        # Generate bootstrap samples
        for _ in range(n_samples):
            # Bootstrap sample indices
            indices = np.random.choice(n_examples, size=n_examples, replace=True)
            
            # Create bootstrap LF output
            bootstrap_votes = lf_output.votes[indices]
            bootstrap_lf_output = LFOutput(
                votes=bootstrap_votes,
                lf_names=lf_output.lf_names
            )
            
            # Clone and retrain model on bootstrap sample
            bootstrap_model = LabelModel(cardinality=self.label_model.cardinality)
            bootstrap_model.fit(bootstrap_lf_output, verbose=False)
            
            # Get predictions on original data
            probs = bootstrap_model.predict_proba(lf_output)
            all_probs.append(probs)
        
        # Calculate statistics
        all_probs = np.array(all_probs)  # shape: (n_samples, n_examples, n_classes)
        
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(all_probs, lower_percentile, axis=0)
        upper_bounds = np.percentile(all_probs, upper_percentile, axis=0)
        
        predictions = np.argmax(mean_probs, axis=1)
        
        return predictions, mean_probs, lower_bounds, upper_bounds
    
    def _dropout_uncertainty(
        self, 
        lf_output: LFOutput, 
        n_samples: int, 
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate uncertainty using Monte Carlo dropout (simplified version)."""
        # For now, implement a simplified version by adding noise to parameters
        # In a full implementation, this would use actual dropout layers
        
        original_params = self._get_model_parameters()
        all_probs = []
        
        for _ in range(n_samples):
            # Add small noise to model parameters to simulate dropout
            self._add_parameter_noise(scale=0.01)
            
            # Get predictions with noisy parameters
            probs = self.label_model.predict_proba(lf_output)
            all_probs.append(probs)
            
            # Restore original parameters
            self._set_model_parameters(original_params)
        
        # Calculate statistics (same as bootstrap)
        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(all_probs, lower_percentile, axis=0)
        upper_bounds = np.percentile(all_probs, upper_percentile, axis=0)
        
        predictions = np.argmax(mean_probs, axis=1)
        
        return predictions, mean_probs, lower_bounds, upper_bounds
    
    def _ensemble_uncertainty(
        self, 
        lf_output: LFOutput, 
        n_samples: int, 
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate uncertainty using ensemble of models with different initializations."""
        all_probs = []
        
        for _ in range(n_samples):
            # Create new model with different random initialization
            ensemble_model = LabelModel(
                cardinality=self.label_model.cardinality,
                seed=np.random.randint(0, 10000)
            )
            ensemble_model.fit(lf_output, verbose=False)
            
            probs = ensemble_model.predict_proba(lf_output)
            all_probs.append(probs)
        
        # Calculate statistics (same as bootstrap)
        all_probs = np.array(all_probs)
        mean_probs = np.mean(all_probs, axis=0)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(all_probs, lower_percentile, axis=0)
        upper_bounds = np.percentile(all_probs, upper_percentile, axis=0)
        
        predictions = np.argmax(mean_probs, axis=1)
        
        return predictions, mean_probs, lower_bounds, upper_bounds
    
    def _get_model_parameters(self) -> Dict[str, np.ndarray]:
        """Get current model parameters."""
        params = {}
        if hasattr(self.label_model, 'mu'):
            params['mu'] = self.label_model.mu.copy()
        if hasattr(self.label_model, 'balance'):
            params['balance'] = self.label_model.balance.copy()
        return params
    
    def _set_model_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters."""
        for name, value in params.items():
            if hasattr(self.label_model, name):
                setattr(self.label_model, name, value.copy())
    
    def _add_parameter_noise(self, scale: float = 0.01) -> None:
        """Add Gaussian noise to model parameters."""
        if hasattr(self.label_model, 'mu'):
            noise = np.random.normal(0, scale, self.label_model.mu.shape)
            self.label_model.mu += noise
        if hasattr(self.label_model, 'balance'):
            noise = np.random.normal(0, scale, self.label_model.balance.shape)
            self.label_model.balance += noise
    
    def uncertainty_summary(
        self, 
        lf_output: LFOutput, 
        method: str = "bootstrap",
        n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Generate summary statistics for prediction uncertainty.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            method: Uncertainty estimation method
            n_samples: Number of samples for estimation
            
        Returns:
            DataFrame with uncertainty statistics
        """
        predictions, probs, lower_bounds, upper_bounds = self.predict_with_uncertainty(
            lf_output, method=method, n_samples=n_samples
        )
        
        # Calculate uncertainty metrics
        max_probs = np.max(probs, axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        confidence_width = upper_bounds[np.arange(len(predictions)), predictions] - \
                          lower_bounds[np.arange(len(predictions)), predictions]
        
        summary = pd.DataFrame({
            'prediction': predictions,
            'max_probability': max_probs,
            'entropy': entropy,
            'confidence_width': confidence_width,
            'high_uncertainty': entropy > np.percentile(entropy, 75),
            'low_confidence': max_probs < 0.7
        })
        
        return summary


class CalibrationAnalyzer:
    """
    Analyze model calibration and reliability.
    
    Provides tools for assessing how well predicted probabilities match
    actual outcomes, including reliability diagrams and calibration metrics.
    """
    
    def __init__(self):
        """Initialize calibration analyzer."""
        pass
    
    def analyze_calibration(
        self, 
        probabilities: np.ndarray, 
        true_labels: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze model calibration.
        
        Args:
            probabilities: Predicted probabilities (n_examples, n_classes)
            true_labels: True labels (optional, for supervised evaluation)
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary with calibration metrics and data
        """
        max_probs = np.max(probabilities, axis=1)
        predicted_labels = np.argmax(probabilities, axis=1)
        
        results = {
            'n_bins': n_bins,
            'bin_boundaries': np.linspace(0, 1, n_bins + 1),
            'confidence_distribution': self._analyze_confidence_distribution(max_probs, n_bins)
        }
        
        if true_labels is not None:
            results.update(self._analyze_supervised_calibration(
                max_probs, predicted_labels, true_labels, n_bins
            ))
        
        return results
    
    def _analyze_confidence_distribution(
        self, 
        max_probs: np.ndarray, 
        n_bins: int
    ) -> Dict[str, np.ndarray]:
        """Analyze distribution of confidence scores."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_counts, _ = np.histogram(max_probs, bins=bin_boundaries)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        return {
            'bin_centers': bin_centers,
            'bin_counts': bin_counts,
            'bin_frequencies': bin_counts / len(max_probs)
        }
    
    def _analyze_supervised_calibration(
        self,
        max_probs: np.ndarray,
        predicted_labels: np.ndarray, 
        true_labels: np.ndarray,
        n_bins: int
    ) -> Dict[str, Any]:
        """Analyze calibration with true labels available."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (max_probs >= bin_boundaries[i]) & (max_probs < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include right boundary for last bin
                mask |= (max_probs == 1.0)
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(predicted_labels[mask] == true_labels[mask])
                bin_confidence = np.mean(max_probs[mask])
                bin_count = np.sum(mask)
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        
        # Calculate calibration metrics
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Expected Calibration Error (ECE)
        ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(max_probs)
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_accuracies - bin_confidences))
        
        return {
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'overall_accuracy': np.mean(predicted_labels == true_labels)
        }
    
    def plot_reliability_diagram(
        self, 
        calibration_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot reliability diagram for model calibration.
        
        Args:
            calibration_data: Output from analyze_calibration
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        if 'bin_accuracies' not in calibration_data:
            raise ValueError("Supervised calibration data required for reliability diagram")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        bin_centers = (calibration_data['bin_boundaries'][:-1] + 
                      calibration_data['bin_boundaries'][1:]) / 2
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax1.bar(bin_centers, calibration_data['bin_accuracies'], 
                width=0.08, alpha=0.7, label='Accuracy')
        ax1.plot(bin_centers, calibration_data['bin_confidences'], 
                'ro-', label='Confidence')
        
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.bar(bin_centers, calibration_data['bin_counts'], 
                width=0.08, alpha=0.7)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add metrics as text
        ece = calibration_data['expected_calibration_error']
        mce = calibration_data['maximum_calibration_error']
        acc = calibration_data['overall_accuracy']
        
        metrics_text = f'ECE: {ece:.3f}\\nMCE: {mce:.3f}\\nAccuracy: {acc:.3f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

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
        self.is_fitted = hasattr(label_model, 'class_priors_') and label_model.class_priors_ is not None
        
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
                lf_names=lf_output.lf_names,
                example_ids=[lf_output.example_ids[i] for i in indices]
            )
            
            # Clone and retrain model on bootstrap sample
            bootstrap_model = LabelModel(cardinality=self.label_model.cardinality)
            bootstrap_model.fit(bootstrap_lf_output)
            
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
        """Estimate uncertainty using Monte Carlo dropout."""
        # Use the enhanced Monte Carlo dropout implementation
        mc_model = MonteCarloDropoutModel(self.label_model, dropout_rate=0.1)
        all_probs = mc_model.predict_with_dropout(lf_output, n_samples)
        
        # Calculate statistics
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
                random_state=np.random.randint(0, 10000)
            )
            ensemble_model.fit(lf_output)
            
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
        if hasattr(self.label_model, 'class_priors_') and self.label_model.class_priors_ is not None:
            params['class_priors_'] = self.label_model.class_priors_.copy()
        if hasattr(self.label_model, 'lf_accuracies_') and self.label_model.lf_accuracies_ is not None:
            params['lf_accuracies_'] = self.label_model.lf_accuracies_.copy()
        return params
    
    def _set_model_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set model parameters."""
        for name, value in params.items():
            if hasattr(self.label_model, name):
                setattr(self.label_model, name, value.copy())
    
    def _add_parameter_noise(self, scale: float = 0.01) -> None:
        """Add Gaussian noise to model parameters."""
        if hasattr(self.label_model, 'class_priors_') and self.label_model.class_priors_ is not None:
            noise = np.random.normal(0, scale, self.label_model.class_priors_.shape)
            self.label_model.class_priors_ += noise
            # Re-normalize to ensure valid probabilities
            self.label_model.class_priors_ = np.abs(self.label_model.class_priors_)
            self.label_model.class_priors_ /= np.sum(self.label_model.class_priors_)
        
        if hasattr(self.label_model, 'lf_accuracies_') and self.label_model.lf_accuracies_ is not None:
            noise = np.random.normal(0, scale, self.label_model.lf_accuracies_.shape)
            self.label_model.lf_accuracies_ += noise
            # Re-normalize to ensure valid probabilities
            self.label_model.lf_accuracies_ = np.abs(self.label_model.lf_accuracies_)
            for j in range(self.label_model.lf_accuracies_.shape[0]):
                for y in range(self.label_model.lf_accuracies_.shape[1]):
                    norm = np.sum(self.label_model.lf_accuracies_[j, y, :])
                    if norm > 0:
                        self.label_model.lf_accuracies_[j, y, :] /= norm
    
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


class MonteCarloDropoutModel:
    """
    Monte Carlo Dropout implementation for uncertainty estimation.
    
    This provides a more sophisticated uncertainty estimation approach
    by simulating dropout behavior in the label model.
    """
    
    def __init__(self, base_model: LabelModel, dropout_rate: float = 0.1):
        """
        Initialize MC Dropout model.
        
        Args:
            base_model: Trained base LabelModel
            dropout_rate: Dropout probability for uncertainty estimation
        """
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.original_params = None
        
    def enable_dropout(self):
        """Enable dropout by storing original parameters."""
        if hasattr(self.base_model, 'class_priors_') and self.base_model.class_priors_ is not None:
            self.original_params = {
                'class_priors_': self.base_model.class_priors_.copy(),
                'lf_accuracies_': self.base_model.lf_accuracies_.copy() if self.base_model.lf_accuracies_ is not None else None
            }
    
    def apply_dropout(self):
        """Apply dropout to model parameters."""
        if self.original_params is None:
            return
            
        # Apply dropout to class priors
        dropout_mask = np.random.random(self.base_model.class_priors_.shape) > self.dropout_rate
        self.base_model.class_priors_ = self.original_params['class_priors_'] * dropout_mask
        # Renormalize
        self.base_model.class_priors_ /= np.sum(self.base_model.class_priors_)
        
        # Apply dropout to LF accuracies
        if self.base_model.lf_accuracies_ is not None and self.original_params['lf_accuracies_'] is not None:
            dropout_mask = np.random.random(self.base_model.lf_accuracies_.shape) > self.dropout_rate
            self.base_model.lf_accuracies_ = self.original_params['lf_accuracies_'] * dropout_mask
            
            # Renormalize each accuracy distribution
            for j in range(self.base_model.lf_accuracies_.shape[0]):
                for y in range(self.base_model.lf_accuracies_.shape[1]):
                    norm = np.sum(self.base_model.lf_accuracies_[j, y, :])
                    if norm > 0:
                        self.base_model.lf_accuracies_[j, y, :] /= norm
    
    def restore_parameters(self):
        """Restore original parameters."""
        if self.original_params is not None:
            self.base_model.class_priors_ = self.original_params['class_priors_'].copy()
            if self.original_params['lf_accuracies_'] is not None:
                self.base_model.lf_accuracies_ = self.original_params['lf_accuracies_'].copy()
    
    def predict_with_dropout(self, lf_output: LFOutput, n_samples: int = 100) -> np.ndarray:
        """
        Generate predictions with Monte Carlo dropout.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            n_samples: Number of MC samples
            
        Returns:
            Array of shape (n_samples, n_examples, n_classes) with predictions
        """
        self.enable_dropout()
        all_predictions = []
        
        for _ in range(n_samples):
            self.apply_dropout()
            predictions = self.base_model.predict_proba(lf_output)
            all_predictions.append(predictions)
            
        self.restore_parameters()
        return np.array(all_predictions)


class AdvancedCalibrationAnalyzer(CalibrationAnalyzer):
    """
    Enhanced calibration analyzer with additional metrics and visualizations.
    """
    
    def __init__(self):
        super().__init__()
    
    def comprehensive_calibration_analysis(
        self,
        probabilities: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Comprehensive calibration analysis with multiple metrics.
        
        Args:
            probabilities: Predicted probabilities (n_examples, n_classes)
            true_labels: True labels (optional)
            n_bins: Number of bins for analysis
            
        Returns:
            Comprehensive calibration results
        """
        results = self.analyze_calibration(probabilities, true_labels, n_bins)
        
        # Add additional metrics
        max_probs = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        
        results.update({
            'confidence_stats': {
                'mean_confidence': np.mean(max_probs),
                'std_confidence': np.std(max_probs),
                'min_confidence': np.min(max_probs),
                'max_confidence': np.max(max_probs),
                'median_confidence': np.median(max_probs)
            },
            'entropy_stats': {
                'mean_entropy': np.mean(entropy),
                'std_entropy': np.std(entropy),
                'min_entropy': np.min(entropy),
                'max_entropy': np.max(entropy),
                'median_entropy': np.median(entropy)
            }
        })
        
        if true_labels is not None:
            # Add Brier score
            predicted_labels = np.argmax(probabilities, axis=1)
            brier_score = self._calculate_brier_score(probabilities, true_labels)
            log_loss = self._calculate_log_loss(probabilities, true_labels)
            
            results.update({
                'brier_score': brier_score,
                'log_loss': log_loss,
                'accuracy': np.mean(predicted_labels == true_labels)
            })
        
        return results
    
    def _calculate_brier_score(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate Brier score for calibration assessment."""
        n_classes = probabilities.shape[1]
        brier_score = 0.0
        
        for i, true_label in enumerate(true_labels):
            for j in range(n_classes):
                target = 1.0 if j == true_label else 0.0
                brier_score += (probabilities[i, j] - target) ** 2
        
        return brier_score / len(true_labels)
    
    def _calculate_log_loss(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate log loss."""
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        log_loss = 0.0
        
        for i, true_label in enumerate(true_labels):
            log_loss -= np.log(probabilities[i, true_label])
        
        return log_loss / len(true_labels)
    
    def plot_enhanced_reliability_diagram(
        self,
        calibration_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create enhanced reliability diagrams with additional insights.
        
        Args:
            calibration_data: Output from comprehensive_calibration_analysis
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        if 'bin_accuracies' not in calibration_data:
            raise ValueError("Supervised calibration data required for reliability diagram")
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reliability diagram
        bin_centers = (calibration_data['bin_boundaries'][:-1] + 
                      calibration_data['bin_boundaries'][1:]) / 2
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        ax1.bar(bin_centers, calibration_data['bin_accuracies'], 
                width=0.08, alpha=0.7, color='skyblue', label='Accuracy')
        ax1.plot(bin_centers, calibration_data['bin_confidences'], 
                'ro-', markersize=8, linewidth=2, label='Confidence')
        
        ax1.set_xlabel('Confidence', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.bar(bin_centers, calibration_data['bin_counts'], 
                width=0.08, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Calibration error by bin
        calibration_errors = np.abs(calibration_data['bin_accuracies'] - calibration_data['bin_confidences'])
        ax3.bar(bin_centers, calibration_errors, width=0.08, alpha=0.7, color='coral')
        ax3.set_xlabel('Confidence', fontsize=12)
        ax3.set_ylabel('|Accuracy - Confidence|', fontsize=12)
        ax3.set_title('Calibration Error by Bin', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Metrics summary
        ax4.axis('off')
        metrics_text = self._format_calibration_metrics(calibration_data)
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _format_calibration_metrics(self, calibration_data: Dict[str, Any]) -> str:
        """Format calibration metrics for display."""
        lines = ["CALIBRATION METRICS", "=" * 20]
        
        if 'expected_calibration_error' in calibration_data:
            lines.append(f"ECE: {calibration_data['expected_calibration_error']:.4f}")
        if 'maximum_calibration_error' in calibration_data:
            lines.append(f"MCE: {calibration_data['maximum_calibration_error']:.4f}")
        if 'brier_score' in calibration_data:
            lines.append(f"Brier Score: {calibration_data['brier_score']:.4f}")
        if 'log_loss' in calibration_data:
            lines.append(f"Log Loss: {calibration_data['log_loss']:.4f}")
        if 'accuracy' in calibration_data:
            lines.append(f"Accuracy: {calibration_data['accuracy']:.4f}")
        
        if 'confidence_stats' in calibration_data:
            conf_stats = calibration_data['confidence_stats']
            lines.extend([
                "",
                "CONFIDENCE STATISTICS",
                "-" * 20,
                f"Mean: {conf_stats['mean_confidence']:.4f}",
                f"Std:  {conf_stats['std_confidence']:.4f}",
                f"Min:  {conf_stats['min_confidence']:.4f}",
                f"Max:  {conf_stats['max_confidence']:.4f}"
            ])
        
        return "\n".join(lines)

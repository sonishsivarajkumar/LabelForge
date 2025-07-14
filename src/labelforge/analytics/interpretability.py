"""
Model interpretability tools for weak supervision.

This module provides tools for understanding and interpreting label model
behavior, including labeling function importance analysis and model diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib not available. Plotting functionality disabled.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. SHAP-based interpretability features disabled.")

from ..types import LFOutput
from ..label_model import LabelModel


class ModelAnalyzer:
    """
    Comprehensive analysis tools for label models.
    
    Provides methods for analyzing model behavior, labeling function
    interactions, and overall model performance.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize model analyzer.
        
        Args:
            label_model: Trained LabelModel instance
        """
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'class_priors_') and label_model.class_priors_ is not None
        
    def analyze_lf_interactions(self, lf_output: LFOutput) -> Dict[str, Any]:
        """
        Analyze interactions between labeling functions.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            
        Returns:
            Dictionary with interaction analysis results
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before analysis")
        
        votes = lf_output.votes
        n_lfs = votes.shape[1]
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        # Calculate pairwise correlations
        correlations = self._calculate_lf_correlations(votes)
        
        # Analyze agreement patterns
        agreements = self._analyze_agreements(votes)
        
        # Calculate coverage statistics
        coverage = self._calculate_coverage_stats(votes, lf_names)
        
        # Identify conflicts
        conflicts = self._identify_conflicts(votes, lf_names)
        
        return {
            'correlations': correlations,
            'agreements': agreements,
            'coverage': coverage,
            'conflicts': conflicts,
            'lf_names': lf_names
        }
    
    def _calculate_lf_correlations(self, votes: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix between labeling functions."""
        # Convert abstentions (-1) to NaN for correlation calculation
        votes_for_corr = votes.astype(float)
        votes_for_corr[votes_for_corr == -1] = np.nan
        
        n_lfs = votes.shape[1]
        correlations = np.full((n_lfs, n_lfs), np.nan)
        
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    # Only calculate correlation where both LFs vote
                    mask = ~(np.isnan(votes_for_corr[:, i]) | np.isnan(votes_for_corr[:, j]))
                    if np.sum(mask) > 1:
                        correlations[i, j] = np.corrcoef(
                            votes_for_corr[mask, i], 
                            votes_for_corr[mask, j]
                        )[0, 1]
        
        return correlations
    
    def _analyze_agreements(self, votes: np.ndarray) -> Dict[str, Any]:
        """Analyze agreement patterns between labeling functions."""
        n_examples, n_lfs = votes.shape
        
        # Calculate pairwise agreements
        agreements = np.zeros((n_lfs, n_lfs))
        overlaps = np.zeros((n_lfs, n_lfs))
        
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i != j:
                    # Find examples where both LFs vote (not abstain)
                    both_vote = (votes[:, i] != -1) & (votes[:, j] != -1)
                    overlaps[i, j] = np.sum(both_vote)
                    
                    if overlaps[i, j] > 0:
                        # Calculate agreement rate among overlapping examples
                        agreements[i, j] = np.sum(
                            votes[both_vote, i] == votes[both_vote, j]
                        ) / overlaps[i, j]
                else:
                    agreements[i, j] = 1.0
                    overlaps[i, j] = np.sum(votes[:, i] != -1)
        
        return {
            'agreement_matrix': agreements,
            'overlap_matrix': overlaps,
            'avg_pairwise_agreement': np.mean(agreements[np.triu_indices(n_lfs, k=1)])
        }
    
    def _calculate_coverage_stats(self, votes: np.ndarray, lf_names: List[str]) -> pd.DataFrame:
        """Calculate coverage statistics for each labeling function."""
        n_examples = votes.shape[0]
        
        coverage_stats = []
        for i, name in enumerate(lf_names):
            lf_votes = votes[:, i]
            
            # Basic coverage
            coverage = np.sum(lf_votes != -1) / n_examples
            
            # Class distribution
            unique_votes, counts = np.unique(lf_votes[lf_votes != -1], return_counts=True)
            class_dist = dict(zip(unique_votes.astype(int), counts))
            
            # Polarization (how often LF votes for each class)
            polarization = np.max(counts) / np.sum(counts) if len(counts) > 0 else 0
            
            coverage_stats.append({
                'lf_name': name,
                'coverage': coverage,
                'n_votes': np.sum(lf_votes != -1),
                'class_distribution': class_dist,
                'polarization': polarization,
                'entropy': self._calculate_entropy(counts) if len(counts) > 0 else 0
            })
        
        return pd.DataFrame(coverage_stats)
    
    def _identify_conflicts(self, votes: np.ndarray, lf_names: List[str]) -> pd.DataFrame:
        """Identify conflicting examples between labeling functions."""
        n_examples, n_lfs = votes.shape
        
        conflicts = []
        for i in range(n_examples):
            example_votes = votes[i]
            active_votes = example_votes[example_votes != -1]
            
            if len(active_votes) > 1 and len(np.unique(active_votes)) > 1:
                # This example has conflicting votes
                conflict_lfs = []
                for j, vote in enumerate(example_votes):
                    if vote != -1:
                        conflict_lfs.append((lf_names[j], int(vote)))
                
                conflicts.append({
                    'example_idx': i,
                    'n_conflicts': len(np.unique(active_votes)),
                    'conflicting_lfs': conflict_lfs,
                    'votes': active_votes.tolist()
                })
        
        return pd.DataFrame(conflicts)
    
    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate entropy of vote distribution."""
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log2(probs + 1e-8))
    
    def plot_lf_correlation_heatmap(
        self, 
        lf_output: LFOutput,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot correlation heatmap between labeling functions.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        analysis = self.analyze_lf_interactions(lf_output)
        correlations = analysis['correlations']
        lf_names = analysis['lf_names']
        
        plt.figure(figsize=(10, 8))
        
        # Create mask for NaN values
        mask = np.isnan(correlations)
        
        sns.heatmap(
            correlations,
            annot=True,
            cmap='RdBu_r',
            center=0,
            mask=mask,
            xticklabels=lf_names,
            yticklabels=lf_names,
            fmt='.2f'
        )
        
        plt.title('Labeling Function Correlation Matrix')
        plt.xlabel('Labeling Functions')
        plt.ylabel('Labeling Functions')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_analysis_report(self, lf_output: LFOutput) -> str:
        """
        Generate a comprehensive text report of model analysis.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            
        Returns:
            Formatted analysis report as string
        """
        analysis = self.analyze_lf_interactions(lf_output)
        
        report = ["LabelForge Model Analysis Report", "=" * 40, ""]
        
        # Coverage summary
        coverage_df = analysis['coverage']
        report.extend([
            "Coverage Summary:",
            f"• Average coverage: {coverage_df['coverage'].mean():.1%}",
            f"• Highest coverage: {coverage_df['coverage'].max():.1%} ({coverage_df.loc[coverage_df['coverage'].idxmax(), 'lf_name']})",
            f"• Lowest coverage: {coverage_df['coverage'].min():.1%} ({coverage_df.loc[coverage_df['coverage'].idxmin(), 'lf_name']})",
            ""
        ])
        
        # Agreement summary
        agreements = analysis['agreements']
        avg_agreement = agreements['avg_pairwise_agreement']
        report.extend([
            "Agreement Analysis:",
            f"• Average pairwise agreement: {avg_agreement:.1%}",
            ""
        ])
        
        # Conflict summary
        conflicts_df = analysis['conflicts']
        if len(conflicts_df) > 0:
            conflict_rate = len(conflicts_df) / lf_output.votes.shape[0]
            report.extend([
                "Conflict Analysis:",
                f"• Examples with conflicts: {len(conflicts_df)} ({conflict_rate:.1%})",
                f"• Average conflicts per conflicted example: {conflicts_df['n_conflicts'].mean():.1f}",
                ""
            ])
        else:
            report.extend(["Conflict Analysis:", "• No conflicts detected", ""])
        
        # Top correlations
        correlations = analysis['correlations']
        lf_names = analysis['lf_names']
        
        # Find strongest positive and negative correlations
        n_lfs = len(lf_names)
        correlations_list = []
        for i in range(n_lfs):
            for j in range(i + 1, n_lfs):
                if not np.isnan(correlations[i, j]):
                    correlations_list.append((
                        correlations[i, j], 
                        lf_names[i], 
                        lf_names[j]
                    ))
        
        if correlations_list:
            correlations_list.sort(reverse=True)
            
            report.extend([
                "Top Correlations:",
                f"• Highest positive: {correlations_list[0][1]} ↔ {correlations_list[0][2]} ({correlations_list[0][0]:.2f})",
                f"• Highest negative: {correlations_list[-1][1]} ↔ {correlations_list[-1][2]} ({correlations_list[-1][0]:.2f})",
                ""
            ])
        
        return "\\n".join(report)


class LFImportanceAnalyzer:
    """
    Analyze the importance and contribution of individual labeling functions.
    
    Provides tools for understanding which labeling functions contribute
    most to model performance and decision making.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize LF importance analyzer.
        
        Args:
            label_model: Trained LabelModel instance
        """
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'class_priors_') and label_model.class_priors_ is not None
    
    def calculate_lf_importance(
        self, 
        lf_output: LFOutput,
        method: str = "permutation"
    ) -> pd.DataFrame:
        """
        Calculate importance scores for labeling functions.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            method: Method for importance calculation ("permutation", "ablation")
            
        Returns:
            DataFrame with importance scores
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before importance analysis")
        
        if method == "permutation":
            return self._permutation_importance(lf_output)
        elif method == "ablation":
            return self._ablation_importance(lf_output)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _permutation_importance(self, lf_output: LFOutput) -> pd.DataFrame:
        """Calculate importance using permutation method."""
        # Get baseline predictions
        baseline_probs = self.label_model.predict_proba(lf_output)
        baseline_entropy = np.mean(-np.sum(baseline_probs * np.log(baseline_probs + 1e-8), axis=1))
        
        n_lfs = lf_output.votes.shape[1]
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        importance_scores = []
        
        for i in range(n_lfs):
            # Permute this LF's votes
            permuted_votes = lf_output.votes.copy()
            permuted_votes[:, i] = np.random.permutation(permuted_votes[:, i])
            
            permuted_lf_output = LFOutput(
                votes=permuted_votes,
                lf_names=lf_output.lf_names
            )
            
            # Get predictions with permuted LF
            permuted_probs = self.label_model.predict_proba(permuted_lf_output)
            permuted_entropy = np.mean(-np.sum(permuted_probs * np.log(permuted_probs + 1e-8), axis=1))
            
            # Importance is change in prediction entropy
            importance = permuted_entropy - baseline_entropy
            
            importance_scores.append({
                'lf_name': lf_names[i],
                'importance_score': importance,
                'baseline_entropy': baseline_entropy,
                'permuted_entropy': permuted_entropy
            })
        
        return pd.DataFrame(importance_scores).sort_values('importance_score', ascending=False)
    
    def _ablation_importance(self, lf_output: LFOutput) -> pd.DataFrame:
        """Calculate importance using ablation method."""
        # Train baseline model with all LFs
        baseline_model = LabelModel(cardinality=self.label_model.cardinality)
        baseline_model.fit(lf_output, verbose=False)
        baseline_probs = baseline_model.predict_proba(lf_output)
        baseline_entropy = np.mean(-np.sum(baseline_probs * np.log(baseline_probs + 1e-8), axis=1))
        
        n_lfs = lf_output.votes.shape[1]
        lf_names = lf_output.lf_names or [f"LF_{i}" for i in range(n_lfs)]
        
        importance_scores = []
        
        for i in range(n_lfs):
            # Create LF output without this LF
            ablated_votes = np.delete(lf_output.votes, i, axis=1)
            ablated_lf_names = [name for j, name in enumerate(lf_names) if j != i]
            
            ablated_lf_output = LFOutput(
                votes=ablated_votes,
                lf_names=ablated_lf_names
            )
            
            # Train model without this LF
            ablated_model = LabelModel(cardinality=self.label_model.cardinality)
            ablated_model.fit(ablated_lf_output, verbose=False)
            
            # Get predictions (need to adjust for removed LF)
            ablated_probs = ablated_model.predict_proba(ablated_lf_output)
            ablated_entropy = np.mean(-np.sum(ablated_probs * np.log(ablated_probs + 1e-8), axis=1))
            
            # Importance is change in prediction quality
            importance = baseline_entropy - ablated_entropy
            
            importance_scores.append({
                'lf_name': lf_names[i],
                'importance_score': importance,
                'baseline_entropy': baseline_entropy,
                'ablated_entropy': ablated_entropy
            })
        
        return pd.DataFrame(importance_scores).sort_values('importance_score', ascending=False)
    
    def plot_importance_scores(
        self, 
        importance_df: pd.DataFrame,
        top_k: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot labeling function importance scores.
        
        Args:
            importance_df: DataFrame from calculate_lf_importance
            top_k: Show only top k functions (None for all)
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting functionality")
        
        df_plot = importance_df.copy()
        if top_k is not None:
            df_plot = df_plot.head(top_k)
        
        plt.figure(figsize=(10, 6))
        
        bars = plt.barh(range(len(df_plot)), df_plot['importance_score'])
        plt.yticks(range(len(df_plot)), df_plot['lf_name'])
        plt.xlabel('Importance Score')
        plt.title('Labeling Function Importance')
        plt.grid(True, alpha=0.3)
        
        # Color bars by positive/negative importance
        for i, bar in enumerate(bars):
            if df_plot.iloc[i]['importance_score'] >= 0:
                bar.set_color('skyblue')
            else:
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SHAPLFAnalyzer:
    """
    SHAP-based analysis for labeling function importance.
    
    Provides SHAP values and feature attribution analysis for understanding
    which labeling functions contribute most to model predictions.
    """
    
    def __init__(self, label_model: LabelModel):
        """
        Initialize SHAP analyzer.
        
        Args:
            label_model: Trained LabelModel instance
        """
        self.label_model = label_model
        self.is_fitted = hasattr(label_model, 'class_priors_') and label_model.class_priors_ is not None
        
        if not HAS_SHAP:
            warnings.warn("SHAP not available. Install with: pip install shap")
    
    def compute_shap_values(
        self, 
        lf_output: LFOutput,
        background_size: int = 100,
        max_evals: int = 1000
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for labeling function importance.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            background_size: Size of background dataset for SHAP
            max_evals: Maximum evaluations for SHAP explainer
            
        Returns:
            Dictionary with SHAP values and analysis
        """
        if not self.is_fitted:
            raise ValueError("Label model must be fitted before SHAP analysis")
        
        if not HAS_SHAP:
            raise ImportError("SHAP package required. Install with: pip install shap")
        
        # Prepare data for SHAP
        votes = lf_output.votes.astype(float)
        
        # Create a wrapper function for the model
        def model_wrapper(votes_batch):
            """Wrapper to make model compatible with SHAP."""
            results = []
            for votes_row in votes_batch:
                # Create temporary LFOutput for single example
                temp_lf_output = LFOutput(
                    votes=votes_row.reshape(1, -1).astype(int),
                    lf_names=lf_output.lf_names,
                    example_ids=['temp']
                )
                probs = self.label_model.predict_proba(temp_lf_output)
                results.append(probs[0])
            return np.array(results)
        
        # Sample background data
        background_indices = np.random.choice(
            len(votes), 
            min(background_size, len(votes)), 
            replace=False
        )
        background_data = votes[background_indices]
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model_wrapper, background_data)
        
        # Calculate SHAP values for a subset of examples
        sample_size = min(50, len(votes))  # Limit for computational efficiency
        sample_indices = np.random.choice(len(votes), sample_size, replace=False)
        sample_votes = votes[sample_indices]
        
        shap_values = explainer.shap_values(sample_votes, nsamples=max_evals)
        
        # Process results
        results = {
            'shap_values': shap_values,
            'feature_names': lf_output.lf_names,
            'sample_indices': sample_indices,
            'background_data': background_data,
            'expected_value': explainer.expected_value
        }
        
        # Calculate feature importance
        if isinstance(shap_values, list):  # Multi-class case
            # Average absolute SHAP values across classes and examples
            mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
        else:  # Binary case
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        results['feature_importance'] = pd.DataFrame({
            'lf_name': lf_output.lf_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return results
    
    def plot_shap_summary(
        self, 
        shap_results: Dict[str, Any],
        plot_type: str = "summary",
        max_display: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create SHAP summary plots.
        
        Args:
            shap_results: Results from compute_shap_values
            plot_type: Type of plot ("summary", "bar", "waterfall")
            max_display: Maximum features to display
            save_path: Optional path to save the plot
        """
        if not HAS_SHAP:
            raise ImportError("SHAP package required for plotting")
        
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting")
        
        shap_values = shap_results['shap_values']
        feature_names = shap_results['feature_names']
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == "summary":
            if isinstance(shap_values, list):
                # Multi-class: plot for each class
                for i, sv in enumerate(shap_values):
                    plt.subplot(len(shap_values), 1, i + 1)
                    shap.summary_plot(sv, feature_names=feature_names, 
                                    max_display=max_display, show=False)
                    plt.title(f"Class {i}")
            else:
                shap.summary_plot(shap_values, feature_names=feature_names, 
                                max_display=max_display, show=False)
        
        elif plot_type == "bar":
            if isinstance(shap_values, list):
                # Average across classes for bar plot
                mean_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create bar plot
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Labeling Function Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AdvancedLFImportanceAnalyzer(LFImportanceAnalyzer):
    """
    Enhanced LF importance analyzer with additional methods.
    """
    
    def __init__(self, label_model: LabelModel):
        super().__init__(label_model)
        self.shap_analyzer = SHAPLFAnalyzer(label_model) if HAS_SHAP else None
    
    def comprehensive_importance_analysis(
        self, 
        lf_output: LFOutput,
        methods: List[str] = ["permutation", "weight", "correlation"],
        include_shap: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive importance analysis using multiple methods.
        
        Args:
            lf_output: LFOutput containing labeling function votes
            methods: List of importance methods to use
            include_shap: Whether to include SHAP analysis
            
        Returns:
            Dictionary with importance results from different methods
        """
        results = {}
        
        # Standard importance methods
        for method in methods:
            try:
                results[method] = self.calculate_lf_importance(lf_output, method=method)
            except Exception as e:
                warnings.warn(f"Failed to calculate {method} importance: {e}")
        
        # SHAP analysis
        if include_shap and self.shap_analyzer and HAS_SHAP:
            try:
                shap_results = self.shap_analyzer.compute_shap_values(lf_output)
                results['shap'] = shap_results['feature_importance']
            except Exception as e:
                warnings.warn(f"Failed to calculate SHAP importance: {e}")
        
        return results
    
    def plot_importance_comparison(
        self, 
        importance_results: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of importance scores from different methods.
        
        Args:
            importance_results: Results from comprehensive_importance_analysis
            save_path: Optional path to save the plot
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib required for plotting")
        
        # Combine results into a single DataFrame
        combined_df = None
        
        for method, df in importance_results.items():
            if 'lf_name' in df.columns and 'importance' in df.columns:
                method_df = df[['lf_name', 'importance']].copy()
                method_df.columns = ['lf_name', f'{method}_importance']
                
                if combined_df is None:
                    combined_df = method_df
                else:
                    combined_df = combined_df.merge(method_df, on='lf_name', how='outer')
        
        if combined_df is None or len(combined_df) == 0:
            warnings.warn("No valid importance results to plot")
            return
        
        # Create comparison plot
        n_methods = len(importance_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, df) in enumerate(importance_results.items()):
            if 'lf_name' in df.columns and 'importance' in df.columns:
                df_sorted = df.sort_values('importance', ascending=True)
                
                axes[i].barh(range(len(df_sorted)), df_sorted['importance'])
                axes[i].set_yticks(range(len(df_sorted)))
                axes[i].set_yticklabels(df_sorted['lf_name'])
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'{method.title()} Importance')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

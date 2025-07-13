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
        self.is_fitted = hasattr(label_model, 'mu')
        
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
        self.is_fitted = hasattr(label_model, 'mu')
    
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

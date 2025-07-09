#!/usr/bin/env python3
"""
Research Example: Weak Supervision for Text Classification

This example demonstrates how LabelForge can be used in research settings
to study weak supervision techniques and compare different approaches.

Based on concepts from:
- Ratner et al. "Snorkel: Rapid Training Data Creation with Weak Supervision"
- Bach et al. "Learning the Structure of Generative Models without Labeled Data"

Usage:
    python examples/research_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from labelforge import lf, LabelModel, apply_lfs, Example
from labelforge.lf import clear_lf_registry
from labelforge.datasets import load_example_data


def research_experiment():
    """
    Demonstrates a research workflow for studying weak supervision.
    """
    
    print("🔬 LabelForge Research Example")
    print("=" * 50)
    print("Studying programmatic weak supervision on text classification")
    print()
    
    # Clear any existing labeling functions
    clear_lf_registry()
    
    # Load research dataset
    print("📚 Loading research dataset...")
    examples = load_example_data("sentiment")
    print(f"Dataset size: {len(examples)} examples")
    print()
    
    # Define labeling functions based on different weak supervision strategies
    print("🏷️  Defining labeling functions...")
    
    # Strategy 1: Keyword-based heuristics
    @lf(name="positive_keywords", tags={"type": "keyword", "strategy": "heuristic"})
    def positive_keywords(example):
        """Simple keyword matching for positive sentiment."""
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect", "best"]
        return 1 if any(word in example.text.lower() for word in positive_words) else 0
    
    @lf(name="negative_keywords", tags={"type": "keyword", "strategy": "heuristic"})
    def negative_keywords(example):
        """Simple keyword matching for negative sentiment."""
        negative_words = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disgusting"]
        return 1 if any(word in example.text.lower() for word in negative_words) else 0
    
    # Strategy 2: Pattern-based rules
    @lf(name="exclamation_positive", tags={"type": "pattern", "strategy": "rule"})
    def exclamation_positive(example):
        """Positive sentiment from exclamation patterns."""
        text = example.text
        if "!" in text:
            # Simple heuristic: exclamations with positive context
            if any(word in text.lower() for word in ["great", "amazing", "love", "awesome"]):
                return 1
        return 0
    
    @lf(name="question_uncertainty", tags={"type": "pattern", "strategy": "rule"})
    def question_uncertainty(example):
        """Questions often indicate uncertainty or negative sentiment."""
        if "?" in example.text and any(word in example.text.lower() for word in ["why", "how", "what"]):
            return 1  # Potentially negative
        return 0
    
    # Strategy 3: Length-based heuristics
    @lf(name="short_negative", tags={"type": "length", "strategy": "heuristic"})
    def short_negative(example):
        """Short texts are often negative (complaints, criticism)."""
        if len(example.text.split()) < 5:
            negative_indicators = ["no", "not", "bad", "hate", "ugh", "meh"]
            if any(word in example.text.lower() for word in negative_indicators):
                return 1
        return 0
    
    print(f"Defined {len([positive_keywords, negative_keywords, exclamation_positive, question_uncertainty, short_negative])} labeling functions")
    print()
    
    # Apply labeling functions
    print("🧪 Applying labeling functions...")
    lf_output = apply_lfs(examples)
    
    print(f"Vote matrix shape: {lf_output.votes.shape}")
    print(f"Abstention rate: {(lf_output.votes == -1).mean():.2%}")
    print()
    
    # Analyze labeling function statistics
    print("📊 Labeling Function Analysis:")
    print("-" * 30)
    
    # Coverage analysis
    coverage = lf_output.coverage()
    for i, lf_name in enumerate(lf_output.lf_names):
        print(f"{lf_name:20} Coverage: {coverage[i]:.2%}")
    
    print()
    
    # Overlap and conflict analysis
    overlap = lf_output.overlap()
    conflict = lf_output.conflict()
    
    print("🔍 Overlap Matrix (showing correlation between LFs):")
    print("LF pairs with high overlap may be redundant")
    for i, name_i in enumerate(lf_output.lf_names):
        for j, name_j in enumerate(lf_output.lf_names):
            if i < j and overlap[i, j] > 0.1:  # Only show significant overlaps
                print(f"  {name_i} ↔ {name_j}: {overlap[i, j]:.2%}")
    
    print()
    
    # Train label model
    print("🧠 Training probabilistic label model...")
    print("Using EM algorithm to learn LF accuracies and correlations")
    
    label_model = LabelModel(
        cardinality=2,
        max_iter=100,
        tol=1e-4,
        verbose=True
    )
    
    label_model.fit(lf_output)
    
    print(f"Model converged in {label_model.history_['n_iter']} iterations")
    print(f"Final log-likelihood: {label_model.history_['log_likelihood'][-1]:.4f}")
    print()
    
    # Generate predictions
    print("📈 Generating probabilistic labels...")
    probs = label_model.predict_proba(lf_output)
    predictions = label_model.predict(lf_output)
    
    # Analysis of results
    print("🎯 Results Analysis:")
    print("-" * 20)
    
    # Class distribution
    class_counts = np.bincount(predictions)
    print(f"Predicted class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} examples ({count/len(predictions):.1%})")
    
    print()
    
    # Confidence analysis
    max_probs = np.max(probs, axis=1)
    print(f"Prediction confidence:")
    print(f"  Mean confidence: {max_probs.mean():.3f}")
    print(f"  High confidence (>0.8): {(max_probs > 0.8).mean():.1%}")
    print(f"  Low confidence (<0.6): {(max_probs < 0.6).mean():.1%}")
    
    print()
    
    # Research insights
    print("💡 Research Insights:")
    print("-" * 20)
    print("1. Weak supervision allows rapid prototyping of labeling strategies")
    print("2. Different heuristic types (keyword, pattern, length) capture different signals")
    print("3. Label model automatically learns to weight and combine noisy sources")
    print("4. Probabilistic outputs provide uncertainty estimates for active learning")
    print("5. Framework enables systematic study of different weak supervision approaches")
    print()
    
    print("📚 For research applications, consider:")
    print("• Comparing with manually labeled gold standard")
    print("• Studying robustness to different LF combinations")
    print("• Analyzing performance across different domains")
    print("• Investigating theoretical properties of the label model")
    print("• Exploring active learning with uncertainty estimates")
    print()
    
    print("🎉 Research example completed!")
    print("This framework enables systematic study of weak supervision methods.")
    

if __name__ == "__main__":
    research_experiment()

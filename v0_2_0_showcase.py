"""
LabelForge v0.2.0 Feature Showcase

This example demonstrates the new advanced analytics, benchmarking,
and research features introduced in LabelForge v0.2.0.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Core LabelForge imports
from labelforge import (
    lf, LabelModel, Example, LFOutput, apply_lfs,
    HAS_ANALYTICS, HAS_BENCHMARKS, HAS_INTEGRATIONS
)

# New v0.2.0 imports
if HAS_ANALYTICS:
    from labelforge.analytics import (
        EnhancedUncertaintyQuantifier,
        AdvancedCalibrationAnalyzer,
        SHAPLFImportance,
        EnhancedConvergenceTracker
    )

if HAS_BENCHMARKS:
    from labelforge.benchmarks import (
        BenchmarkDataLoader,
        BenchmarkEvaluator,
        BenchmarkMetrics,
        ExperimentConfig,
        ResultsLogger,
        ReproducibilityReport
    )

if HAS_INTEGRATIONS:
    from labelforge.integrations import (
        PyTorchExporter,
        HuggingFaceExporter
    )


def create_sample_labeling_functions():
    """Create sample labeling functions for demonstration."""
    
    @lf(name="positive_sentiment")
    def positive_words(example):
        """Detect positive sentiment words."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        text_lower = example.text.lower()
        
        if any(word in text_lower for word in positive_words):
            return 1  # Positive
        return -1  # Abstain
    
    @lf(name="negative_sentiment")
    def negative_words(example):
        """Detect negative sentiment words."""
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'worst']
        text_lower = example.text.lower()
        
        if any(word in text_lower for word in negative_words):
            return 0  # Negative
        return -1  # Abstain
    
    @lf(name="length_based")
    def text_length_indicator(example):
        """Use text length as sentiment indicator."""
        if len(example.text.split()) > 20:
            return 1  # Longer reviews tend to be more positive
        elif len(example.text.split()) < 5:
            return 0  # Very short reviews tend to be negative
        return -1  # Abstain
    
    @lf(name="exclamation_positive")
    def exclamation_positive(example):
        """Exclamation marks often indicate positive sentiment."""
        if '!' in example.text and not any(word in example.text.lower() 
                                         for word in ['not', 'no', 'never']):
            return 1  # Positive
        return -1  # Abstain
    
    @lf(name="caps_emphasis")
    def caps_emphasis(example):
        """ALL CAPS words can indicate strong sentiment."""
        words = example.text.split()
        caps_words = [w for w in words if w.isupper() and len(w) > 2]
        
        if caps_words:
            # Simple heuristic: if caps word is positive/negative
            positive_caps = ['AMAZING', 'GREAT', 'EXCELLENT', 'LOVE']
            negative_caps = ['TERRIBLE', 'AWFUL', 'HATE', 'WORST']
            
            for word in caps_words:
                if word in positive_caps:
                    return 1
                elif word in negative_caps:
                    return 0
        return -1  # Abstain
    
    return [positive_words, negative_words, text_length_indicator, 
            exclamation_positive, caps_emphasis]


def create_sample_data():
    """Create sample movie review data."""
    reviews = [
        ("This movie was absolutely amazing! Great acting and wonderful story.", 1),
        ("Terrible film. One of the worst movies I've ever seen.", 0),
        ("The cinematography was excellent and the plot was fantastic!", 1),
        ("Boring and disappointing. Not worth watching at all.", 0),
        ("Good entertainment value but nothing special.", 1),
        ("AWFUL movie! Complete waste of time and money!", 0),
        ("I loved every minute of it. Highly recommended!", 1),
        ("The story was okay, not bad but not great either.", 1),
        ("Horrible acting and terrible script. Avoid this one.", 0),
        ("An excellent piece of cinema with amazing performances.", 1),
        ("Bad plot, poor direction, just terrible overall.", 0),
        ("Really enjoyed this film. Great fun for the whole family!", 1),
        ("The movie was good with some great moments throughout.", 1),
        ("Disappointing sequel. The original was much better.", 0),
        ("Fantastic special effects and wonderful character development.", 1),
        ("Not the worst film but definitely not good either.", 0),
        ("AMAZING storyline and EXCELLENT character arcs!", 1),
        ("Terrible waste of talented actors on a bad script.", 0),
        ("Good movie overall, would recommend to friends.", 1),
        ("Awful film. Couldn't even finish watching it.", 0),
    ]
    
    examples = []
    true_labels = []
    
    for i, (text, label) in enumerate(reviews):
        examples.append(Example(text=text, example_id=str(i)))
        true_labels.append(label)
    
    return examples, np.array(true_labels)


def demonstrate_basic_workflow():
    """Demonstrate the basic LabelForge workflow with new features."""
    print("🚀 LabelForge v0.2.0 Feature Showcase")
    print("=" * 50)
    
    # Create sample data
    print("\n📊 Creating sample movie review dataset...")
    examples, true_labels = create_sample_data()
    print(f"Created {len(examples)} examples with {len(np.unique(true_labels))} classes")
    
    # Create labeling functions
    print("\n🏷️  Creating labeling functions...")
    labeling_functions = create_sample_labeling_functions()
    print(f"Created {len(labeling_functions)} labeling functions")
    
    # Apply labeling functions
    print("\n⚙️  Applying labeling functions...")
    lf_output = apply_lfs(labeling_functions, examples)
    print(f"LF output shape: {lf_output.lf_labels.shape}")
    
    # Train label model
    print("\n🤖 Training label model...")
    model = LabelModel(cardinality=2, verbose=True)
    model.fit(lf_output)
    
    # Get predictions
    predictions = model.predict(lf_output)
    probabilities = model.predict_proba(lf_output)
    
    print(f"Model accuracy: {np.mean(predictions == true_labels):.3f}")
    
    return examples, lf_output, model, predictions, probabilities, true_labels


def demonstrate_advanced_analytics(examples, lf_output, model, predictions, probabilities, true_labels):
    """Demonstrate the new advanced analytics features."""
    if not HAS_ANALYTICS:
        print("\n⚠️  Analytics module not available. Skipping analytics demo.")
        return
    
    print("\n🔬 Advanced Analytics Features")
    print("-" * 30)
    
    # Uncertainty Quantification
    print("\n📊 Uncertainty Quantification...")
    uncertainty_quantifier = EnhancedUncertaintyQuantifier(model)
    uncertainty_results = uncertainty_quantifier.estimate_uncertainty(
        lf_output, method='monte_carlo', n_samples=100
    )
    
    print(f"Mean uncertainty: {np.mean(uncertainty_results['uncertainty']):.3f}")
    print(f"High uncertainty examples: {np.sum(uncertainty_results['uncertainty'] > 0.3)}")
    
    # Calibration Analysis
    print("\n📈 Calibration Analysis...")
    calibration_analyzer = AdvancedCalibrationAnalyzer()
    calibration_results = calibration_analyzer.analyze_calibration(
        probabilities, true_labels
    )
    
    print(f"Expected Calibration Error: {calibration_results['ece']:.3f}")
    print(f"Brier Score: {calibration_results['brier_score']:.3f}")
    
    # SHAP-based LF Importance
    print("\n🔍 Labeling Function Importance (SHAP)...")
    try:
        shap_analyzer = SHAPLFImportance(model)
        importance_results = shap_analyzer.compute_importance(lf_output)
        
        print("LF Importance Scores:")
        for i, importance in enumerate(importance_results['importance_scores']):
            print(f"  LF {i}: {importance:.3f}")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
    
    # Convergence Analysis
    print("\n📉 Convergence Analysis...")
    convergence_tracker = EnhancedConvergenceTracker()
    # Note: In real usage, this would be called during model training
    print("Convergence tracking would be integrated during model.fit()")


def demonstrate_benchmarking(examples, lf_output, true_labels):
    """Demonstrate the new benchmarking capabilities."""
    if not HAS_BENCHMARKS:
        print("\n⚠️  Benchmarks module not available. Skipping benchmarking demo.")
        return
    
    print("\n🏆 Benchmarking Suite")
    print("-" * 20)
    
    # Benchmark Metrics
    print("\n📏 Comprehensive Metrics...")
    benchmark_metrics = BenchmarkMetrics()
    
    # Train model for benchmarking
    model = LabelModel(cardinality=2)
    model.fit(lf_output)
    predictions = model.predict(lf_output)
    probabilities = model.predict_proba(lf_output)
    
    # Generate comprehensive report
    report = benchmark_metrics.compute_full_report(
        lf_output, predictions, true_labels, probabilities
    )
    
    print(f"Model Accuracy: {report['model_performance']['accuracy']:.3f}")
    print(f"F1 Macro: {report['model_performance']['f1_macro']:.3f}")
    print(f"Mean LF Coverage: {report['lf_performance']['summary']['mean_coverage']:.3f}")
    print(f"Conflict Rate: {report['agreement_analysis']['conflict_rate']:.3f}")
    
    # Reproducibility Tools
    print("\n🔄 Reproducibility Tools...")
    
    # Create experiment configuration
    config = ExperimentConfig()
    config.set_model_config('LabelModel', cardinality=2, tolerance=1e-3)
    config.set_data_config('movie_reviews_demo')
    config.set_labeling_functions(['positive_sentiment', 'negative_sentiment', 'length_based'])
    config.set_evaluation_config(['accuracy', 'f1_macro'], cv_folds=3)
    config.set_random_seed(42)
    
    print(f"Experiment hash: {config.get_hash()}")
    
    # Results logging
    logger = ResultsLogger("v0_2_0_demo")
    logger.log_config(config)
    logger.log_metrics(report['model_performance'])
    logger.log_model(model)
    logger.log_predictions(predictions, probabilities)
    
    summary_path = logger.create_summary_report()
    print(f"Results logged to: {logger.experiment_dir}")


def demonstrate_integrations(examples, lf_output, model):
    """Demonstrate the ML ecosystem integrations."""
    if not HAS_INTEGRATIONS:
        print("\n⚠️  Integrations module not available. Skipping integrations demo.")
        return
    
    print("\n🔗 ML Ecosystem Integrations")
    print("-" * 30)
    
    # PyTorch Integration
    print("\n🔥 PyTorch Integration...")
    try:
        pytorch_exporter = PyTorchExporter(model)
        pytorch_dataset = pytorch_exporter.to_pytorch_dataset(examples, lf_output)
        print(f"PyTorch dataset created with {len(pytorch_dataset)} examples")
        
        # Example of accessing dataset
        sample_item = pytorch_dataset[0]
        print(f"Sample item keys: {list(sample_item.keys())}")
        
    except ImportError:
        print("PyTorch not available")
    except Exception as e:
        print(f"PyTorch export failed: {e}")
    
    # Hugging Face Integration
    print("\n🤗 Hugging Face Integration...")
    try:
        hf_exporter = HuggingFaceExporter(model)
        hf_dataset = hf_exporter.to_hf_dataset(examples, lf_output)
        print(f"Hugging Face dataset created with {len(hf_dataset)} examples")
        print(f"Dataset features: {list(hf_dataset.features.keys())}")
        
    except ImportError:
        print("Hugging Face datasets not available")
    except Exception as e:
        print(f"Hugging Face export failed: {e}")


def demonstrate_synthetic_benchmarks():
    """Demonstrate synthetic benchmark generation."""
    if not HAS_BENCHMARKS:
        print("\n⚠️  Benchmarks module not available. Skipping synthetic demo.")
        return
    
    print("\n🧪 Synthetic Benchmark Generation")
    print("-" * 35)
    
    # Create synthetic dataset
    loader = BenchmarkDataLoader()
    examples, lf_output, true_labels = loader.load_synthetic_dataset(
        n_examples=200,
        n_classes=2,
        n_lfs=8,
        lf_accuracy_range=(0.6, 0.85),
        coverage_range=(0.4, 0.7)
    )
    
    print(f"Generated synthetic dataset:")
    print(f"  - {len(examples)} examples")
    print(f"  - {lf_output.lf_labels.shape[1]} labeling functions")
    print(f"  - {len(np.unique(true_labels))} classes")
    
    # Train and evaluate model
    model = LabelModel(cardinality=2)
    model.fit(lf_output)
    
    evaluator = BenchmarkEvaluator()
    results = evaluator.evaluate_model(model, lf_output, true_labels)
    
    print(f"\nSynthetic benchmark results:")
    for metric, score in results.items():
        if not np.isnan(score):
            print(f"  {metric}: {score:.3f}")


def main():
    """Run the complete v0.2.0 feature showcase."""
    print("🎯 Welcome to LabelForge v0.2.0!")
    print("This showcase demonstrates the new advanced features.")
    print(f"\nFeature availability:")
    print(f"  - Analytics: {'✅' if HAS_ANALYTICS else '❌'}")
    print(f"  - Benchmarks: {'✅' if HAS_BENCHMARKS else '❌'}")
    print(f"  - Integrations: {'✅' if HAS_INTEGRATIONS else '❌'}")
    
    try:
        # Basic workflow
        examples, lf_output, model, predictions, probabilities, true_labels = demonstrate_basic_workflow()
        
        # Advanced analytics
        demonstrate_advanced_analytics(examples, lf_output, model, predictions, probabilities, true_labels)
        
        # Benchmarking
        demonstrate_benchmarking(examples, lf_output, true_labels)
        
        # Integrations
        demonstrate_integrations(examples, lf_output, model)
        
        # Synthetic benchmarks
        demonstrate_synthetic_benchmarks()
        
        print("\n🎉 LabelForge v0.2.0 showcase completed!")
        print("Check out the documentation for more advanced features and examples.")
        
    except Exception as e:
        print(f"\n❌ Error during showcase: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

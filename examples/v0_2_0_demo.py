"""
LabelForge v0.2.0 Advanced Features Demo

This script demonstrates the new advanced analytics, research tools,
ML ecosystem integrations, and optimization features in LabelForge v0.2.0.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Core LabelForge imports
from labelforge import LabelModel, Example, LFOutput
from labelforge.lf import LabelingFunction

# Advanced analytics (new in v0.2.0)
try:
    from labelforge.analytics import (
        EnhancedUncertaintyQuantifier,
        AdvancedCalibrationAnalyzer,
        SHAPLFImportance,
        EnhancedConvergenceTracker
    )
    print("✓ Advanced analytics available")
except ImportError:
    print("✗ Advanced analytics not available")

# ML Ecosystem integrations (enhanced in v0.2.0)
try:
    from labelforge.integrations import PyTorchExporter, HuggingFaceExporter
    print("✓ ML integrations available")
except ImportError:
    print("✗ ML integrations not available")

# Research tools (new in v0.2.0)
try:
    from labelforge.research import (
        BenchmarkSuite,
        WRENCHBenchmark,
        SyntheticDataGenerator,
        StatisticalTester,
        CrossValidationEvaluator,
        LaTeXExporter,
        ExperimentConfig,
        SeedManager
    )
    print("✓ Research tools available")
except ImportError:
    print("✗ Research tools not available")

# Templates and tools (enhanced in v0.2.0)
try:
    from labelforge.templates import KeywordLF, SentimentLF, RegexBuilder
    print("✓ LF templates available")
except ImportError:
    print("✗ LF templates not available")

# Optimization tools (new in v0.2.0)
try:
    from labelforge.optimization import ParallelLFApplicator, parallel_apply_lfs
    print("✓ Optimization tools available")
except ImportError:
    print("✗ Optimization tools not available")


def create_demo_data() -> tuple:
    """Create demo dataset for demonstrating features."""
    print("\\n=== Creating Demo Dataset ===")
    
    # Create synthetic examples
    texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The film was boring and poorly executed. Waste of time.",
        "Great cinematography and excellent acting. Highly recommend!",
        "Terrible plot and bad dialogue. Very disappointed.",
        "Amazing special effects but weak storyline.",
        "One of the best films I've ever seen. Perfect in every way.",
        "Not my cup of tea, but decent production quality.",
        "Incredible performance by the lead actor. Outstanding!",
        "The movie was okay, nothing special but watchable.",
        "Absolutely awful. Couldn't even finish watching it."
    ]
    
    examples = [
        Example(text=text, id=str(i), metadata={'domain': 'movie_reviews'})
        for i, text in enumerate(texts)
    ]
    
    # Ground truth labels (0=negative, 1=positive)
    true_labels = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    
    print(f"Created {len(examples)} movie review examples")
    return examples, true_labels


def create_labeling_functions() -> List[LabelingFunction]:
    """Create labeling functions for the demo."""
    print("\\n=== Creating Labeling Functions ===")
    
    # Simple keyword-based LFs
    def positive_keywords_lf(example):
        positive_words = ['fantastic', 'amazing', 'excellent', 'great', 'outstanding', 'perfect', 'loved']
        text_lower = example.text.lower()
        return 1 if any(word in text_lower for word in positive_words) else -1
    
    def negative_keywords_lf(example):
        negative_words = ['terrible', 'awful', 'boring', 'bad', 'waste', 'disappointed']
        text_lower = example.text.lower()
        return 0 if any(word in text_lower for word in negative_words) else -1
    
    def exclamation_lf(example):
        return 1 if '!' in example.text else -1
    
    def length_lf(example):
        return 1 if len(example.text.split()) > 10 else 0
    
    labeling_functions = [
        LabelingFunction("positive_keywords", positive_keywords_lf),
        LabelingFunction("negative_keywords", negative_keywords_lf),
        LabelingFunction("exclamation_marks", exclamation_lf),
        LabelingFunction("long_reviews", length_lf)
    ]
    
    print(f"Created {len(labeling_functions)} labeling functions")
    return labeling_functions


def demo_basic_workflow(examples: List[Example], lfs: List[LabelingFunction], true_labels: np.ndarray):
    """Demonstrate basic LabelForge workflow."""
    print("\\n=== Basic Weak Supervision Workflow ===")
    
    # Apply labeling functions
    votes = np.full((len(examples), len(lfs)), -1)
    for i, example in enumerate(examples):
        for j, lf in enumerate(lfs):
            try:
                vote = lf(example)
                votes[i, j] = vote if vote is not None else -1
            except:
                votes[i, j] = -1
    
    lf_output = LFOutput(
        votes=votes,
        example_ids=[ex.id for ex in examples],
        lf_names=[lf.name for lf in lfs]
    )
    
    print(f"LF coverage: {np.mean(votes != -1):.2%}")
    print(f"Vote matrix shape: {votes.shape}")
    
    # Train label model
    label_model = LabelModel(cardinality=2)
    label_model.fit(lf_output)
    
    # Make predictions
    predictions = label_model.predict(lf_output)
    probabilities = label_model.predict_proba(lf_output)
    
    # Evaluate accuracy
    accuracy = np.mean(predictions == true_labels)
    print(f"Label model accuracy: {accuracy:.2%}")
    
    return label_model, lf_output, predictions, probabilities


def demo_advanced_analytics(label_model: LabelModel, lf_output: LFOutput, 
                          predictions: np.ndarray, probabilities: np.ndarray):
    """Demonstrate advanced analytics features."""
    print("\\n=== Advanced Analytics Demo ===")
    
    try:
        # Uncertainty quantification with Monte Carlo dropout
        print("\\n1. Monte Carlo Dropout Uncertainty:")
        uncertainty_quantifier = EnhancedUncertaintyQuantifier(label_model)
        
        # Check if model is fitted before uncertainty estimation
        if hasattr(label_model, 'class_priors_'):
            uncertainties = uncertainty_quantifier.monte_carlo_uncertainty(
                lf_output, n_samples=50, dropout_rate=0.1
            )
            
            print(f"   Mean epistemic uncertainty: {np.mean(uncertainties['epistemic']):.3f}")
            print(f"   Mean aleatoric uncertainty: {np.mean(uncertainties['aleatoric']):.3f}")
        else:
            print("   Model not properly fitted for uncertainty estimation")
        
        # Calibration analysis
        print("\\n2. Advanced Calibration Analysis:")
        calibration_analyzer = AdvancedCalibrationAnalyzer()
        
        # Create dummy true labels for calibration (in real use, you'd have validation data)
        dummy_true_labels = (probabilities[:, 1] > 0.5).astype(int)
        
        calibration_result = calibration_analyzer.comprehensive_calibration_analysis(
            probabilities, dummy_true_labels
        )
        
        print(f"   Expected Calibration Error: {calibration_result['ece']:.3f}")
        print(f"   Brier Score: {calibration_result['brier_score']:.3f}")
        
        # SHAP-based LF importance
        print("\\n3. SHAP Labeling Function Importance:")
        shap_analyzer = SHAPLFImportance(label_model)
        
        if hasattr(label_model, 'class_priors_'):
            shap_values = shap_analyzer.explain_predictions(lf_output, max_samples=len(lf_output.votes))
            
            print("   SHAP importance scores:")
            for i, importance in enumerate(np.mean(np.abs(shap_values), axis=0)):
                print(f"     LF {i}: {importance:.3f}")
        else:
            print("   Model not properly fitted for SHAP analysis")
        
        # Convergence analysis
        print("\\n4. Enhanced Convergence Tracking:")
        convergence_tracker = EnhancedConvergenceTracker()
        
        # Re-train model with convergence tracking
        tracked_model = LabelModel(cardinality=2)
        convergence_tracker.track_training(tracked_model, lf_output)
        
        if convergence_tracker.convergence_history:
            print(f"   Converged after {len(convergence_tracker.convergence_history)} iterations")
            print(f"   Final log-likelihood: {convergence_tracker.convergence_history[-1]:.3f}")
        
    except Exception as e:
        print(f"Analytics demo failed: {e}")


def demo_research_tools(examples: List[Example], lf_output: LFOutput, true_labels: np.ndarray):
    """Demonstrate research and benchmarking tools."""
    print("\\n=== Research Tools Demo ===")
    
    try:
        # Reproducibility setup
        print("\\n1. Reproducibility Setup:")
        seed_manager = SeedManager(master_seed=42)
        seed_manager.set_all_seeds()
        print("   ✓ Random seeds set for reproducibility")
        
        experiment_config = ExperimentConfig(
            experiment_name="movie_sentiment_demo",
            method_name="LabelModel",
            hyperparameters={"cardinality": 2},
            data_config={"n_examples": len(examples), "n_lfs": lf_output.votes.shape[1]},
            environment_config={"platform": "demo"},
            random_seed=42,
            timestamp=1234567890.0,
            experiment_id="demo_exp_001"
        )
        print("   ✓ Experiment configuration created")
        
        # Statistical testing
        print("\\n2. Statistical Testing:")
        statistical_tester = StatisticalTester(alpha=0.05)
        
        # Create dummy comparison data
        method_a_scores = np.random.normal(0.75, 0.1, 10)
        method_b_scores = np.random.normal(0.70, 0.1, 10)
        
        t_test_result = statistical_tester.paired_t_test(method_a_scores, method_b_scores)
        print(f"   Paired t-test p-value: {t_test_result.p_value:.3f}")
        print(f"   Statistically significant: {t_test_result.is_significant}")
        
        # Cross-validation
        print("\\n3. Cross-Validation:")
        cv_evaluator = CrossValidationEvaluator(cv_folds=3)  # Small for demo
        
        cv_result = cv_evaluator.stratified_cv(
            examples=examples,
            lf_output=lf_output,
            true_labels=true_labels,
            model_class=LabelModel,
            model_params={"cardinality": 2}
        )
        
        print(f"   CV Accuracy: {cv_result.mean_score:.3f} ± {cv_result.std_score:.3f}")
        print(f"   95% CI: ({cv_result.confidence_interval[0]:.3f}, {cv_result.confidence_interval[1]:.3f})")
        
        # Synthetic data generation
        print("\\n4. Synthetic Data Generation:")
        data_generator = SyntheticDataGenerator(seed=42)
        
        synthetic_examples, synthetic_features, synthetic_labels = data_generator.generate_classification_dataset(
            n_examples=50, n_classes=2, n_features=5
        )
        
        print(f"   Generated {len(synthetic_examples)} synthetic examples")
        print(f"   Feature matrix shape: {synthetic_features.shape}")
        
        # LaTeX export
        print("\\n5. LaTeX Export:")
        latex_exporter = LaTeXExporter(precision=3)
        
        # Create dummy benchmark results
        from labelforge.research.benchmarks import BenchmarkResult
        dummy_results = [
            BenchmarkResult("LabelModel", "MovieReviews", 0.75, 0.73, 0.74, 0.72),
            BenchmarkResult("MajorityVote", "MovieReviews", 0.60, 0.58, 0.59, 0.57)
        ]
        
        latex_table = latex_exporter.results_table(
            dummy_results,
            caption="Demo results comparison"
        )
        
        print("   ✓ LaTeX table generated successfully")
        
    except Exception as e:
        print(f"Research tools demo failed: {e}")


def demo_ml_integrations(label_model: LabelModel, examples: List[Example], lf_output: LFOutput):
    """Demonstrate ML ecosystem integrations."""
    print("\\n=== ML Ecosystem Integrations Demo ===")
    
    try:
        # PyTorch integration
        print("\\n1. PyTorch Integration:")
        pytorch_exporter = PyTorchExporter(label_model)
        
        if hasattr(label_model, 'class_priors_'):  # Check if model is fitted
            pytorch_dataset = pytorch_exporter.to_pytorch_dataset(
                examples=examples,
                lf_output=lf_output,
                use_probabilities=True
            )
            
            print(f"   ✓ Created PyTorch dataset with {len(pytorch_dataset)} examples")
            
            # Show sample
            sample = pytorch_dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
        else:
            print("   Model not fitted, skipping PyTorch export")
        
        # Hugging Face integration
        print("\\n2. Hugging Face Integration:")
        hf_exporter = HuggingFaceExporter(label_model)
        
        if hasattr(label_model, 'class_priors_'):  # Check if model is fitted
            hf_dataset = hf_exporter.to_hf_dataset(
                examples=examples,
                lf_output=lf_output,
                include_probabilities=True
            )
            
            print(f"   ✓ Created HF dataset with {len(hf_dataset)} examples")
            print(f"   Dataset features: {hf_dataset.features}")
        else:
            print("   Model not fitted, skipping Hugging Face export")
        
    except Exception as e:
        print(f"ML integrations demo failed: {e}")


def demo_optimization_features(examples: List[Example], labeling_functions: List[LabelingFunction]):
    """Demonstrate optimization and performance features."""
    print("\\n=== Optimization Features Demo ===")
    
    try:
        # Parallel LF application
        print("\\n1. Parallel Labeling Function Application:")
        
        parallel_applicator = ParallelLFApplicator(
            n_jobs=2,  # Use 2 workers for demo
            backend='multiprocessing',
            verbose=True
        )
        
        import time
        start_time = time.time()
        
        parallel_lf_output = parallel_applicator.apply_lfs(
            examples=examples,
            labeling_functions=labeling_functions
        )
        
        parallel_time = time.time() - start_time
        
        print(f"   ✓ Parallel processing completed in {parallel_time:.3f} seconds")
        print(f"   Vote matrix shape: {parallel_lf_output.votes.shape}")
        
        # Compare with sequential processing
        start_time = time.time()
        
        sequential_votes = np.full((len(examples), len(labeling_functions)), -1)
        for i, example in enumerate(examples):
            for j, lf in enumerate(labeling_functions):
                try:
                    vote = lf(example)
                    sequential_votes[i, j] = vote if vote is not None else -1
                except:
                    sequential_votes[i, j] = -1
        
        sequential_time = time.time() - start_time
        
        print(f"   Sequential processing: {sequential_time:.3f} seconds")
        
        if parallel_time < sequential_time:
            speedup = sequential_time / parallel_time
            print(f"   ✓ Speedup: {speedup:.2f}x faster with parallel processing")
        else:
            print("   Note: Sequential was faster (expected for small datasets)")
        
    except Exception as e:
        print(f"Optimization demo failed: {e}")


def main():
    """Run the complete v0.2.0 feature demonstration."""
    print("🚀 LabelForge v0.2.0 Advanced Features Demo")
    print("=" * 50)
    
    # Create demo data
    examples, true_labels = create_demo_data()
    labeling_functions = create_labeling_functions()
    
    # Basic workflow
    label_model, lf_output, predictions, probabilities = demo_basic_workflow(
        examples, labeling_functions, true_labels
    )
    
    # Advanced features demonstrations
    demo_advanced_analytics(label_model, lf_output, predictions, probabilities)
    demo_research_tools(examples, lf_output, true_labels)
    demo_ml_integrations(label_model, examples, lf_output)
    demo_optimization_features(examples, labeling_functions)
    
    print("\\n" + "=" * 50)
    print("🎉 Demo completed! LabelForge v0.2.0 features showcased.")
    print("\\nKey improvements in v0.2.0:")
    print("• 🔬 Advanced model diagnostics & uncertainty quantification")
    print("• 🤖 Enhanced ML ecosystem integrations") 
    print("• 📊 Comprehensive research & benchmarking tools")
    print("• ⚡ Performance optimization & parallel processing")
    print("• 📝 Publication-ready exports & reproducibility tools")


if __name__ == "__main__":
    main()

"""
LabelForge v0.2.0 Analytics Demo

This script demonstrates the new advanced analytics capabilities
introduced in v0.2.0, including uncertainty quantification, 
model interpretability, and ML ecosystem integrations.
"""

import numpy as np
import pandas as pd
from typing import List

# Import LabelForge core
from labelforge import Example, LabelModel, lf, apply_lfs
from labelforge.lf import get_registered_lfs

# Import new v0.2.0 analytics modules
from labelforge.analytics import (
    UncertaintyQuantifier, 
    CalibrationAnalyzer,
    ModelAnalyzer, 
    LFImportanceAnalyzer,
    AdvancedEvaluator,
    CrossValidator
)

# Import ML integrations (optional, only if packages installed)
try:
    from labelforge.integrations import PyTorchExporter, HuggingFaceExporter
    HAS_INTEGRATIONS = True
except ImportError:
    print("⚠️  ML integrations not available (PyTorch/HuggingFace not installed)")
    HAS_INTEGRATIONS = False

try:
    from labelforge.integrations import MLflowTracker
    HAS_MLFLOW = True
except ImportError:
    print("⚠️  MLflow not available")
    HAS_MLFLOW = False


def create_sample_data():
    """Create sample data for demonstration."""
    print("📊 Creating sample dataset...")
    
    # Sample text data for sentiment analysis
    examples = [
        Example("This movie is absolutely fantastic! I loved every minute.", metadata={"source": "review"}),
        Example("Terrible film, waste of time and money.", metadata={"source": "review"}),
        Example("Great acting and beautiful cinematography.", metadata={"source": "review"}),
        Example("Boring and predictable plot.", metadata={"source": "review"}),
        Example("Amazing soundtrack and visual effects!", metadata={"source": "review"}),
        Example("Could not finish watching, too slow.", metadata={"source": "review"}),
        Example("Best movie I've seen this year.", metadata={"source": "review"}),
        Example("Not worth the hype, disappointing.", metadata={"source": "review"}),
        Example("Excellent storytelling and character development.", metadata={"source": "review"}),
        Example("Poor direction and weak script.", metadata={"source": "review"}),
        Example("Masterpiece of modern cinema.", metadata={"source": "review"}),
        Example("Average movie, nothing special.", metadata={"source": "review"}),
        Example("Incredible performances by all actors.", metadata={"source": "review"}),
        Example("Confusing plot that makes no sense.", metadata={"source": "review"}),
        Example("Highly recommended for all audiences.", metadata={"source": "review"}),
        # Add more neutral examples
        Example("The movie was okay, had some good moments.", metadata={"source": "review"}),
        Example("Decent film with mixed reviews.", metadata={"source": "review"}),
        Example("Standard Hollywood production.", metadata={"source": "review"}),
        Example("Watchable but forgettable.", metadata={"source": "review"}),
        Example("Some good scenes, some not so much.", metadata={"source": "review"}),
    ]
    
    print(f"✅ Created {len(examples)} sample examples")
    return examples


def define_labeling_functions():
    """Define sample labeling functions for sentiment analysis."""
    print("🏷️  Defining labeling functions...")
    
    # Clear any existing LFs
    from labelforge.lf import clear_lf_registry
    clear_lf_registry()
    
    @lf(name="positive_words")
    def lf_positive_words(example: Example) -> int:
        """Label positive if contains positive words."""
        positive_words = ["great", "amazing", "excellent", "fantastic", "best", "love", "incredible", "masterpiece"]
        text_lower = example.text.lower()
        if any(word in text_lower for word in positive_words):
            return 1  # Positive
        return -1  # Abstain
    
    @lf(name="negative_words")
    def lf_negative_words(example: Example) -> int:
        """Label negative if contains negative words."""
        negative_words = ["terrible", "awful", "boring", "disappointing", "waste", "poor", "bad", "horrible"]
        text_lower = example.text.lower()
        if any(word in text_lower for word in negative_words):
            return 0  # Negative
        return -1  # Abstain
    
    @lf(name="exclamation_positive")
    def lf_exclamation_positive(example: Example) -> int:
        """Label positive if has exclamation and positive sentiment."""
        if "!" in example.text:
            positive_indicators = ["great", "amazing", "love", "fantastic", "excellent"]
            if any(word in example.text.lower() for word in positive_indicators):
                return 1  # Positive
        return -1  # Abstain
    
    @lf(name="length_sentiment")
    def lf_length_sentiment(example: Example) -> int:
        """Longer reviews tend to be more extreme."""
        if len(example.text.split()) > 8:
            # Longer reviews, check for sentiment indicators
            if any(word in example.text.lower() for word in ["great", "amazing", "excellent", "best"]):
                return 1  # Positive
            elif any(word in example.text.lower() for word in ["terrible", "awful", "worst", "hate"]):
                return 0  # Negative
        return -1  # Abstain
    
    @lf(name="recommendation")
    def lf_recommendation(example: Example) -> int:
        """Label based on recommendation language."""
        text_lower = example.text.lower()
        if "recommend" in text_lower or "must watch" in text_lower:
            return 1  # Positive
        elif "not worth" in text_lower or "avoid" in text_lower:
            return 0  # Negative
        return -1  # Abstain
    
    registered_lfs = get_registered_lfs()
    print(f"✅ Defined {len(registered_lfs)} labeling functions:")
    for lf_func in registered_lfs:
        print(f"   • {lf_func.name}")
    
    return registered_lfs


def demonstrate_uncertainty_quantification(model, lf_output, examples):
    """Demonstrate uncertainty quantification features."""
    print("\\n🎯 Demonstrating Uncertainty Quantification...")
    
    # Initialize uncertainty quantifier
    uncertainty_quantifier = UncertaintyQuantifier(model)
    
    # Test different uncertainty methods
    methods = ["bootstrap", "ensemble"]
    
    for method in methods:
        print(f"\\n📊 Testing {method} uncertainty estimation...")
        
        try:
            predictions, probabilities, lower_bounds, upper_bounds = uncertainty_quantifier.predict_with_uncertainty(
                lf_output, method=method, n_samples=20  # Reduced for demo speed
            )
            
            # Calculate uncertainty metrics
            max_probs = np.max(probabilities, axis=1)
            uncertainty_width = upper_bounds[np.arange(len(predictions)), predictions] - \
                               lower_bounds[np.arange(len(predictions)), predictions]
            
            print(f"   ✅ {method.title()} method completed")
            print(f"   📈 Mean confidence: {np.mean(max_probs):.3f}")
            print(f"   📏 Mean uncertainty width: {np.mean(uncertainty_width):.3f}")
            print(f"   🎪 High uncertainty examples: {np.sum(uncertainty_width > np.percentile(uncertainty_width, 75))}")
            
            # Show most uncertain examples
            most_uncertain_idx = np.argmax(uncertainty_width)
            print(f"   🔍 Most uncertain: '{examples[most_uncertain_idx].text[:50]}...' (width: {uncertainty_width[most_uncertain_idx]:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error with {method}: {e}")
    
    # Calibration analysis
    print("\\n📏 Testing calibration analysis...")
    try:
        calibration_analyzer = CalibrationAnalyzer()
        probabilities = model.predict_proba(lf_output)
        
        calibration_data = calibration_analyzer.analyze_calibration(probabilities, n_bins=5)
        
        print("   ✅ Calibration analysis completed")
        print(f"   📊 Confidence distribution computed with {calibration_data['n_bins']} bins")
        
    except Exception as e:
        print(f"   ❌ Error in calibration analysis: {e}")


def demonstrate_interpretability(model, lf_output, examples):
    """Demonstrate model interpretability features."""
    print("\\n🔍 Demonstrating Model Interpretability...")
    
    # Model analyzer
    model_analyzer = ModelAnalyzer(model)
    
    print("📊 Analyzing LF interactions...")
    try:
        lf_analysis = model_analyzer.analyze_lf_interactions(lf_output)
        
        print("   ✅ LF interaction analysis completed")
        
        # Coverage analysis
        coverage_df = lf_analysis['coverage']
        print(f"   📈 Coverage analysis:")
        print(f"      • Mean coverage: {coverage_df['coverage'].mean():.1%}")
        print(f"      • Best performing LF: {coverage_df.loc[coverage_df['coverage'].idxmax(), 'lf_name']} ({coverage_df['coverage'].max():.1%})")
        
        # Agreement analysis
        agreements = lf_analysis['agreements']
        avg_agreement = agreements['avg_pairwise_agreement']
        print(f"      • Average pairwise agreement: {avg_agreement:.1%}")
        
        # Conflicts
        conflicts_df = lf_analysis['conflicts']
        print(f"      • Examples with conflicts: {len(conflicts_df)}")
        
        if len(conflicts_df) > 0:
            print("      • Example conflict:")
            conflict_example = conflicts_df.iloc[0]
            example_text = examples[conflict_example['example_idx']].text
            print(f"        '{example_text[:50]}...' - {conflict_example['conflicting_lfs']}")
        
    except Exception as e:
        print(f"   ❌ Error in LF interaction analysis: {e}")
    
    # LF Importance Analysis
    print("\\n🎯 Analyzing LF importance...")
    try:
        importance_analyzer = LFImportanceAnalyzer(model)
        
        for method in ["permutation", "ablation"]:
            print(f"   📊 Testing {method} importance...")
            try:
                importance_df = importance_analyzer.calculate_lf_importance(lf_output, method=method)
                
                print(f"      ✅ {method.title()} importance analysis completed")
                print(f"      🏆 Most important LF: {importance_df.iloc[0]['lf_name']} (score: {importance_df.iloc[0]['importance_score']:.4f})")
                print(f"      📊 Top 3 LFs:")
                
                for i in range(min(3, len(importance_df))):
                    row = importance_df.iloc[i]
                    print(f"         {i+1}. {row['lf_name']}: {row['importance_score']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error with {method}: {e}")
                
    except Exception as e:
        print(f"   ❌ Error in LF importance analysis: {e}")


def demonstrate_advanced_evaluation(model, lf_output, examples):
    """Demonstrate advanced evaluation features."""
    print("\\n⚖️  Demonstrating Advanced Evaluation...")
    
    # Comprehensive evaluation
    evaluator = AdvancedEvaluator()
    
    print("📋 Running comprehensive evaluation...")
    try:
        eval_results = evaluator.evaluate_comprehensive(model, lf_output, examples=examples)
        
        print("   ✅ Comprehensive evaluation completed")
        
        # Model statistics
        model_stats = eval_results['model_stats']
        print(f"   📊 Model Statistics:")
        print(f"      • Examples: {model_stats['n_examples']}")
        print(f"      • Classes: {model_stats['n_classes']}")
        print(f"      • Mean confidence: {model_stats['mean_confidence']:.3f}")
        print(f"      • Mean entropy: {model_stats['mean_entropy']:.3f}")
        print(f"      • High confidence examples: {model_stats['high_confidence_examples']}")
        
        # Weak supervision metrics
        ws_metrics = eval_results['weak_supervision_metrics']
        print(f"   🏷️  Weak Supervision Metrics:")
        print(f"      • Total coverage: {ws_metrics['total_coverage']:.1%}")
        print(f"      • Conflict rate: {ws_metrics['conflict_rate']:.1%}")
        print(f"      • Average agreement: {ws_metrics['avg_pairwise_agreement']:.1%}")
        
        # Coverage analysis
        coverage_analysis = eval_results['coverage_analysis']
        print(f"   📈 Coverage Analysis:")
        print(f"      • Uncovered examples: {coverage_analysis['uncovered_examples']}")
        print(f"      • Mean coverage per example: {coverage_analysis['mean_coverage_per_example']:.1f}")
        
    except Exception as e:
        print(f"   ❌ Error in comprehensive evaluation: {e}")
    
    # Cross-validation
    print("\\n🔄 Running cross-validation...")
    try:
        cross_validator = CrossValidator(cv_folds=3, random_state=42)  # Reduced folds for demo
        
        cv_results = cross_validator.cross_validate_ws(
            lf_output, examples, 
            model_params={'cardinality': model.cardinality}
        )
        
        print("   ✅ Cross-validation completed")
        print(f"   📊 CV Results:")
        
        # Extract available metrics
        for metric in ['total_coverage', 'conflict_rate', 'avg_pairwise_agreement']:
            if f'{metric}_mean' in cv_results:
                mean_val = cv_results[f'{metric}_mean']
                std_val = cv_results[f'{metric}_std']
                display_name = metric.replace('_', ' ').title()
                print(f"      • {display_name}: {mean_val:.3f} ± {std_val:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in cross-validation: {e}")


def demonstrate_ml_integrations(model, lf_output, examples):
    """Demonstrate ML ecosystem integrations."""
    if not HAS_INTEGRATIONS:
        print("\\n⚠️  Skipping ML integrations demo (dependencies not installed)")
        return
    
    print("\\n🔧 Demonstrating ML Ecosystem Integrations...")
    
    # PyTorch integration
    print("🔥 Testing PyTorch integration...")
    try:
        pytorch_exporter = PyTorchExporter(model)
        
        # Create PyTorch dataset
        dataset = pytorch_exporter.to_pytorch_dataset(examples, lf_output)
        print(f"   ✅ Created PyTorch dataset with {len(dataset)} examples")
        
        # Export training data
        training_data = pytorch_exporter.export_training_data(
            examples, lf_output, confidence_threshold=0.7
        )
        print(f"   📊 Exported {training_data['n_examples']} high-confidence examples")
        print(f"   📈 Coverage rate: {training_data['coverage_rate']:.1%}")
        
        # Export model weights
        weights = pytorch_exporter.export_model_weights()
        print(f"   🏋️  Exported model weights: {list(weights.keys())}")
        
    except Exception as e:
        print(f"   ❌ Error in PyTorch integration: {e}")
    
    # Hugging Face integration
    print("\\n🤗 Testing Hugging Face integration...")
    try:
        hf_exporter = HuggingFaceExporter(model)
        
        # Create HF dataset
        hf_dataset = hf_exporter.to_hf_dataset(examples, lf_output)
        print(f"   ✅ Created HuggingFace dataset with {len(hf_dataset)} examples")
        print(f"   📋 Columns: {list(hf_dataset.column_names)}")
        
        # Export for inference
        inference_data = hf_exporter.export_for_inference(
            examples, lf_output, confidence_threshold=0.8
        )
        print(f"   🎯 Exported {len(inference_data)} examples for inference")
        
    except Exception as e:
        print(f"   ❌ Error in Hugging Face integration: {e}")


def demonstrate_experiment_tracking(model, lf_output, examples):
    """Demonstrate experiment tracking."""
    if not HAS_MLFLOW:
        print("\\n⚠️  Skipping MLflow demo (MLflow not installed)")
        return
        
    print("\\n📊 Demonstrating Experiment Tracking with MLflow...")
    
    try:
        # Initialize MLflow tracker
        mlflow_tracker = MLflowTracker(experiment_name="labelforge_v2_demo")
        
        # Start run
        mlflow_tracker.start_run(run_name="analytics_demo")
        
        print("   ✅ Started MLflow run")
        
        # Log experiment config
        registered_lfs = get_registered_lfs()
        lf_names = [lf_func.name for lf_func in registered_lfs]
        
        mlflow_tracker.log_experiment_config(
            examples, lf_names, 
            model_params={'cardinality': model.cardinality}
        )
        print("   📋 Logged experiment configuration")
        
        # Log LF statistics
        mlflow_tracker.log_lf_statistics(lf_output, examples)
        print("   🏷️  Logged labeling function statistics")
        
        # Log model training info
        mlflow_tracker.log_model_training(model, lf_output)
        print("   🏋️  Logged model training information")
        
        # Log predictions
        mlflow_tracker.log_predictions(model, lf_output, examples)
        print("   🎯 Logged model predictions")
        
        # End run
        mlflow_tracker.end_run()
        print("   ✅ MLflow tracking completed")
        
        print(f"   🌐 View results at MLflow UI: mlflow ui")
        
    except Exception as e:
        print(f"   ❌ Error in MLflow tracking: {e}")


def main():
    """Main demonstration function."""
    print("🚀 LabelForge v0.2.0 Analytics Demo")
    print("=" * 50)
    
    # Create sample data
    examples = create_sample_data()
    
    # Define labeling functions
    registered_lfs = define_labeling_functions()
    
    # Apply labeling functions
    print("\\n🔄 Applying labeling functions...")
    lf_output = apply_lfs(registered_lfs, examples)
    print(f"✅ Applied {len(registered_lfs)} LFs to {len(examples)} examples")
    
    # Train label model
    print("\\n🏋️  Training label model...")
    model = LabelModel(cardinality=2)  # Binary sentiment classification
    model.fit(lf_output, verbose=True)
    print("✅ Label model training completed")
    
    # Get basic predictions
    predictions = model.predict(lf_output)
    probabilities = model.predict_proba(lf_output)
    
    print(f"\\n📊 Basic Results:")
    print(f"   • Predictions: {np.bincount(predictions)}")
    print(f"   • Mean confidence: {np.max(probabilities, axis=1).mean():.3f}")
    
    # Demonstrate new v0.2.0 features
    try:
        demonstrate_uncertainty_quantification(model, lf_output, examples)
    except Exception as e:
        print(f"\\n❌ Error in uncertainty demo: {e}")
    
    try:
        demonstrate_interpretability(model, lf_output, examples)
    except Exception as e:
        print(f"\\n❌ Error in interpretability demo: {e}")
    
    try:
        demonstrate_advanced_evaluation(model, lf_output, examples)
    except Exception as e:
        print(f"\\n❌ Error in evaluation demo: {e}")
    
    try:
        demonstrate_ml_integrations(model, lf_output, examples)
    except Exception as e:
        print(f"\\n❌ Error in ML integrations demo: {e}")
    
    try:
        demonstrate_experiment_tracking(model, lf_output, examples)
    except Exception as e:
        print(f"\\n❌ Error in experiment tracking demo: {e}")
    
    print("\\n🎉 LabelForge v0.2.0 Analytics Demo Complete!")
    print("\\n💡 Next Steps:")
    print("   • Try the enhanced web interface: ./start_web.sh")
    print("   • Explore ML integrations with your own models")
    print("   • Set up experiment tracking for your research")
    print("   • Check the updated documentation for detailed usage")


if __name__ == "__main__":
    main()

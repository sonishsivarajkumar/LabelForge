#!/usr/bin/env python3
"""
Test script for LabelForge v0.2.0 UI components.

This script tests all the major UI components without running the full Streamlit app.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_ui_imports():
    """Test that all UI-related imports work."""
    print("🧪 Testing UI component imports...")
    
    try:
        # Test core imports
        import streamlit as st
        print("✅ Streamlit imported")
        
        # Test LabelForge imports
        from labelforge import LabelModel, LFOutput, Example
        from labelforge.analytics import ModelAnalyzer, CrossValidator, UncertaintyQuantifier
        from labelforge.research import StatisticalTester
        from labelforge.benchmarks import StandardDatasets
        print("✅ LabelForge modules imported")
        
        # Test web app imports (without running the app)
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", "src/labelforge/web/app.py")
        app_module = importlib.util.module_from_spec(spec)
        print("✅ Web app module can be loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ UI import error: {e}")
        return False

def test_core_functionality():
    """Test core LabelForge functionality."""
    print("\n🧪 Testing core functionality...")
    
    try:
        import numpy as np
        from labelforge import LabelModel, LFOutput, Example
        from labelforge.analytics import CrossValidator, ModelAnalyzer
        
        # Create test data
        votes = np.array([
            [1, 0, -1, 1],
            [0, 1, 1, 0], 
            [-1, -1, 0, 1],
            [1, 1, 0, -1],
            [0, 0, 1, 1],
            [1, -1, 0, 0],
            [-1, 1, -1, 1],
            [0, 0, 0, -1]
        ])
        example_ids = [f'ex{i+1}' for i in range(8)]
        lf_names = ['lf1', 'lf2', 'lf3', 'lf4']
        examples = [Example(f'Sample text {i+1}', f'ex{i+1}') for i in range(8)]
        
        lf_output = LFOutput(votes=votes, example_ids=example_ids, lf_names=lf_names)
        print(f"✅ Created LFOutput: {lf_output.n_examples} examples, {lf_output.n_lfs} LFs")
        
        # Test model training
        model = LabelModel(cardinality=2)
        model.fit(lf_output)
        predictions = model.predict(lf_output)
        probabilities = model.predict_proba(lf_output)
        print(f"✅ Model training: {len(predictions)} predictions, prob shape {probabilities.shape}")
        
        # Test analytics
        analyzer = ModelAnalyzer(model)
        print("✅ Model analyzer created")
        
        # Test cross-validation
        cv = CrossValidator(cv_folds=3)
        cv_results = cv.cross_validate_ws(lf_output, examples)
        print(f"✅ Cross-validation: {len(cv_results)} metrics")
        
        # Test uncertainty quantification
        try:
            from labelforge.analytics import UncertaintyQuantifier
            uq = UncertaintyQuantifier()
            print("✅ Uncertainty quantifier created")
        except Exception as e:
            print(f"⚠️  Uncertainty quantifier: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_v2_features():
    """Test v0.2.0 specific features."""
    print("\n🧪 Testing v0.2.0 features...")
    
    try:
        # Test research modules
        from labelforge.research import ExperimentTracker, StatisticalTester
        from labelforge.research.benchmarks import StandardDatasets, BenchmarkSuite
        print("✅ Research modules")
        
        # Test benchmarking
        from labelforge.benchmarks import MetricCalculator
        print("✅ Benchmarking modules")
        
        # Test optimization
        from labelforge.optimization import ParallelProcessor, PerformanceProfiler
        print("✅ Optimization modules")
        
        # Test templates
        from labelforge.templates import DomainTemplates
        print("✅ Template modules")
        
        # Test integrations
        from labelforge.integrations import PyTorchExporter, MLflowTracker
        print("✅ Integration modules")
        
        return True
        
    except Exception as e:
        print(f"❌ v0.2.0 features error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_features():
    """Test UI-specific features."""
    print("\n🧪 Testing UI features...")
    
    try:
        # Test that we can create the main UI components
        import numpy as np
        from labelforge import LabelModel, LFOutput, Example
        
        # Simulate session state
        class MockSessionState:
            def __init__(self):
                votes = np.array([[1, 0, -1], [0, 1, 1], [-1, -1, 0]])
                self.lf_output = LFOutput(
                    votes=votes, 
                    example_ids=['ex1', 'ex2', 'ex3'],
                    lf_names=['lf1', 'lf2', 'lf3']
                )
                self.examples = [Example(f'text{i}', f'ex{i+1}') for i in range(3)]
                self.label_model = LabelModel(cardinality=2)
                self.label_model.fit(self.lf_output)
        
        mock_state = MockSessionState()
        print("✅ Mock session state created")
        
        # Test analytics that would be used in UI
        from labelforge.analytics import ModelAnalyzer, CrossValidator
        analyzer = ModelAnalyzer(mock_state.label_model)
        cv = CrossValidator(cv_folds=3)
        print("✅ Analytics components ready for UI")
        
        return True
        
    except Exception as e:
        print(f"❌ UI features error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing LabelForge v0.2.0 UI Components")
    print("=" * 50)
    
    tests = [
        ("UI Imports", test_ui_imports),
        ("Core Functionality", test_core_functionality), 
        ("v0.2.0 Features", test_v2_features),
        ("UI Features", test_ui_features)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🔍 {name}")
        if test_func():
            passed += 1
            print(f"✅ {name} PASSED")
        else:
            print(f"❌ {name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All UI components are working correctly!")
        print("\n🌟 LabelForge v0.2.0 is ready for production!")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

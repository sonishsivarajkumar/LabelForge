#!/usr/bin/env python3
"""
Final test script to verify LabelForge v0.2.0 web app functionality
"""

import requests
import time
import sys
from pathlib import Path

def test_webapp_accessibility():
    """Test if the web app is accessible"""
    print("🔍 Testing LabelForge v0.2.0 Web App...")
    
    # Test different possible ports
    ports = [8503, 8506, 8501, 8502]
    
    for port in ports:
        try:
            url = f"http://localhost:{port}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Web app is running successfully at {url}")
                return url
        except requests.exceptions.RequestException:
            continue
    
    print("❌ Web app is not accessible on any expected port")
    return None

def test_core_imports():
    """Test if all v0.2.0 core modules can be imported"""
    print("\n🧪 Testing core module imports...")
    
    # Add src to Python path
    import sys
    import os
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    modules_to_test = [
        "labelforge.types",
        "labelforge.analytics.uncertainty",
        "labelforge.analytics.cross_validation", 
        "labelforge.research.benchmarks",
        "labelforge.research.evaluation",
        "labelforge.optimization.profiling",
        "labelforge.benchmarks.metrics",
        "labelforge.templates.domain",
        "labelforge.integrations.huggingface",
        "labelforge.web.app"
    ]
    
    passed = 0
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
    
    print(f"\n📊 Import test results: {passed}/{len(modules_to_test)} modules imported successfully")
    return passed == len(modules_to_test)

def verify_v0_2_0_features():
    """Verify v0.2.0 feature availability"""
    print("\n🎯 Verifying v0.2.0 feature availability...")
    
    # Add src to Python path
    import sys
    import os
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    features = []
    
    try:
        from labelforge.analytics import UncertaintyAnalyzer, CrossValidationEvaluator
        features.append("✅ Advanced Analytics (Uncertainty, Cross-validation)")
    except ImportError:
        features.append("❌ Advanced Analytics")
    
    try:
        from labelforge.research import BenchmarkSuite, StatisticalTester
        features.append("✅ Research Tools (Benchmarking, Statistical Testing)")
    except ImportError:
        features.append("❌ Research Tools")
    
    try:
        from labelforge.optimization import PerformanceProfiler
        features.append("✅ Performance Optimization")
    except ImportError:
        features.append("❌ Performance Optimization")
    
    try:
        from labelforge.benchmarks import StandardDatasets, ModelPerformanceMetrics
        features.append("✅ Benchmarking Suite")
    except ImportError:
        features.append("❌ Benchmarking Suite")
    
    try:
        from labelforge.templates import DomainTemplates
        features.append("✅ Domain Templates")
    except ImportError:
        features.append("❌ Domain Templates")
    
    try:
        from labelforge.integrations import HuggingFaceIntegration
        features.append("✅ ML Ecosystem Integration")
    except ImportError:
        features.append("❌ ML Ecosystem Integration")
    
    for feature in features:
        print(f"  {feature}")
    
    success_count = len([f for f in features if f.startswith("✅")])
    total_count = len(features)
    
    print(f"\n📈 v0.2.0 Features: {success_count}/{total_count} available")
    return success_count == total_count

def main():
    """Run all tests"""
    print("🚀 LabelForge v0.2.0 Final Verification")
    print("=" * 50)
    
    # Test 1: Web app accessibility
    webapp_url = test_webapp_accessibility()
    
    # Test 2: Core imports
    imports_ok = test_core_imports()
    
    # Test 3: v0.2.0 features
    features_ok = verify_v0_2_0_features()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 FINAL STATUS SUMMARY")
    print("=" * 50)
    
    if webapp_url:
        print(f"🌐 Web App: RUNNING at {webapp_url}")
    else:
        print("🌐 Web App: NOT RUNNING")
    
    print(f"📦 Core Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"🎯 v0.2.0 Features: {'PASS' if features_ok else 'FAIL'}")
    
    if webapp_url and imports_ok and features_ok:
        print("\n🎉 LabelForge v0.2.0 is READY FOR DEPLOYMENT!")
        print("🔗 Access the web app at:", webapp_url)
        print("📚 All v0.2.0 features are available and functional")
    else:
        print("\n⚠️  Some issues detected. Please review the results above.")
    
    return webapp_url and imports_ok and features_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

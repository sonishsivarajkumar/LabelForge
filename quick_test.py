#!/usr/bin/env python3
import sys
import os

# Add src to Python path
src_path = os.path.join(os.getcwd(), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("🚀 LabelForge v0.2.0 Quick Test")
print("=" * 40)

try:
    import labelforge
    print("✅ Core LabelForge imported")
except ImportError as e:
    print(f"❌ Core LabelForge: {e}")

try:
    from labelforge.analytics import UncertaintyAnalyzer
    print("✅ Analytics module imported")
except ImportError as e:
    print(f"❌ Analytics: {e}")

try:
    from labelforge.research import BenchmarkSuite
    print("✅ Research module imported")
except ImportError as e:
    print(f"❌ Research: {e}")

try:
    from labelforge.optimization import PerformanceProfiler
    print("✅ Optimization module imported")
except ImportError as e:
    print(f"❌ Optimization: {e}")

try:
    from labelforge.benchmarks import StandardDatasets
    print("✅ Benchmarks module imported")
except ImportError as e:
    print(f"❌ Benchmarks: {e}")

try:
    from labelforge.templates import DomainTemplates
    print("✅ Templates module imported")
except ImportError as e:
    print(f"❌ Templates: {e}")

print("=" * 40)
print("🎉 LabelForge v0.2.0 module test complete!")
print(f"🌐 Web app running at: http://localhost:8503")

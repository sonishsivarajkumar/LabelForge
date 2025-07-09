#!/usr/bin/env python3
"""
Debug script to test LF registration and application.
"""

import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from labelforge import lf, load_example_data
from labelforge.lf import apply_lfs, LF_REGISTRY
from labelforge.types import Example

print("🔍 Debugging LabelForge LF Registration...")

# Clear any existing LFs
from labelforge.lf import clear_lf_registry
clear_lf_registry()

# Test simple LF
@lf(name="test_diabetes")
def test_lf(example: Example) -> int:
    """Simple test LF."""
    return 1 if "diabetes" in example.text.lower() else 0

print(f"LF Registry after definition: {list(LF_REGISTRY.keys())}")
print(f"LF object type: {type(LF_REGISTRY['test_diabetes'])}")
print(f"LF object: {LF_REGISTRY['test_diabetes']}")

# Test application
examples = load_example_data("medical_texts")
print(f"\nTesting LF on first example: '{examples[0].text}'")

lf_obj = LF_REGISTRY['test_diabetes']
result = lf_obj(examples[0])
print(f"LF result: {result} (type: {type(result)})")

# Test apply method
try:
    results = lf_obj.apply(examples[:3])
    print(f"Apply method results: {results}")
except Exception as e:
    print(f"Error in apply method: {e}")
    
# Test manual loop
manual_results = []
for ex in examples[:3]:
    try:
        res = lf_obj(ex)
        manual_results.append(res)
        print(f"Manual call on '{ex.text[:30]}...': {res} (type: {type(res)})")
    except Exception as e:
        print(f"Error calling LF: {e}")

print(f"Manual results: {manual_results}")

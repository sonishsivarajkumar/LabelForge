#!/usr/bin/env python3
"""
LabelForge Example: Medical Text Classification

This example demonstrates the complete LabelForge workflow:
1. Define labeling functions for medical text classification
2. Apply LFs to generate weak supervision signals
3. Train a probabilistic label model to combine LF outputs
4. Evaluate the results and analyze LF performance

The task: Classify medical texts as mentioning diabetes (1) or not (0).
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).parent / "src"))

from labelforge import lf, LabelModel, load_example_data
from labelforge.lf import apply_lfs, clear_lf_registry
from labelforge.types import Example

print("🔨 LabelForge Example: Medical Text Classification")
print("=" * 60)

# Clear any existing LFs
clear_lf_registry()

# Step 1: Define Labeling Functions
print("\n📝 Step 1: Defining Labeling Functions...")

@lf(name="mentions_diabetes", tags={"type": "keyword", "domain": "medical"})
def lf_mentions_diabetes(example: Example) -> int:
    """Returns 1 if text explicitly mentions 'diabetes'."""
    return 1 if "diabetes" in example.text.lower() else 0

@lf(name="diabetes_medication", tags={"type": "keyword", "domain": "medical"})
def lf_diabetes_medication(example: Example) -> int:
    """Returns 1 if text mentions diabetes-related terms."""
    diabetes_terms = ["glucose", "blood sugar", "diabetic", "insulin", "diabetes medication"]
    text_lower = example.text.lower()
    return 1 if any(term in text_lower for term in diabetes_terms) else 0

@lf(name="diabetes_symptoms", tags={"type": "medical_knowledge", "domain": "medical"})
def lf_diabetes_symptoms(example: Example) -> int:
    """Returns 1 if text mentions diabetes symptoms or complications."""
    symptoms = ["neuropathy", "diabetic complications", "blood glucose", "metabolic disorder"]
    text_lower = example.text.lower()
    return 1 if any(symptom in text_lower for symptom in symptoms) else 0

@lf(name="negative_diabetes", tags={"type": "negation", "domain": "medical"})
def lf_negative_diabetes(example: Example) -> int:
    """Returns 0 if text explicitly negates diabetes."""
    negative_patterns = ["no diabetes", "no signs of diabetes", "diabetes-free", "not diabetic"]
    text_lower = example.text.lower()
    return 0 if any(pattern in text_lower for pattern in negative_patterns) else -1  # abstain

@lf(name="diabetes_management", tags={"type": "treatment", "domain": "medical"})
def lf_diabetes_management(example: Example) -> int:
    """Returns 1 if text mentions diabetes management or monitoring."""
    management_terms = ["diabetes management", "diabetes screening", "managing diabetes", "diabetes check"]
    text_lower = example.text.lower()
    return 1 if any(term in text_lower for term in management_terms) else 0

print(f"✅ Defined 5 labeling functions")

# Step 2: Load Example Data
print("\n📊 Step 2: Loading Example Data...")
examples = load_example_data("medical_texts")
print(f"✅ Loaded {len(examples)} examples")

# Show some examples
print("\n📖 Sample texts:")
for i, ex in enumerate(examples[:3]):
    print(f"  {i+1}. {ex.text}")

# Step 3: Apply Labeling Functions
print("\n🔄 Step 3: Applying Labeling Functions...")
lf_output = apply_lfs(examples)

print(f"✅ Applied {lf_output.n_lfs} LFs to {lf_output.n_examples} examples")
print(f"Vote matrix shape: {lf_output.votes.shape}")

# Analyze LF statistics
print("\n📈 LF Performance Analysis:")
print(f"{'LF Name':<25} {'Coverage':<10} {'# Votes':<10}")
print("-" * 45)

coverage = lf_output.coverage()
for i, lf_name in enumerate(lf_output.lf_names):
    n_votes = np.sum(lf_output.votes[:, i] != -1)  # Non-abstain votes
    print(f"{lf_name:<25} {coverage[i]:.2f}       {n_votes:<10}")

# Step 4: Analyze LF Conflicts and Overlaps
print("\n🔍 Step 4: Analyzing LF Interactions...")
overlap_matrix = lf_output.overlap()
conflict_matrix = lf_output.conflict()

print("\nLF Overlap Matrix (fraction of examples where both LFs vote):")
print("LF Names:", lf_output.lf_names)
for i, row in enumerate(overlap_matrix):
    print(f"{lf_output.lf_names[i][:15]:<15}: {[f'{x:.2f}' for x in row]}")

print("\nLF Conflict Matrix (fraction of overlapping votes that disagree):")
for i, row in enumerate(conflict_matrix):
    print(f"{lf_output.lf_names[i][:15]:<15}: {[f'{x:.2f}' for x in row]}")

# Step 5: Train Label Model
print("\n🧠 Step 5: Training Probabilistic Label Model...")
label_model = LabelModel(cardinality=2, max_iter=50, verbose=True)
label_model.fit(lf_output)

print(f"✅ Model converged: {label_model.history_['converged']}")
print(f"Training iterations: {label_model.history_['n_iter']}")
print(f"Final log-likelihood: {label_model.history_['log_likelihood'][-1]:.4f}")

# Step 6: Generate Probabilistic Labels
print("\n🏷️  Step 6: Generating Probabilistic Labels...")
soft_labels = label_model.predict_proba(lf_output)
hard_labels = label_model.predict(lf_output)

print(f"✅ Generated soft labels shape: {soft_labels.shape}")
print(f"Class distribution in predictions: {np.bincount(hard_labels)}")

# Step 7: Show Results
print("\n📋 Step 7: Results Summary...")
print(f"{'Example':<50} {'Prediction':<12} {'Confidence':<12}")
print("-" * 74)

for i in range(min(10, len(examples))):
    text = examples[i].text[:47] + "..." if len(examples[i].text) > 47 else examples[i].text
    pred = hard_labels[i]
    conf = np.max(soft_labels[i])
    pred_label = "DIABETES" if pred == 1 else "NO DIABETES"
    print(f"{text:<50} {pred_label:<12} {conf:.3f}")

# Step 8: Model Analysis
print("\n🔬 Step 8: Model Analysis...")
lf_stats = label_model.get_lf_stats()

print(f"Learned class priors: {lf_stats['class_priors']}")
print("\nLF Accuracy Analysis:")
for lf_name, stats in lf_stats['lf_accuracies'].items():
    print(f"\n{lf_name}:")
    accuracy_matrix = np.array(stats['accuracy_matrix'])
    print(f"  Accuracy when true class = 0: {accuracy_matrix[0]}")
    print(f"  Accuracy when true class = 1: {accuracy_matrix[1]}")

# Step 9: Command Line Interface Demo
print("\n🖥️  Step 9: CLI Demo...")
print("You can also use LabelForge from the command line:")
print("  labelforge --help")
print("  labelforge run --dataset medical_texts")

print("\n🎉 LabelForge Example Complete!")
print("=" * 60)
print("Key takeaways:")
print("1. Easy LF definition with @lf decorator")
print("2. Automatic LF conflict and coverage analysis")
print("3. Probabilistic label model learns LF accuracies")
print("4. End-to-end pipeline from weak supervision to labels")
print("5. Rich analytics for understanding LF performance")

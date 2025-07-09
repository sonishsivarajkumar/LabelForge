# Test example labeling functions
import sys
import os

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from labelforge import lf, LabelModel, load_example_data, apply_lfs
import numpy as np

# Example LFs for medical text classification
@lf(name="mention_diabetes", tags={"entity": "disease", "domain": "medical"})
def lf_mention_diabetes(example):
    """Detect mentions of diabetes."""
    keywords = ["diabetes", "diabetic", "blood sugar", "glucose"]
    text_lower = example.text.lower()
    return 1 if any(keyword in text_lower for keyword in keywords) else 0


@lf(name="mention_cancer", tags={"entity": "disease", "domain": "medical"})
def lf_mention_cancer(example):
    """Detect mentions of cancer."""
    keywords = ["cancer", "tumor", "malignant", "chemotherapy", "oncology"]
    text_lower = example.text.lower()
    return 1 if any(keyword in text_lower for keyword in keywords) else 0


@lf(name="no_disease_indicators", tags={"type": "negative", "domain": "medical"})
def lf_no_disease(example):
    """Detect explicitly negative disease indicators."""
    negative_phrases = ["no signs of", "within normal range", "no history of"]
    text_lower = example.text.lower()
    return 0 if any(phrase in text_lower for phrase in negative_phrases) else -1


@lf(name="family_history", tags={"type": "risk_factor", "domain": "medical"})
def lf_family_history(example):
    """Detect family history mentions."""
    if "family history" in example.text.lower():
        return 1
    return -1  # Abstain if no family history mentioned


if __name__ == "__main__":
    print("🔨 LabelForge Example - Medical Text Classification")
    print("=" * 50)
    
    # Load example data
    print("📊 Loading medical text examples...")
    examples = load_example_data("medical_texts")
    print(f"Loaded {len(examples)} examples")
    
    # Show a few examples
    print("\n📝 Sample texts:")
    for i, ex in enumerate(examples[:3]):
        print(f"  {i+1}. {ex.text}")
    
    # Apply labeling functions
    print(f"\n🏷️  Applying {len([name for name in locals() if name.startswith('lf_')])} labeling functions...")
    lf_output = apply_lfs(examples)
    
    # Show LF statistics
    print("\n📈 LF Coverage:")
    coverage = lf_output.coverage()
    for i, lf_name in enumerate(lf_output.lf_names):
        print(f"  {lf_name}: {coverage[i]:.1%}")
    
    # Show some vote examples
    print("\n🗳️  Sample LF votes:")
    print(f"{'Text':<50} {'LF Votes'}")
    print("-" * 70)
    for i in range(min(5, len(examples))):
        text = examples[i].text[:47] + "..." if len(examples[i].text) > 50 else examples[i].text
        votes = lf_output.votes[i]
        print(f"{text:<50} {votes}")
    
    # Train label model
    print("\n🧠 Training probabilistic label model...")
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lf_output)
    
    # Get predictions
    probs = label_model.predict_proba(lf_output)
    preds = label_model.predict(lf_output)
    
    print(f"\n✅ Generated predictions for {len(examples)} examples")
    
    # Show some predictions
    print("\n🎯 Sample predictions:")
    print(f"{'Text':<50} {'Pred':<6} {'Prob':<10}")
    print("-" * 70)
    for i in range(min(5, len(examples))):
        text = examples[i].text[:47] + "..." if len(examples[i].text) > 50 else examples[i].text
        pred = preds[i]
        prob = probs[i, pred]
        print(f"{text:<50} {pred:<6} {prob:.3f}")
    
    # Show class distribution
    print(f"\n📊 Predicted class distribution:")
    unique, counts = np.unique(preds, return_counts=True)
    for class_label, count in zip(unique, counts):
        print(f"  Class {class_label}: {count} examples ({count/len(preds):.1%})")
    
    print("\n🎉 Example completed successfully!")
    print("Try modifying the labeling functions or adding new ones!")

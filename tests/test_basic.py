# Basic test suite for LabelForge
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from labelforge import lf, LabelModel, load_example_data, apply_lfs, Example
from labelforge.types import ABSTAIN
from labelforge.lf import clear_lf_registry


def test_basic_lf():
    """Test basic labeling function functionality."""
    print("Testing basic LF functionality...")
    clear_lf_registry()

    @lf(name="test_lf")
    def test_keyword(example):
        return 1 if "test" in example.text.lower() else 0

    example = Example(text="This is a test example")
    result = test_keyword(example)
    assert result == 1, f"Expected 1, got {result}"

    example2 = Example(text="This is not a good example")
    result2 = test_keyword(example2)
    assert result2 == 0, f"Expected 0, got {result2}"

    print("✅ Basic LF test passed")


def test_lf_application():
    """Test applying LFs to datasets."""
    print("Testing LF application...")
    clear_lf_registry()

    @lf(name="positive_sentiment")
    def positive_lf(example):
        positive_words = ["good", "great", "excellent", "amazing"]
        return 1 if any(word in example.text.lower() for word in positive_words) else 0

    @lf(name="negative_sentiment")
    def negative_lf(example):
        negative_words = ["bad", "terrible", "awful", "horrible"]
        return 1 if any(word in example.text.lower() for word in negative_words) else 0

    examples = [
        Example(text="This movie is great!"),
        Example(text="Terrible acting and bad plot"),
        Example(text="Just an average film"),
    ]

    lf_output = apply_lfs(examples)

    assert lf_output.n_examples == 3, f"Expected 3 examples, got {lf_output.n_examples}"
    assert lf_output.n_lfs == 2, f"Expected 2 LFs, got {lf_output.n_lfs}"

    # Check votes
    assert lf_output.votes[0, 0] == 1, "First example should be positive"
    assert lf_output.votes[1, 1] == 1, "Second example should be negative"

    print("✅ LF application test passed")


def test_label_model():
    """Test label model training and prediction."""
    print("Testing label model...")
    clear_lf_registry()

    # Create simple examples
    examples = [
        Example(text="This is good"),
        Example(text="This is bad"),
        Example(text="This is great"),
        Example(text="This is terrible"),
        Example(text="This is okay"),
    ]

    @lf(name="good_words")
    def good_lf(example):
        return (
            1 if any(word in example.text.lower() for word in ["good", "great"]) else 0
        )

    @lf(name="bad_words")
    def bad_lf(example):
        return (
            1
            if any(word in example.text.lower() for word in ["bad", "terrible"])
            else 0
        )

    lf_output = apply_lfs(examples)

    # Train label model
    label_model = LabelModel(cardinality=2, max_iter=10)
    label_model.fit(lf_output)

    # Get predictions
    probs = label_model.predict_proba(lf_output)
    preds = label_model.predict(lf_output)

    assert probs.shape == (5, 2), f"Expected shape (5, 2), got {probs.shape}"
    assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"

    # Check that probabilities sum to 1
    prob_sums = np.sum(probs, axis=1)
    assert np.allclose(prob_sums, 1.0), f"Probabilities don't sum to 1: {prob_sums}"

    print("✅ Label model test passed")


def test_example_datasets():
    """Test example dataset loading."""
    print("Testing example datasets...")

    # Test medical texts
    medical_data = load_example_data("medical_texts")
    assert len(medical_data) > 0, "Medical dataset should not be empty"
    assert all(
        isinstance(ex, Example) for ex in medical_data
    ), "All items should be Examples"

    # Test sentiment data
    sentiment_data = load_example_data("sentiment")
    assert len(sentiment_data) > 0, "Sentiment dataset should not be empty"

    print("✅ Example datasets test passed")


def test_lf_stats():
    """Test LF statistics calculation."""
    print("Testing LF statistics...")
    clear_lf_registry()

    examples = [
        Example(text="keyword present"),
        Example(text="no match here"),
        Example(text="another keyword"),
        Example(text="nothing special"),
    ]

    @lf(name="keyword_lf")
    def keyword_lf(example):
        return 1 if "keyword" in example.text else 0

    lf_output = apply_lfs(examples)

    # Test coverage
    coverage = lf_output.coverage()
    assert coverage[0] == 1.0, f"Expected 100% coverage, got {coverage[0]}"

    # Test overlap and conflict (with single LF, should be identity)
    overlap = lf_output.overlap()
    conflict = lf_output.conflict()

    assert overlap.shape == (
        1,
        1,
    ), f"Expected (1, 1) overlap matrix, got {overlap.shape}"
    assert conflict.shape == (
        1,
        1,
    ), f"Expected (1, 1) conflict matrix, got {conflict.shape}"

    print("✅ LF statistics test passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running LabelForge Tests")
    print("=" * 30)

    try:
        test_basic_lf()
        test_lf_application()
        test_label_model()
        test_example_datasets()
        test_lf_stats()

        print("\n🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

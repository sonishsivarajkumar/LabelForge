# Basic test suite for LabelForge
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from labelforge import lf, LabelModel, load_example_data, apply_lfs, Example
from labelforge.lf import clear_lf_registry
from labelforge.types import ABSTAIN


def test_basic_lf():
    """Test basic labeling function functionality."""
    print("Testing basic LF functionality...")

    # Clear registry first
    clear_lf_registry()

    @lf(name="test_diabetes")
    def diabetes_lf(example: Example) -> int:
        return 1 if "diabetes" in example.text.lower() else 0

    # Test single example
    example = Example(text="Patient has diabetes")
    result = diabetes_lf(example)
    assert result == 1, f"Expected 1, got {result}"

    example2 = Example(text="Patient is healthy")
    result2 = diabetes_lf(example2)
    assert result2 == 0, f"Expected 0, got {result2}"

    print("✓ Basic LF test passed")


def test_lf_application():
    """Test applying LFs to datasets."""
    print("Testing LF application...")

    # Clear registry first
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

    # Check vote matrix
    expected_votes = np.array(
        [
            [1, 0],  # "great" -> positive=1, negative=0
            [0, 1],  # "terrible", "bad" -> positive=0, negative=1
            [0, 0],  # "average" -> positive=0, negative=0
        ]
    )
    assert np.array_equal(
        lf_output.votes, expected_votes
    ), f"Vote matrix mismatch: {lf_output.votes}"

    print("✓ LF application test passed")


def test_label_model():
    """Test label model training and prediction."""
    print("Testing label model...")

    # Clear registry first
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
    label_model = LabelModel(cardinality=2, max_iter=10, verbose=False)
    label_model.fit(lf_output)

    # Test predictions
    predictions = label_model.predict(lf_output)
    probabilities = label_model.predict_proba(lf_output)

    assert len(predictions) == 5, f"Expected 5 predictions, got {len(predictions)}"
    assert probabilities.shape == (
        5,
        2,
    ), f"Expected (5, 2) probabilities, got {probabilities.shape}"
    assert np.allclose(
        np.sum(probabilities, axis=1), 1.0
    ), "Probabilities should sum to 1"

    print("✓ Label model test passed")


def test_example_datasets():
    """Test loading example datasets."""
    print("Testing example datasets...")

    examples = load_example_data("medical_texts")
    assert len(examples) > 0, "Should load some examples"
    assert all(
        isinstance(ex, Example) for ex in examples
    ), "All items should be Examples"
    assert all(hasattr(ex, "text") for ex in examples), "All examples should have text"

    print("✓ Example datasets test passed")


def test_lf_stats():
    """Test LF statistics calculation."""
    print("Testing LF statistics...")

    # Clear registry first
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
    assert len(coverage) == 1, f"Expected 1 LF, got {len(coverage)}"
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

    print("✓ LF stats test passed")


if __name__ == "__main__":
    test_basic_lf()
    test_lf_application()
    test_label_model()
    test_example_datasets()
    test_lf_stats()
    print("All tests passed!")

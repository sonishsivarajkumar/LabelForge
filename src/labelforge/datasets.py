"""
Dataset utilities and example data loaders for LabelForge.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging

from .types import Example

logger = logging.getLogger(__name__)


def load_example_data(dataset_name: str = "medical_texts") -> List[Example]:
    """
    Load example datasets for testing and tutorials.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        List of Example objects
    """
    if dataset_name == "medical_texts":
        return _load_medical_texts()
    elif dataset_name == "sentiment":
        return _load_sentiment_data()
    elif dataset_name == "spam":
        return _load_spam_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_medical_texts() -> List[Example]:
    """Load synthetic medical text examples."""
    texts = [
        "Patient diagnosed with type 2 diabetes last month.",
        "No signs of diabetes or other metabolic disorders.",
        "Family history of diabetes and heart disease.",
        "Blood glucose levels are within normal range.",
        "Diabetes medication adjusted due to low blood sugar.",
        "Patient shows symptoms of diabetic neuropathy.",
        "Regular check-ups recommended for diabetes management.",
        "No diabetes-related complications observed.",
        "Patient has been managing diabetes for 5 years.",
        "Diabetes screening recommended for high-risk patients.",
        "Patient diagnosed with lung cancer stage 2.",
        "Cancer screening came back negative.",
        "Family history of breast cancer and ovarian cancer.",
        "No signs of cancer recurrence after treatment.",
        "Cancer treatment response has been positive.",
        "Patient undergoing chemotherapy for cancer.",
        "Regular cancer screenings are important.",
        "Cancer remission achieved after surgery.",
        "Patient concerned about cancer risk factors.",
        "Cancer support group has been helpful.",
        "Patient has high blood pressure.",
        "Normal blood pressure readings today.",
        "Hypertension medication working effectively.",
        "Patient needs to monitor blood pressure daily.",
        "No history of cardiovascular disease.",
        "Heart rate and blood pressure are stable.",
        "Patient complains of occasional headaches.",
        "Physical examination shows no abnormalities.",
        "Patient reports feeling tired lately.",
        "Overall health assessment is positive.",
    ]

    examples = []
    for i, text in enumerate(texts):
        # Add some metadata
        metadata = {
            "source": "synthetic",
            "patient_id": f"patient_{i:03d}",
            "date": "2025-01-01",
        }

        examples.append(Example(text=text, metadata=metadata, id=f"medical_{i:03d}"))

    return examples


def _load_sentiment_data() -> List[Example]:
    """Load synthetic sentiment analysis examples."""
    examples_data = [
        ("This movie is absolutely fantastic!", "positive"),
        ("I hate this boring film.", "negative"),
        ("The movie was okay, nothing special.", "neutral"),
        ("Amazing performance by the lead actor!", "positive"),
        ("Terrible plot and bad acting.", "negative"),
        ("Best movie I've seen this year!", "positive"),
        ("Couldn't even finish watching it.", "negative"),
        ("Pretty good movie overall.", "positive"),
        ("Not worth the time or money.", "negative"),
        ("Decent entertainment for a weekend.", "neutral"),
        ("Absolutely love this restaurant!", "positive"),
        ("Food was cold and tasteless.", "negative"),
        ("Service was friendly and quick.", "positive"),
        ("Worst dining experience ever.", "negative"),
        ("The ambiance is quite nice.", "positive"),
        ("Overpriced for what you get.", "negative"),
        ("Great place for a date night.", "positive"),
        ("Staff seemed uninterested.", "negative"),
        ("Food quality has improved lately.", "positive"),
        ("Average food, average service.", "neutral"),
    ]

    examples = []
    for i, (text, sentiment) in enumerate(examples_data):
        metadata = {"true_label": sentiment, "source": "synthetic", "domain": "review"}

        examples.append(Example(text=text, metadata=metadata, id=f"sentiment_{i:03d}"))

    return examples


def _load_spam_data() -> List[Example]:
    """Load synthetic spam detection examples."""
    examples_data = [
        ("URGENT: You've won $1000! Click here now!", "spam"),
        ("Meeting scheduled for tomorrow at 2 PM.", "ham"),
        ("FREE MONEY! No strings attached!!!", "spam"),
        ("Can you pick up milk on your way home?", "ham"),
        ("Congratulations! You're our lucky winner!", "spam"),
        ("Project deadline moved to next Friday.", "ham"),
        ("LOSE WEIGHT FAST! Buy our pills now!", "spam"),
        ("Thanks for the birthday wishes!", "ham"),
        ("Limited time offer! Act now!", "spam"),
        ("See you at the conference next week.", "ham"),
        ("Your account will be closed! Click here!", "spam"),
        ("Happy anniversary! Hope you have a great day.", "ham"),
        ("Make money from home! No experience needed!", "spam"),
        ("Reminder: dentist appointment tomorrow.", "ham"),
        ("URGENT: Verify your account immediately!", "spam"),
        ("Great job on the presentation today.", "ham"),
        ("Buy now and get 50% off!", "spam"),
        ("Let's grab lunch sometime this week.", "ham"),
        ("Your subscription is about to expire!", "spam"),
        ("Looking forward to our meeting.", "ham"),
    ]

    examples = []
    for i, (text, label) in enumerate(examples_data):
        metadata = {"true_label": label, "source": "synthetic", "domain": "email"}

        examples.append(Example(text=text, metadata=metadata, id=f"spam_{i:03d}"))

    return examples


def load_from_csv(
    filepath: str,
    text_column: str = "text",
    id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
) -> List[Example]:
    """
    Load examples from a CSV file.

    Args:
        filepath: Path to the CSV file
        text_column: Name of the column containing text
        id_column: Name of the column containing IDs (optional)
        metadata_columns: List of columns to include as metadata

    Returns:
        List of Example objects
    """
    df = pd.read_csv(filepath)

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in CSV")

    examples = []
    metadata_cols = metadata_columns or []

    for i, row in df.iterrows():
        # Extract text
        text = str(row[text_column])

        # Extract ID
        example_id = str(row[id_column]) if id_column else f"csv_{i:06d}"

        # Extract metadata
        metadata = {}
        for col in metadata_cols:
            if col in df.columns:
                metadata[col] = row[col]

        examples.append(Example(text=text, metadata=metadata, id=example_id))

    logger.info(f"Loaded {len(examples)} examples from {filepath}")
    return examples


def load_from_jsonl(filepath: str) -> List[Example]:
    """
    Load examples from a JSONL file.

    Expected format:
    {"text": "example text", "id": "ex1", "metadata": {...}}

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of Example objects
    """
    import json

    examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                if "text" not in data:
                    logger.warning(f"Line {line_no}: missing 'text' field")
                    continue

                example = Example(
                    text=data["text"],
                    metadata=data.get("metadata", {}),
                    id=data.get("id", f"jsonl_{line_no:06d}"),
                )

                examples.append(example)

            except json.JSONDecodeError:
                logger.warning(f"Line {line_no}: invalid JSON")
                continue

    logger.info(f"Loaded {len(examples)} examples from {filepath}")
    return examples


def split_examples(
    examples: List[Example],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: Optional[int] = None,
) -> Dict[str, List[Example]]:
    """
    Split examples into train/validation/test sets.

    Args:
        examples: List of examples to split
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test'
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle examples
    shuffled = examples.copy()
    np.random.shuffle(shuffled)

    n_examples = len(shuffled)
    n_train = int(n_examples * train_ratio)
    n_val = int(n_examples * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    logger.info(
        f"Split {n_examples} examples into train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    return splits

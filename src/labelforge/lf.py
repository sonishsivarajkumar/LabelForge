"""
Labeling Function API and registry for LabelForge.
"""

from typing import Any, Optional, Callable
import logging
import numpy as np

from .types import Example, Label, ABSTAIN, LFOutput

logger = logging.getLogger(__name__)


class LabelingFunction:
    """
    Encapsulates a labeling function with metadata and execution logic.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[Example], Label],
        tags: Optional[dict[str, Any]] = None,
        abstain_label: Label = ABSTAIN,
        description: Optional[str] = None,
    ):
        self.name = name
        self.func = func
        self.tags = tags or {}
        self.abstain_label = abstain_label
        self.description = description or f"Labeling function: {name}"

        # Performance tracking
        self.n_calls = 0
        self.n_abstains = 0
        self.n_errors = 0

    def __call__(self, example: Example) -> Label:
        """Apply the LF to a single example."""
        self.n_calls += 1

        try:
            result = self.func(example)
            if result == self.abstain_label:
                self.n_abstains += 1
            return result
        except Exception as e:
            self.n_errors += 1
            logger.warning(f"LF {self.name} failed on example {example.id}: {e}")
            return self.abstain_label

    def apply(self, examples: list[Example]) -> np.ndarray:
        """Apply the LF to a list of examples."""
        return np.array([self(ex) for ex in examples])

    def reset_stats(self):
        """Reset performance tracking statistics."""
        self.n_calls = 0
        self.n_abstains = 0
        self.n_errors = 0

    @property
    def coverage(self) -> float:
        """Fraction of examples that don't abstain."""
        if self.n_calls == 0:
            return 0.0
        return 1.0 - (self.n_abstains / self.n_calls)

    @property
    def error_rate(self) -> float:
        """Fraction of examples that caused errors."""
        if self.n_calls == 0:
            return 0.0
        return self.n_errors / self.n_calls

    def __repr__(self) -> str:
        return f"LabelingFunction(name='{self.name}', coverage={self.coverage:.2f})"


# Global registry for labeling functions
LF_REGISTRY: dict[str, LabelingFunction] = {}


def lf(
    name: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
    abstain_label: Label = ABSTAIN,
    description: Optional[str] = None,
) -> Callable[[Callable], LabelingFunction]:
    """
    Decorator for defining a labeling function.

    Args:
        name: Name for the LF (defaults to function name)
        tags: Metadata tags for organization
        abstain_label: Label to return when abstaining
        description: Human-readable description

    Example:
        @lf(name="has_age", tags={"type": "demographic"}, abstain_label=-1)
        def lf_has_age(ex: Example) -> int:
            return 1 if "age" in ex.text.lower() else -1
    """

    def decorator(func: Callable[[Example], Label]) -> LabelingFunction:
        lf_name = name or func.__name__

        labeling_fn = LabelingFunction(
            name=lf_name,
            func=func,
            tags=tags,
            abstain_label=abstain_label,
            description=description,
        )

        # Register the LF globally
        LF_REGISTRY[lf_name] = labeling_fn

        return labeling_fn

    return decorator


def apply_lfs(
    examples: list[Example], lfs: Optional[list[LabelingFunction]] = None
) -> LFOutput:
    """
    Apply multiple labeling functions to a dataset.

    Args:
        examples: List of examples to label
        lfs: List of LFs to apply (defaults to all registered LFs)

    Returns:
        LFOutput containing the vote matrix and metadata
    """
    if lfs is None:
        lfs = list(LF_REGISTRY.values())

    if not lfs:
        raise ValueError("No labeling functions provided or registered")

    # Reset stats for fresh run
    for lf in lfs:
        lf.reset_stats()

    # Apply each LF to all examples
    votes = np.zeros((len(examples), len(lfs)), dtype=int)

    for j, lf in enumerate(lfs):
        logger.info(f"Applying LF: {lf.name}")
        votes[:, j] = lf.apply(examples)

    return LFOutput(
        votes=votes,
        lf_names=[lf.name for lf in lfs],
        example_ids=[ex.id for ex in examples],
        abstain_value=ABSTAIN,
    )


def get_lf_summary() -> dict[str, Any]:
    """Get summary statistics for all registered LFs."""
    summary = {"total_lfs": len(LF_REGISTRY), "lfs": {}}

    for name, lf in LF_REGISTRY.items():
        summary["lfs"][name] = {
            "name": lf.name,
            "tags": lf.tags,
            "description": lf.description,
            "coverage": lf.coverage,
            "error_rate": lf.error_rate,
            "n_calls": lf.n_calls,
        }

    return summary


def clear_lf_registry():
    """Clear all registered labeling functions."""
    LF_REGISTRY.clear()


# Factory functions for creating LFs (not decorated)
def keyword_contains_factory(
    keywords: list[str], case_sensitive: bool = False
) -> LabelingFunction:
    """Factory for creating keyword matching LFs."""

    def lf_keyword_contains(ex: Example) -> int:
        text = ex.text if case_sensitive else ex.text.lower()
        search_keywords = keywords if case_sensitive else [k.lower() for k in keywords]
        return 1 if any(keyword in text for keyword in search_keywords) else 0

    return LabelingFunction(
        name=f"keyword_contains_{hash(tuple(keywords)) % 10000}",
        func=lf_keyword_contains,
        tags={"type": "keyword", "factory": True},
        description=f"Keyword matching: {keywords}",
    )


def regex_match_factory(pattern: str) -> LabelingFunction:
    """Factory for creating regex matching LFs."""
    import re

    compiled_pattern = re.compile(pattern)

    def lf_regex_match(ex: Example) -> int:
        match = compiled_pattern.search(ex.text)
        return 1 if match else 0

    return LabelingFunction(
        name=f"regex_match_{hash(pattern) % 10000}",
        func=lf_regex_match,
        tags={"type": "regex", "factory": True},
        description=f"Regex pattern: {pattern}",
    )

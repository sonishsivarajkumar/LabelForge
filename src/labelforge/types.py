"""
Core types and data structures for LabelForge.
"""

from typing import Any, Union, Optional
from dataclasses import dataclass
import numpy as np

# Type aliases
Label = Union[int, str]
ABSTAIN = -1


@dataclass
class Example:
    """
    A single data example that can be labeled by labeling functions.

    Attributes:
        text: The primary text content (for NLP tasks)
        metadata: Additional structured data
        features: Numerical features for the example
        id: Unique identifier for the example
    """

    text: str
    metadata: Optional[dict[str, Any]] = None
    features: Optional[np.ndarray] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            self.id = str(hash(self.text))


@dataclass
class LFOutput:
    """
    Output from applying labeling functions to a dataset.

    Attributes:
        votes: Matrix of LF votes (n_examples x n_lfs)
        lf_names: Names of the labeling functions
        example_ids: IDs of the examples
        abstain_value: Value used for abstentions
    """

    votes: np.ndarray
    lf_names: list[str]
    example_ids: list[str]
    abstain_value: Label = ABSTAIN

    @property
    def n_examples(self) -> int:
        return self.votes.shape[0]

    @property
    def n_lfs(self) -> int:
        return self.votes.shape[1]

    def coverage(self) -> np.ndarray:
        """Coverage (% non-abstain) for each LF."""
        return np.mean(self.votes != self.abstain_value, axis=0)

    def overlap(self) -> np.ndarray:
        """Pairwise overlap matrix between LFs."""
        non_abstain = self.votes != self.abstain_value
        overlap_matrix = np.zeros((self.n_lfs, self.n_lfs))

        for i in range(self.n_lfs):
            for j in range(self.n_lfs):
                both_vote = non_abstain[:, i] & non_abstain[:, j]
                if np.sum(both_vote) > 0:
                    overlap_matrix[i, j] = np.sum(both_vote) / self.n_examples

        return overlap_matrix

    def conflict(self) -> np.ndarray:
        """Pairwise conflict matrix between LFs."""
        conflict_matrix = np.zeros((self.n_lfs, self.n_lfs))

        for i in range(self.n_lfs):
            for j in range(self.n_lfs):
                both_vote = (self.votes[:, i] != self.abstain_value) & (
                    self.votes[:, j] != self.abstain_value
                )
                if np.sum(both_vote) > 0:
                    disagree = self.votes[both_vote, i] != self.votes[both_vote, j]
                    conflict_matrix[i, j] = np.mean(disagree)

        return conflict_matrix

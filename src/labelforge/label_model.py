"""
Probabilistic Label Model for combining weak supervision signals.

Implements an EM-based generative model to learn labeling function accuracies
and correlations, then outputs probabilistic labels for training.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from scipy.special import logsumexp
from sklearn.utils import check_random_state

from .types import LFOutput, ABSTAIN

logger = logging.getLogger(__name__)


class LabelModel:
    """
    Probabilistic model for combining labeling function outputs.

    Uses an EM algorithm to learn:
    - Class priors π_y
    - LF accuracy matrices α_{j,y,ℓ}

    Then predicts soft labels P(Y|L) for downstream training.
    """

    def __init__(
        self,
        cardinality: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Label Model.

        Args:
            cardinality: Number of classes (default: 2 for binary)
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.cardinality = cardinality
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters (learned during fit)
        self.class_priors_ = None  # π_y
        self.lf_accuracies_ = None  # α_{j,y,ℓ}
        self.n_lfs_ = None
        self.lf_names_ = None

        # Training history
        self.history_ = {"log_likelihood": [], "converged": False, "n_iter": 0}

    def fit(self, lf_output: LFOutput) -> "LabelModel":
        """
        Fit the label model using EM algorithm.

        Args:
            lf_output: Output from applying labeling functions

        Returns:
            self (fitted model)
        """
        # Extract data
        L = lf_output.votes  # (n_examples, n_lfs)
        self.n_lfs_ = lf_output.n_lfs
        self.lf_names_ = lf_output.lf_names
        n_examples = lf_output.n_examples

        if self.verbose:
            logger.info(
                f"Fitting label model on {n_examples} examples, {self.n_lfs_} LFs"
            )

        # Initialize parameters
        rng = check_random_state(self.random_state)
        self._initialize_parameters(L, rng)

        # EM Algorithm
        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step: compute responsibilities γ_{i,y} = P(Y_i = y | L_i)
            log_probs, log_likelihood = self._e_step(L)

            # M-step: update parameters
            self._m_step(L, log_probs)

            # Check convergence
            improvement = log_likelihood - prev_log_likelihood
            self.history_["log_likelihood"].append(log_likelihood)

            if self.verbose:
                logger.info(
                    f"Iteration {iteration + 1}: "
                    f"log_likelihood = {log_likelihood:.6f}, "
                    f"improvement = {improvement:.8f}"
                )

            if improvement < self.tol:
                self.history_["converged"] = True
                break

            prev_log_likelihood = log_likelihood

        self.history_["n_iter"] = iteration + 1

        if not self.history_["converged"] and self.verbose:
            logger.warning(f"EM did not converge after {self.max_iter} iterations")

        return self

    def predict_proba(self, lf_output: LFOutput) -> np.ndarray:
        """
        Predict class probabilities for examples.

        Args:
            lf_output: LF outputs for examples to predict

        Returns:
            Array of shape (n_examples, n_classes) with class probabilities
        """
        if self.class_priors_ is None:
            raise ValueError("Model must be fitted before predicting")

        L = lf_output.votes
        log_probs, _ = self._e_step(L)

        return np.exp(log_probs)

    def predict(self, lf_output: LFOutput) -> np.ndarray:
        """
        Predict hard class labels.

        Args:
            lf_output: LF outputs for examples to predict

        Returns:
            Array of predicted class labels
        """
        probs = self.predict_proba(lf_output)
        return np.argmax(probs, axis=1)

    def _initialize_parameters(self, L: np.ndarray, rng: np.random.RandomState):
        """Initialize model parameters randomly."""
        n_examples, n_lfs = L.shape

        # Initialize class priors uniformly
        self.class_priors_ = np.ones(self.cardinality) / self.cardinality

        # Initialize LF accuracy matrices
        # α_{j,y,ℓ} = P(L_j = ℓ | Y = y)
        unique_votes = np.unique(L)
        n_vote_values = len(unique_votes)

        self.lf_accuracies_ = np.zeros((n_lfs, self.cardinality, n_vote_values))
        self.vote_to_idx_ = {vote: idx for idx, vote in enumerate(unique_votes)}
        self.idx_to_vote_ = {idx: vote for idx, vote in enumerate(unique_votes)}

        # Random initialization with some structure
        for j in range(n_lfs):
            for y in range(self.cardinality):
                # Start with uniform + noise
                self.lf_accuracies_[j, y, :] = rng.dirichlet(np.ones(n_vote_values))

                # Bias towards correct class (simple heuristic)
                if len(unique_votes) > 1:
                    non_abstain_votes = [v for v in unique_votes if v != ABSTAIN]
                    if non_abstain_votes and y < len(non_abstain_votes):
                        vote_idx = self.vote_to_idx_[non_abstain_votes[y]]
                        self.lf_accuracies_[j, y, vote_idx] += 0.3

                # Re-normalize
                self.lf_accuracies_[j, y, :] /= np.sum(self.lf_accuracies_[j, y, :])

    def _e_step(self, L: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        E-step: compute posterior probabilities P(Y | L).

        Returns:
            log_probs: Log probabilities of shape (n_examples, n_classes)
            log_likelihood: Total log likelihood
        """
        n_examples, n_lfs = L.shape
        log_probs = np.zeros((n_examples, self.cardinality))

        for i in range(n_examples):
            for y in range(self.cardinality):
                # Log P(Y_i = y)
                log_prob = np.log(self.class_priors_[y])

                # Log ∏_j P(L_{i,j} | Y_i = y)
                for j in range(n_lfs):
                    vote = L[i, j]
                    if vote in self.vote_to_idx_:
                        vote_idx = self.vote_to_idx_[vote]
                        log_prob += np.log(self.lf_accuracies_[j, y, vote_idx] + 1e-15)

                log_probs[i, y] = log_prob

        # Normalize using logsumexp for numerical stability
        log_probs = log_probs - logsumexp(log_probs, axis=1, keepdims=True)

        # Compute log likelihood
        log_likelihood = np.sum(logsumexp(log_probs, axis=1))

        return log_probs, log_likelihood

    def _m_step(self, L: np.ndarray, log_probs: np.ndarray):
        """
        M-step: update parameters given posterior probabilities.

        Args:
            L: Vote matrix
            log_probs: Log posterior probabilities from E-step
        """
        n_examples, n_lfs = L.shape
        probs = np.exp(log_probs)  # Convert to regular probabilities

        # Update class priors: π_y = (1/n) Σ_i γ_{i,y}
        self.class_priors_ = np.mean(probs, axis=0)

        # Update LF accuracies: α_{j,y,ℓ} = Σ_i γ_{i,y} * 1{L_{i,j} = ℓ} / Σ_i γ_{i,y}
        for j in range(n_lfs):
            for y in range(self.cardinality):
                class_weight = np.sum(probs[:, y])

                if class_weight > 1e-15:  # Avoid division by zero
                    for vote_val, vote_idx in self.vote_to_idx_.items():
                        vote_mask = L[:, j] == vote_val
                        numerator = np.sum(probs[vote_mask, y])
                        self.lf_accuracies_[j, y, vote_idx] = numerator / class_weight
                else:
                    # Uniform if no weight
                    self.lf_accuracies_[j, y, :] = 1.0 / len(self.vote_to_idx_)

    def get_lf_stats(self) -> Dict[str, Any]:
        """Get statistics about learned LF parameters."""
        if self.lf_accuracies_ is None:
            raise ValueError("Model must be fitted first")

        stats = {
            "lf_names": self.lf_names_,
            "class_priors": self.class_priors_.tolist(),
            "lf_accuracies": {},
            "training_history": self.history_,
        }

        for j, lf_name in enumerate(self.lf_names_):
            stats["lf_accuracies"][lf_name] = {
                "accuracy_matrix": self.lf_accuracies_[j].tolist(),
                "vote_mapping": self.vote_to_idx_,
            }

        return stats

    def score(self, lf_output: LFOutput, y_true: np.ndarray) -> Dict[str, float]:
        """
        Score the model against true labels.

        Args:
            lf_output: LF outputs
            y_true: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            log_loss,
        )

        y_pred = self.predict(lf_output)
        y_proba = self.predict_proba(lf_output)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Log loss (if probabilities are well-calibrated)
        try:
            logloss = log_loss(y_true, y_proba)
        except ValueError:
            logloss = np.nan

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": logloss,
        }

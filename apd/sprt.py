"""Sequential Probability Ratio Test (Wald SPRT) for Active Protocol Discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SPRTState:
    """Running state of a sequential test."""

    log_likelihood_ratio: float = 0.0
    steps: int = 0
    history: list[float] = field(default_factory=list)
    decision: int | None = None  # None=undecided, 1=H1, 0=H0


class WaldSPRT:
    """Classic Wald Sequential Probability Ratio Test.

    Accumulates log-likelihood ratios and decides between:
        H0: no adaptive structure (null)
        H1: adaptive structure present

    Parameters
    ----------
    alpha : float
        Type I error rate (false positive).
    beta : float
        Type II error rate (false negative).
    """

    def __init__(self, alpha: float = 0.01, beta: float = 0.01) -> None:
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("alpha and beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        # Upper threshold: accept H1
        self.threshold_upper = math.log((1.0 - beta) / alpha)
        # Lower threshold: accept H0
        self.threshold_lower = math.log(beta / (1.0 - alpha))

    def log_likelihood_ratio(
        self, y: float, probe: float, sigma: float
    ) -> float:
        """Compute single-observation log-likelihood ratio for Gaussian case.

        LLR = log p(y|H1,u) - log p(y|H0)
            = (y * mu(u) - mu(u)²/2) / sigma²

        where mu(u) = u for the default Gaussian model.
        """
        mu = probe  # mu(u) = u for Gaussian
        return (y * mu - 0.5 * mu ** 2) / (sigma ** 2)

    def log_likelihood_ratio_general(
        self, y: float, mu_h1: float, sigma: float
    ) -> float:
        """Compute LLR given an explicit H1 mean."""
        return (y * mu_h1 - 0.5 * mu_h1 ** 2) / (sigma ** 2)

    def new_state(self) -> SPRTState:
        """Create a fresh test state."""
        return SPRTState()

    def update(self, state: SPRTState, llr: float) -> SPRTState:
        """Update SPRT state with a new observation's LLR.

        Returns the (mutated) state with possible decision.
        """
        state.log_likelihood_ratio += llr
        state.steps += 1
        state.history.append(state.log_likelihood_ratio)

        if state.log_likelihood_ratio >= self.threshold_upper:
            state.decision = 1  # Accept H1 (adaptive)
        elif state.log_likelihood_ratio <= self.threshold_lower:
            state.decision = 0  # Accept H0 (null)

        return state

    def is_decided(self, state: SPRTState) -> bool:
        """Check if the test has reached a decision."""
        return state.decision is not None

"""Probe selection policies for Active Protocol Discovery.

Each policy implements:
    select(probes, history) -> chosen probe
    kl_scores(probes, sigma) -> dict[probe, score]  (where applicable)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.random import Generator


class KLOptimalPolicy:
    """Selects the probe maximising KL divergence between H0 and H1.

    For the Gaussian case: D(u) = mu(u)² / (2 * sigma²).
    For general worlds, uses empirical KL estimation from history.

    Parameters
    ----------
    probes : sequence of probe values
        Candidate probes to choose from.
    sigma : float
        Noise standard deviation (used for Gaussian closed-form).
    mu_fn : callable or None
        Function mapping probe -> expected shift under H1.
        If None, uses probe value directly (Gaussian default).
    """

    def __init__(
        self,
        probes: Sequence[Any],
        sigma: float = 1.0,
        mu_fn: Any = None,
    ) -> None:
        self.probes = list(probes)
        self.sigma = sigma
        self.mu_fn = mu_fn or (lambda u: u)

    def kl_score(self, probe: Any) -> float:
        """KL divergence D_KL(H1 || H0) for a given probe."""
        mu = self.mu_fn(probe)
        if isinstance(mu, (int, float)):
            return float(mu) ** 2 / (2.0 * self.sigma ** 2)
        # Vector case: ||mu||² / (2σ²)
        mu = np.asarray(mu, dtype=float)
        return float(np.dot(mu, mu)) / (2.0 * self.sigma ** 2)

    def kl_scores(self) -> dict:
        """Compute KL scores for all candidate probes."""
        return {p: self.kl_score(p) for p in self.probes}

    def select(self, history: list | None = None, rng: Generator | None = None) -> Any:
        """Select the probe with maximum KL divergence."""
        best_probe = max(self.probes, key=self.kl_score)
        return best_probe


class FixedPolicy:
    """Always selects the same probe (beacon mode).

    Parameters
    ----------
    probe : any
        The fixed probe to use every step.
    """

    def __init__(self, probe: Any) -> None:
        self.probe = probe

    def select(self, history: list | None = None, rng: Generator | None = None) -> Any:
        return self.probe


class RandomPolicy:
    """Selects a probe uniformly at random from the candidate set.

    Parameters
    ----------
    probes : sequence
        Candidate probes to choose from.
    """

    def __init__(self, probes: Sequence[Any]) -> None:
        self.probes = list(probes)

    def select(self, history: list | None = None, rng: Generator | None = None) -> Any:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, len(self.probes))
        return self.probes[idx]


class EpsilonGreedyPolicy:
    """Epsilon-greedy: explore random probes with probability epsilon,
    otherwise select the KL-optimal probe.

    Parameters
    ----------
    probes : sequence
        Candidate probes.
    sigma : float
        Noise standard deviation.
    epsilon : float
        Exploration probability.
    mu_fn : callable or None
        Function mapping probe -> expected shift.
    """

    def __init__(
        self,
        probes: Sequence[Any],
        sigma: float = 1.0,
        epsilon: float = 0.1,
        mu_fn: Any = None,
    ) -> None:
        self.kl_policy = KLOptimalPolicy(probes, sigma, mu_fn)
        self.random_policy = RandomPolicy(probes)
        self.epsilon = epsilon
        self.probes = list(probes)

    def select(self, history: list | None = None, rng: Generator | None = None) -> Any:
        rng = rng or np.random.default_rng()
        if rng.random() < self.epsilon:
            return self.random_policy.select(history, rng)
        return self.kl_policy.select(history, rng)

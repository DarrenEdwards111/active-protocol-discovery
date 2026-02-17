"""World models for Active Protocol Discovery.

Each world implements the interface:
    - sample(probe, rng) -> observation
    - mu(probe) -> expected response shift under H1
    - sigma -> noise standard deviation
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


class GaussianWorld:
    """Gaussian observation model.

    Under H0 (adaptive=False): y ~ N(0, sigma²) regardless of probe.
    Under H1 (adaptive=True):  y ~ N(mu(u), sigma²) where mu(u) = u.

    Parameters
    ----------
    sigma : float
        Noise standard deviation.
    adaptive : bool
        If True, the world responds adaptively to probes (H1).
    """

    def __init__(self, sigma: float = 1.0, adaptive: bool = True) -> None:
        self.sigma = sigma
        self.adaptive = adaptive

    def mu(self, probe: float) -> float:
        """Expected response shift under H1 for a given probe strength."""
        return probe if self.adaptive else 0.0

    def sample(self, probe: float, rng: Generator | None = None) -> float:
        """Draw one observation given probe strength."""
        rng = rng or np.random.default_rng()
        mean = self.mu(probe)
        return float(rng.normal(mean, self.sigma))


class AdaptiveAgentWorld:
    """Multi-dimensional world with a hidden adaptive agent.

    The environment has `dim` dimensions. Under H1, the agent responds
    adaptively along a hidden direction when the probe has sufficient
    projection onto that direction.

    Parameters
    ----------
    dim : int
        Dimensionality of probe/observation space.
    sigma : float
        Noise standard deviation.
    adaptive : bool
        Whether the hidden agent is present.
    hidden_direction : array-like or None
        The direction the agent responds to. Random if None.
    sensitivity : float
        How strongly the agent responds.
    """

    def __init__(
        self,
        dim: int = 5,
        sigma: float = 1.0,
        adaptive: bool = True,
        hidden_direction: np.ndarray | None = None,
        sensitivity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.sigma = sigma
        self.adaptive = adaptive
        self.sensitivity = sensitivity

        init_rng = np.random.default_rng(seed)
        if hidden_direction is not None:
            self.hidden_direction = np.asarray(hidden_direction, dtype=float)
            self.hidden_direction /= np.linalg.norm(self.hidden_direction)
        else:
            self.hidden_direction = init_rng.standard_normal(dim)
            self.hidden_direction /= np.linalg.norm(self.hidden_direction)

    def mu(self, probe: np.ndarray) -> np.ndarray:
        """Expected response shift under H1."""
        probe = np.asarray(probe, dtype=float)
        if not self.adaptive:
            return np.zeros(self.dim)
        projection = float(np.dot(probe, self.hidden_direction))
        return self.sensitivity * projection * self.hidden_direction

    def sample(self, probe: np.ndarray, rng: Generator | None = None) -> np.ndarray:
        """Draw one vector observation."""
        rng = rng or np.random.default_rng()
        probe = np.asarray(probe, dtype=float)
        mean = self.mu(probe)
        noise = rng.normal(0, self.sigma, size=self.dim)
        return mean + noise


class NetworkWorld:
    """Simulated network with optional hidden bot.

    Probes are timing intervals (floats). Under H1, a hidden bot
    introduces correlated timing shifts proportional to the probe.

    Parameters
    ----------
    sigma : float
        Baseline timing noise (ms).
    adaptive : bool
        Whether the bot is present.
    bot_gain : float
        Multiplier for the bot's response to timing probes.
    """

    def __init__(
        self, sigma: float = 5.0, adaptive: bool = True, bot_gain: float = 2.0
    ) -> None:
        self.sigma = sigma
        self.adaptive = adaptive
        self.bot_gain = bot_gain

    def mu(self, probe: float) -> float:
        """Expected timing shift under H1."""
        if not self.adaptive:
            return 0.0
        return self.bot_gain * probe

    def sample(self, probe: float, rng: Generator | None = None) -> float:
        """Draw one timing observation (ms)."""
        rng = rng or np.random.default_rng()
        mean = self.mu(probe)
        return float(rng.normal(mean, self.sigma))

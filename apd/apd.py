"""Main APD engine combining world model, probe policy, and sequential test."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.random import Generator

from apd.sprt import WaldSPRT, SPRTState


@dataclass
class APDResult:
    """Result of an APD run.

    Attributes
    ----------
    decision : int or None
        1 = adaptive (H1), 0 = null (H0), None = undecided (max steps hit).
    steps : int
        Number of observations taken.
    log_odds_history : list[float]
        Cumulative log-likelihood ratio at each step.
    probes_used : list
        Probe selected at each step.
    """

    decision: int | None
    steps: int
    log_odds_history: list[float] = field(default_factory=list)
    probes_used: list = field(default_factory=list)


class APDEngine:
    """Active Protocol Discovery engine.

    Combines a world model, probe selection policy, and sequential test
    to actively detect hidden adaptive structure.

    Parameters
    ----------
    world : object
        World model with .sample(probe, rng) and .sigma attributes.
    policy : object
        Probe selection policy with .select(history, rng) method.
    sprt : WaldSPRT
        Sequential test instance.
    callback : callable or None
        Called after each step with (step, probe, observation, state).
    """

    def __init__(
        self,
        world: Any,
        policy: Any,
        sprt: WaldSPRT,
        callback: Callable | None = None,
        mu_h1_fn: Callable | None = None,
    ) -> None:
        self.world = world
        self.policy = policy
        self.sprt = sprt
        self.callback = callback
        # mu_h1_fn: maps probe -> expected response under H1.
        # This is the *hypothesised* H1 model, NOT the true world.
        # Default: mu(u) = u (Gaussian identity).
        self.mu_h1_fn = mu_h1_fn

    def run(
        self,
        max_steps: int = 1000,
        seed: int | None = None,
    ) -> APDResult:
        """Run the APD procedure.

        Parameters
        ----------
        max_steps : int
            Maximum number of observations before stopping.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        APDResult
        """
        rng = np.random.default_rng(seed)
        state = self.sprt.new_state()
        probes_used: list = []
        history: list[tuple] = []

        for step in range(max_steps):
            # Select probe
            probe = self.policy.select(history=history, rng=rng)

            # Observe
            y = self.world.sample(probe, rng=rng)

            # Compute LLR using the H1 hypothesis model
            if self.mu_h1_fn is not None:
                mu_h1 = self.mu_h1_fn(probe)
            else:
                # Default: mu(u) = u (scalar Gaussian)
                mu_h1 = probe

            if isinstance(mu_h1, (int, float)):
                llr = self.sprt.log_likelihood_ratio_general(
                    float(y), float(mu_h1), self.world.sigma
                )
            else:
                # Vector case: sum of component LLRs
                mu_h1 = np.asarray(mu_h1)
                y_arr = np.asarray(y)
                llr = float(
                    np.sum(y_arr * mu_h1 - 0.5 * mu_h1 ** 2)
                ) / (self.world.sigma ** 2)

            # Update SPRT
            self.sprt.update(state, llr)
            probes_used.append(probe)
            history.append((probe, y))

            if self.callback:
                self.callback(step, probe, y, state)

            if self.sprt.is_decided(state):
                break

        return APDResult(
            decision=state.decision,
            steps=state.steps,
            log_odds_history=list(state.history),
            probes_used=probes_used,
        )

"""Integration tests for the APD engine."""

import numpy as np
import pytest

from apd.models import GaussianWorld
from apd.policy import KLOptimalPolicy, FixedPolicy
from apd.sprt import WaldSPRT
from apd.apd import APDEngine


class TestAPDEngine:
    def test_detects_h1(self):
        """APD should detect adaptive structure under H1."""
        world = GaussianWorld(sigma=1.0, adaptive=True)
        policy = KLOptimalPolicy(probes=[0.5, 1.0], sigma=1.0)
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        engine = APDEngine(world, policy, sprt)
        result = engine.run(max_steps=10000, seed=42)
        assert result.decision == 1
        assert result.steps < 100  # Should be fast with u=1.0

    def test_rejects_h0(self):
        """APD should accept null under H0."""
        world = GaussianWorld(sigma=1.0, adaptive=False)
        policy = KLOptimalPolicy(probes=[0.5, 1.0], sigma=1.0)
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        engine = APDEngine(world, policy, sprt)
        result = engine.run(max_steps=10000, seed=42)
        assert result.decision == 0

    def test_result_fields(self):
        world = GaussianWorld(sigma=1.0, adaptive=True)
        policy = FixedPolicy(1.0)
        sprt = WaldSPRT(alpha=0.05, beta=0.05)
        engine = APDEngine(world, policy, sprt)
        result = engine.run(max_steps=500, seed=0)
        assert result.decision in (0, 1)
        assert result.steps > 0
        assert len(result.log_odds_history) == result.steps
        assert len(result.probes_used) == result.steps

    def test_callback_called(self):
        calls = []
        def cb(step, probe, y, state):
            calls.append(step)

        world = GaussianWorld(sigma=1.0, adaptive=True)
        policy = FixedPolicy(1.0)
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        engine = APDEngine(world, policy, sprt, callback=cb)
        result = engine.run(max_steps=500, seed=0)
        assert len(calls) == result.steps

    def test_statistical_error_rates(self):
        """Over many trials, error rates should approximately match alpha/beta."""
        alpha, beta = 0.05, 0.05
        n_trials = 500

        # H1 trials
        miss = 0
        for i in range(n_trials):
            world = GaussianWorld(sigma=1.0, adaptive=True)
            policy = FixedPolicy(1.0)
            sprt = WaldSPRT(alpha=alpha, beta=beta)
            engine = APDEngine(world, policy, sprt)
            result = engine.run(max_steps=10000, seed=i)
            if result.decision != 1:
                miss += 1
        miss_rate = miss / n_trials
        assert miss_rate < beta * 3  # Allow some slack

        # H0 trials
        false_alarm = 0
        for i in range(n_trials):
            world = GaussianWorld(sigma=1.0, adaptive=False)
            policy = FixedPolicy(1.0)
            sprt = WaldSPRT(alpha=alpha, beta=beta)
            engine = APDEngine(world, policy, sprt)
            result = engine.run(max_steps=10000, seed=i + 10000)
            if result.decision == 1:
                false_alarm += 1
        fp_rate = false_alarm / n_trials
        assert fp_rate < alpha * 3

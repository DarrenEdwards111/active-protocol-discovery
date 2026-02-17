"""Tests for probe selection policies."""

import numpy as np
import pytest

from apd.policy import KLOptimalPolicy, FixedPolicy, RandomPolicy, EpsilonGreedyPolicy


class TestKLOptimalPolicy:
    def test_selects_strongest_probe(self):
        policy = KLOptimalPolicy(probes=[0.2, 0.5, 1.0], sigma=1.0)
        assert policy.select() == 1.0

    def test_kl_scores(self):
        policy = KLOptimalPolicy(probes=[0.2, 1.0], sigma=1.0)
        scores = policy.kl_scores()
        assert abs(scores[0.2] - 0.02) < 1e-10
        assert abs(scores[1.0] - 0.5) < 1e-10

    def test_kl_score_formula(self):
        # D(u) = u²/(2σ²)
        policy = KLOptimalPolicy(probes=[2.0], sigma=3.0)
        expected = 4.0 / 18.0
        assert abs(policy.kl_score(2.0) - expected) < 1e-10

    def test_custom_mu_fn(self):
        policy = KLOptimalPolicy(
            probes=[1.0, 2.0], sigma=1.0, mu_fn=lambda u: 2 * u
        )
        # mu(2) = 4, D = 16/2 = 8; mu(1) = 2, D = 4/2 = 2
        assert policy.select() == 2.0


class TestFixedPolicy:
    def test_always_same(self):
        policy = FixedPolicy(0.5)
        for _ in range(10):
            assert policy.select() == 0.5


class TestRandomPolicy:
    def test_covers_all_probes(self):
        probes = [0.1, 0.5, 1.0]
        policy = RandomPolicy(probes)
        rng = np.random.default_rng(0)
        selected = {policy.select(rng=rng) for _ in range(100)}
        assert selected == set(probes)


class TestEpsilonGreedyPolicy:
    def test_mostly_optimal(self):
        policy = EpsilonGreedyPolicy(
            probes=[0.2, 1.0], sigma=1.0, epsilon=0.1
        )
        rng = np.random.default_rng(0)
        selections = [policy.select(rng=rng) for _ in range(1000)]
        optimal_frac = sum(1 for s in selections if s == 1.0) / 1000
        assert optimal_frac > 0.85

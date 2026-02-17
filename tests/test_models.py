"""Tests for world models."""

import numpy as np
import pytest

from apd.models import GaussianWorld, AdaptiveAgentWorld, NetworkWorld


class TestGaussianWorld:
    def test_h0_mean_zero(self):
        world = GaussianWorld(sigma=1.0, adaptive=False)
        rng = np.random.default_rng(0)
        samples = [world.sample(1.0, rng) for _ in range(10000)]
        assert abs(np.mean(samples)) < 0.1

    def test_h1_mean_shifted(self):
        world = GaussianWorld(sigma=1.0, adaptive=True)
        rng = np.random.default_rng(0)
        probe = 0.5
        samples = [world.sample(probe, rng) for _ in range(10000)]
        assert abs(np.mean(samples) - probe) < 0.1

    def test_h0_mu_zero(self):
        world = GaussianWorld(sigma=1.0, adaptive=False)
        assert world.mu(1.0) == 0.0

    def test_h1_mu_equals_probe(self):
        world = GaussianWorld(sigma=1.0, adaptive=True)
        assert world.mu(0.7) == 0.7

    def test_sigma_affects_variance(self):
        for sigma in [0.5, 1.0, 2.0]:
            world = GaussianWorld(sigma=sigma, adaptive=False)
            rng = np.random.default_rng(0)
            samples = [world.sample(0.0, rng) for _ in range(10000)]
            assert abs(np.std(samples) - sigma) < 0.1


class TestAdaptiveAgentWorld:
    def test_h0_zero_mean(self):
        world = AdaptiveAgentWorld(dim=3, sigma=1.0, adaptive=False, seed=0)
        rng = np.random.default_rng(0)
        samples = [world.sample(np.array([1.0, 0.0, 0.0]), rng) for _ in range(5000)]
        mean = np.mean(samples, axis=0)
        assert np.all(np.abs(mean) < 0.1)

    def test_h1_shift_along_hidden(self):
        direction = np.array([1.0, 0.0, 0.0])
        world = AdaptiveAgentWorld(
            dim=3, sigma=1.0, adaptive=True,
            hidden_direction=direction, sensitivity=1.0,
        )
        probe = np.array([1.0, 0.0, 0.0])
        mu = world.mu(probe)
        assert abs(mu[0] - 1.0) < 1e-10
        assert abs(mu[1]) < 1e-10

    def test_orthogonal_probe_no_shift(self):
        direction = np.array([1.0, 0.0, 0.0])
        world = AdaptiveAgentWorld(
            dim=3, sigma=1.0, adaptive=True,
            hidden_direction=direction, sensitivity=1.0,
        )
        probe = np.array([0.0, 1.0, 0.0])
        mu = world.mu(probe)
        assert np.allclose(mu, 0.0)


class TestNetworkWorld:
    def test_h0_mean_zero(self):
        world = NetworkWorld(sigma=5.0, adaptive=False)
        rng = np.random.default_rng(0)
        samples = [world.sample(1.0, rng) for _ in range(10000)]
        assert abs(np.mean(samples)) < 0.5

    def test_h1_mean_shifted(self):
        world = NetworkWorld(sigma=5.0, adaptive=True, bot_gain=2.0)
        rng = np.random.default_rng(0)
        probe = 3.0
        samples = [world.sample(probe, rng) for _ in range(10000)]
        assert abs(np.mean(samples) - 6.0) < 0.5

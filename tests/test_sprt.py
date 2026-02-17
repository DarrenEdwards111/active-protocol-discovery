"""Tests for SPRT module."""

import math

import pytest

from apd.sprt import WaldSPRT, SPRTState


class TestWaldSPRT:
    def test_thresholds(self):
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        expected_upper = math.log(99)
        expected_lower = math.log(1 / 99)
        assert abs(sprt.threshold_upper - expected_upper) < 1e-10
        assert abs(sprt.threshold_lower - expected_lower) < 1e-10

    def test_thresholds_asymmetric(self):
        sprt = WaldSPRT(alpha=0.05, beta=0.1)
        assert abs(sprt.threshold_upper - math.log(0.9 / 0.05)) < 1e-10
        assert abs(sprt.threshold_lower - math.log(0.1 / 0.95)) < 1e-10

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            WaldSPRT(alpha=0.0, beta=0.5)
        with pytest.raises(ValueError):
            WaldSPRT(alpha=0.5, beta=1.0)

    def test_llr_gaussian(self):
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        # y=1, u=1, sigma=1: LLR = (1*1 - 0.5*1)/1 = 0.5
        assert abs(sprt.log_likelihood_ratio(1.0, 1.0, 1.0) - 0.5) < 1e-10

    def test_accumulation_h1(self):
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        state = sprt.new_state()
        # Feed strong positive evidence
        for _ in range(100):
            llr = sprt.log_likelihood_ratio(1.5, 1.0, 1.0)
            sprt.update(state, llr)
            if sprt.is_decided(state):
                break
        assert state.decision == 1

    def test_accumulation_h0(self):
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        state = sprt.new_state()
        # Feed evidence against H1 (y centered at 0, probe=1)
        for _ in range(100):
            llr = sprt.log_likelihood_ratio(-0.5, 1.0, 1.0)
            sprt.update(state, llr)
            if sprt.is_decided(state):
                break
        assert state.decision == 0

    def test_history_tracked(self):
        sprt = WaldSPRT(alpha=0.01, beta=0.01)
        state = sprt.new_state()
        sprt.update(state, 0.5)
        sprt.update(state, 0.3)
        assert len(state.history) == 2
        assert state.steps == 2
        assert abs(state.history[-1] - 0.8) < 1e-10

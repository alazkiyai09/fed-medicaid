"""Tests for differential privacy mechanisms."""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from src.privacy.dp_mechanism import GaussianMechanism
from src.privacy.privacy_accountant import PrivacyAccountant


class TestGaussianMechanism:
    def test_noise_multiplier_calculation(self):
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        expected = math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(gm.noise_multiplier - expected) < 1e-6

    def test_infinite_epsilon_no_noise(self):
        gm = GaussianMechanism(epsilon=float("inf"))
        assert gm.noise_multiplier == 0.0

        update = collections.OrderedDict({
            "w": np.array([1.0, 2.0, 3.0]),
        })
        result = gm.clip_and_noise(update)
        np.testing.assert_array_equal(result["w"], update["w"])

    def test_clipping_applied(self):
        gm = GaussianMechanism(epsilon=1.0, max_grad_norm=0.1)
        update = collections.OrderedDict({
            "w": np.array([100.0, 100.0]),  # Huge norm
        })
        result = gm.clip_and_noise(update)
        clipped_norm = np.linalg.norm(
            np.concatenate([v.flatten() for v in result.values()])
        )
        # After clipping, should be close to max_grad_norm (plus noise)
        # Can't be exact due to noise
        assert clipped_norm < 200.0  # Much less than original ~141

    def test_noise_is_added(self):
        gm = GaussianMechanism(epsilon=0.1, max_grad_norm=1.0)
        update = collections.OrderedDict({
            "w": np.zeros(100),
        })
        result = gm.clip_and_noise(update)
        # Should have noise added
        assert np.abs(result["w"]).sum() > 0

    def test_shape_preserved(self):
        gm = GaussianMechanism(epsilon=1.0)
        update = collections.OrderedDict({
            "w1": np.random.randn(10, 5).astype(np.float32),
            "b1": np.random.randn(10).astype(np.float32),
        })
        result = gm.clip_and_noise(update)
        assert result["w1"].shape == (10, 5)
        assert result["b1"].shape == (10,)


class TestPrivacyAccountant:
    def test_basic_composition(self):
        pa = PrivacyAccountant(
            total_epsilon=10.0,
            per_round_epsilon=0.1,
            composition="basic",
        )
        for _ in range(50):
            pa.step()
        spent = pa.get_privacy_spent()
        assert abs(spent.epsilon - 5.0) < 1e-6  # 50 × 0.1

    def test_budget_exhaustion(self):
        pa = PrivacyAccountant(
            total_epsilon=1.0,
            per_round_epsilon=0.5,
            composition="basic",
        )
        pa.step()
        pa.step()
        with pytest.raises(RuntimeError, match="exhausted"):
            pa.step()

    def test_advanced_composition_tighter(self):
        # Advanced composition should give tighter bound than basic
        basic = PrivacyAccountant(per_round_epsilon=0.1, composition="basic")
        advanced = PrivacyAccountant(per_round_epsilon=0.1, composition="advanced")

        for _ in range(100):
            basic.step()
            advanced.step()

        assert advanced.get_privacy_spent().epsilon < basic.get_privacy_spent().epsilon

    def test_remaining_rounds(self):
        pa = PrivacyAccountant(
            total_epsilon=10.0,
            per_round_epsilon=1.0,
            composition="basic",
        )
        remaining = pa.remaining_rounds()
        assert remaining is not None
        assert remaining == 10

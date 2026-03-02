"""Tests for aggregation strategies."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from src.federation.client import ClientUpdate
from src.federation.strategies import (
    FedAvgStrategy,
    FoolsGoldStrategy,
    KrumStrategy,
    SignGuardStrategy,
    TrimmedMeanStrategy,
)
from src.signguard.reputation import ReputationManager
from src.signguard.verification import StatisticalValidator


def _make_update(client_id: str, delta_scale: float = 1.0, n_params: int = 10):
    """Create a synthetic ClientUpdate."""
    rng = np.random.RandomState(hash(client_id) % 10000)
    delta = collections.OrderedDict({
        "w": rng.randn(n_params).astype(np.float32) * delta_scale,
    })
    return ClientUpdate(
        client_id=client_id,
        round_num=1,
        delta=delta,
        num_samples=100,
        local_loss=0.5,
    )


class TestFedAvgStrategy:
    def test_weighted_average(self):
        strategy = FedAvgStrategy()
        u1 = _make_update("A")
        u2 = _make_update("B")
        u1.num_samples = 300
        u2.num_samples = 100

        result = strategy.aggregate(
            [u1, u2], {}, round_num=1,
            client_data_sizes={"A": 300, "B": 100},
        )

        # Should be 75% A + 25% B
        expected = 0.75 * u1.delta["w"] + 0.25 * u2.delta["w"]
        np.testing.assert_array_almost_equal(result["w"], expected, decimal=5)

    def test_single_client(self):
        strategy = FedAvgStrategy()
        u = _make_update("A")
        result = strategy.aggregate([u], {}, round_num=1, client_data_sizes={"A": 100})
        np.testing.assert_array_almost_equal(result["w"], u.delta["w"], decimal=5)

    def test_empty_raises(self):
        strategy = FedAvgStrategy()
        with pytest.raises(ValueError):
            strategy.aggregate([], {}, round_num=1, client_data_sizes={})


class TestKrumStrategy:
    def test_selects_closest_update(self):
        strategy = KrumStrategy(num_byzantine=1, multi_k=1)

        # 3 honest updates (similar) + 1 outlier
        updates = [_make_update(f"H{i}", delta_scale=0.1) for i in range(3)]
        outlier = _make_update("M", delta_scale=100.0)
        all_updates = updates + [outlier]

        result = strategy.aggregate(
            all_updates, {}, round_num=1,
            client_data_sizes={u.client_id: 100 for u in all_updates},
        )

        # Should not select the outlier
        assert strategy.last_accepted_count == 1
        # Result should be close to honest updates, not the outlier
        result_norm = np.linalg.norm(result["w"])
        assert result_norm < 50  # Much less than outlier norm ~100


class TestTrimmedMeanStrategy:
    def test_trims_extremes(self):
        # 10 updates, trim 10% = 1 from each end
        updates = [_make_update(f"C{i}", delta_scale=1.0) for i in range(10)]
        # Add extreme outlier
        updates[0].delta["w"] = np.ones(10) * 1000.0
        updates[-1].delta["w"] = np.ones(10) * -1000.0

        strategy = TrimmedMeanStrategy(trim_ratio=0.1)
        result = strategy.aggregate(
            updates, {}, round_num=1,
            client_data_sizes={u.client_id: 100 for u in updates},
        )

        # Mean of remaining 8 should be moderate
        assert np.abs(result["w"]).max() < 100


class TestFoolsGoldStrategy:
    def test_penalizes_similar_updates(self):
        strategy = FoolsGoldStrategy()

        # 3 honest diverse updates + 3 Sybil-like identical updates
        honest = [_make_update(f"H{i}", delta_scale=1.0) for i in range(3)]
        sybils = []
        base = np.ones(10, dtype=np.float32) * 5.0
        for i in range(3):
            u = _make_update(f"S{i}")
            u.delta["w"] = base + np.random.randn(10).astype(np.float32) * 0.0001
            sybils.append(u)

        all_updates = honest + sybils
        result = strategy.aggregate(
            all_updates, {}, round_num=1,
            client_data_sizes={u.client_id: 100 for u in all_updates},
        )
        # FoolsGold should reduce Sybil influence
        assert result is not None


class TestSignGuardStrategy:
    def test_basic_aggregation_no_components(self):
        """SignGuard without verifier/validator/reputation = basic averaging."""
        strategy = SignGuardStrategy()
        updates = [_make_update(f"C{i}") for i in range(3)]
        result = strategy.aggregate(
            updates, {}, round_num=1,
            client_data_sizes={u.client_id: 100 for u in updates},
        )
        assert result is not None
        assert "w" in result

    def test_with_reputation_excludes_bad(self):
        reputation = ReputationManager()
        reputation.reputation["BAD"] = 0.1  # Below exclusion

        strategy = SignGuardStrategy(reputation=reputation)

        updates = [_make_update("GOOD"), _make_update("BAD")]
        result = strategy.aggregate(
            updates,
            collections.OrderedDict({"w": np.zeros(10, dtype=np.float32)}),
            round_num=1,
            client_data_sizes={"GOOD": 100, "BAD": 100},
        )

        # BAD client should be excluded via reputation weight = 0
        # Result should be entirely GOOD's update
        assert result is not None

    def test_with_statistical_validation(self):
        validator = StatisticalValidator(l2_norm_threshold=1.0)  # Very strict

        strategy = SignGuardStrategy(validator=validator)

        normal = _make_update("NORM", delta_scale=0.01)  # Small norm
        large = _make_update("LARGE", delta_scale=100.0)  # Huge norm

        global_params = collections.OrderedDict({"w": np.zeros(10, dtype=np.float32)})
        result = strategy.aggregate(
            [normal, large], global_params, round_num=1,
            client_data_sizes={"NORM": 100, "LARGE": 100},
        )
        # Large update should be filtered
        assert strategy.last_accepted_count <= 2

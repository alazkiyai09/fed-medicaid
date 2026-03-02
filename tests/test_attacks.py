"""Tests for attack implementations."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from src.attacks.data_poison import LabelFlipAttack
from src.attacks.free_rider import FreeRiderAttack
from src.attacks.model_poison import ModelPoisonAttack
from src.attacks.random_poison import RandomPoisonAttack
from src.attacks.sybil import SybilAttack


@pytest.fixture
def honest_update():
    return collections.OrderedDict({
        "w1": np.ones((10, 5), dtype=np.float32),
        "b1": np.ones(10, dtype=np.float32) * 0.5,
    })


@pytest.fixture
def global_weights():
    return collections.OrderedDict({
        "w1": np.random.randn(10, 5).astype(np.float32) * 0.1,
        "b1": np.random.randn(10).astype(np.float32) * 0.1,
    })


class TestRandomPoisonAttack:
    def test_output_shape(self, honest_update, global_weights):
        attack = RandomPoisonAttack(intensity=1.0)
        poisoned = attack.poison_update(honest_update, global_weights, round_num=1)
        assert poisoned["w1"].shape == honest_update["w1"].shape
        assert poisoned["b1"].shape == honest_update["b1"].shape

    def test_output_differs_from_honest(self, honest_update, global_weights):
        attack = RandomPoisonAttack(intensity=1.0)
        poisoned = attack.poison_update(honest_update, global_weights, round_num=1)
        assert not np.allclose(poisoned["w1"], honest_update["w1"])


class TestModelPoisonAttack:
    def test_negated_direction(self, honest_update, global_weights):
        attack = ModelPoisonAttack(intensity=1.0)
        poisoned = attack.poison_update(honest_update, global_weights, round_num=1)
        np.testing.assert_array_almost_equal(poisoned["w1"], -honest_update["w1"])

    def test_intensity_scaling(self, honest_update, global_weights):
        attack = ModelPoisonAttack(intensity=2.0)
        poisoned = attack.poison_update(honest_update, global_weights, round_num=1)
        np.testing.assert_array_almost_equal(poisoned["w1"], -2.0 * honest_update["w1"])


class TestLabelFlipAttack:
    def test_labels_flipped(self):
        attack = LabelFlipAttack(intensity=1.0, flip_ratio=0.5)
        X = np.random.randn(100, 5)
        y = np.zeros(100)
        y[:10] = 1  # 10% fraud
        _, y_poisoned = attack.poison_training_data(X, y)
        assert not np.array_equal(y, y_poisoned)

    def test_flip_count(self):
        attack = LabelFlipAttack(intensity=1.0, flip_ratio=1.0)
        y = np.zeros(100)
        _, y_poisoned = attack.poison_training_data(np.zeros((100, 5)), y)
        assert np.sum(y != y_poisoned) == 100

    def test_features_unchanged(self):
        attack = LabelFlipAttack()
        X = np.random.randn(50, 5)
        y = np.ones(50)
        X_out, _ = attack.poison_training_data(X, y)
        np.testing.assert_array_equal(X, X_out)


class TestFreeRiderAttack:
    def test_near_zero_updates(self, honest_update, global_weights):
        attack = FreeRiderAttack(intensity=0.001)
        poisoned = attack.poison_update(honest_update, global_weights, round_num=1)
        for v in poisoned.values():
            assert np.abs(v).max() < 0.1  # Very small updates


class TestSybilAttack:
    def test_generates_multiple_updates(self, honest_update, global_weights):
        attack = SybilAttack(num_sybils=5)
        updates = attack.generate_sybil_updates(honest_update, global_weights, round_num=1)
        assert len(updates) == 5

    def test_sybil_updates_are_similar_but_not_identical(self, honest_update, global_weights):
        attack = SybilAttack(num_sybils=3)
        updates = attack.generate_sybil_updates(honest_update, global_weights, round_num=1)
        # Very similar to each other
        for i in range(1, len(updates)):
            diff = np.abs(updates[0]["w1"] - updates[i]["w1"]).max()
            assert diff < 0.01  # Tiny perturbation
        # But not identical
        assert not np.array_equal(updates[0]["w1"], updates[1]["w1"])

    def test_sybil_shapes_match(self, honest_update, global_weights):
        attack = SybilAttack(num_sybils=2)
        updates = attack.generate_sybil_updates(honest_update, global_weights, round_num=1)
        for u in updates:
            assert u["w1"].shape == honest_update["w1"].shape

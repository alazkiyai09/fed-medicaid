"""Tests for the FraudMLP model and weight utilities."""

from __future__ import annotations

import collections

import numpy as np
import torch
import pytest

from src.federation.models import (
    FraudMLP,
    apply_update_delta,
    compute_update_delta,
    deserialize_weights,
    get_model_params,
    serialize_weights,
    set_model_params,
)


class TestFraudMLP:
    def test_forward_shape(self):
        model = FraudMLP(input_dim=38)
        x = torch.randn(16, 38)
        out = model(x)
        assert out.shape == (16,)

    def test_predict_proba_range(self):
        model = FraudMLP(input_dim=38)
        x = torch.randn(32, 38)
        proba = model.predict_proba(x)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_custom_dims(self):
        model = FraudMLP(input_dim=20, hidden_dims=[64, 32])
        x = torch.randn(8, 20)
        out = model(x)
        assert out.shape == (8,)

    def test_no_batch_norm(self):
        model = FraudMLP(input_dim=10, use_batch_norm=False)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4,)


class TestWeightUtilities:
    def test_get_set_params_roundtrip(self):
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        original_params = get_model_params(model)

        # Modify and restore
        modified = collections.OrderedDict(
            (k, np.zeros_like(v)) for k, v in original_params.items()
        )
        set_model_params(model, modified)
        restored = get_model_params(model)

        for k in restored:
            np.testing.assert_array_equal(restored[k], np.zeros_like(restored[k]))

    def test_serialize_deserialize_roundtrip(self):
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        params = get_model_params(model)

        serialized = serialize_weights(params)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        deserialized = deserialize_weights(serialized)
        for k in params:
            np.testing.assert_array_almost_equal(params[k], deserialized[k])

    def test_serialize_deterministic(self):
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        params = get_model_params(model)

        s1 = serialize_weights(params)
        s2 = serialize_weights(params)
        assert s1 == s2

    def test_compute_delta(self):
        old = collections.OrderedDict({"w": np.array([1.0, 2.0, 3.0])})
        new = collections.OrderedDict({"w": np.array([1.1, 2.2, 3.3])})
        delta = compute_update_delta(new, old)
        np.testing.assert_array_almost_equal(delta["w"], [0.1, 0.2, 0.3])

    def test_apply_delta(self):
        params = collections.OrderedDict({"w": np.array([1.0, 2.0])})
        delta = collections.OrderedDict({"w": np.array([0.5, -0.5])})
        updated = apply_update_delta(params, delta)
        np.testing.assert_array_almost_equal(updated["w"], [1.5, 1.5])

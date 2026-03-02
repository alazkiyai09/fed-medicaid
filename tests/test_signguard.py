"""Tests for SignGuard — ECDSA keys, signing, verification, and reputation."""

from __future__ import annotations

import collections
import hashlib

import numpy as np
import pytest
from ecdsa import NIST256p, SigningKey

from src.federation.models import serialize_weights
from src.signguard.keys import KeyManager
from src.signguard.reputation import ReputationManager
from src.signguard.signing import UpdateSigner
from src.signguard.verification import StatisticalValidator, UpdateVerifier


@pytest.fixture
def sample_update():
    """A small synthetic model update delta."""
    return collections.OrderedDict({
        "layer1.weight": np.random.randn(16, 10).astype(np.float32),
        "layer1.bias": np.random.randn(16).astype(np.float32),
        "layer2.weight": np.random.randn(1, 16).astype(np.float32),
    })


@pytest.fixture
def sample_global_params():
    """Synthetic global model parameters."""
    return collections.OrderedDict({
        "layer1.weight": np.random.randn(16, 10).astype(np.float32) * 0.1,
        "layer1.bias": np.random.randn(16).astype(np.float32) * 0.1,
        "layer2.weight": np.random.randn(1, 16).astype(np.float32) * 0.1,
    })


class TestKeyManager:
    def test_generate_keypair(self):
        sk, vk = KeyManager.generate_keypair()
        assert sk is not None
        assert vk is not None

    def test_generate_client_keys(self, tmp_path):
        km = KeyManager(keys_dir=tmp_path)
        sk, vk = km.generate_client_keys("CA")
        assert (tmp_path / "CA_private.pem").exists()
        assert (tmp_path / "CA_public.pem").exists()

    def test_load_keys(self, tmp_path):
        km = KeyManager(keys_dir=tmp_path)
        sk_orig, vk_orig = km.generate_client_keys("NY")

        sk_loaded = km.load_private_key("NY")
        vk_loaded = km.load_public_key("NY")

        assert sk_loaded.to_pem() == sk_orig.to_pem()
        assert vk_loaded.to_pem() == vk_orig.to_pem()

    def test_generate_all(self, tmp_path):
        km = KeyManager(keys_dir=tmp_path)
        keys = km.generate_all_client_keys(["CA", "NY", "TX"])
        assert len(keys) == 3
        assert set(km.get_all_public_keys().keys()) == {"CA", "NY", "TX"}


class TestSigningAndVerification:
    def test_sign_and_verify_success(self, sample_update):
        sk, vk = KeyManager.generate_keypair()
        signer = UpdateSigner("CA", sk)
        signed = signer.sign_update(sample_update, round_num=1)

        verifier = UpdateVerifier()
        verifier.register_public_key("CA", vk)

        assert verifier.verify_signature(
            "CA", signed.update_weights, signed.signature, signed.update_hash
        )

    def test_tampered_weights_detected(self, sample_update):
        sk, vk = KeyManager.generate_keypair()
        signer = UpdateSigner("CA", sk)
        signed = signer.sign_update(sample_update, round_num=1)

        # Tamper with weights
        tampered = collections.OrderedDict(signed.update_weights)
        first_key = list(tampered.keys())[0]
        tampered[first_key] = tampered[first_key] + 1.0

        verifier = UpdateVerifier()
        verifier.register_public_key("CA", vk)

        assert not verifier.verify_signature(
            "CA", tampered, signed.signature, signed.update_hash
        )

    def test_wrong_key_rejected(self, sample_update):
        sk1, vk1 = KeyManager.generate_keypair()
        _, vk2 = KeyManager.generate_keypair()

        signer = UpdateSigner("CA", sk1)
        signed = signer.sign_update(sample_update, round_num=1)

        verifier = UpdateVerifier()
        verifier.register_public_key("CA", vk2)  # Wrong key!

        assert not verifier.verify_signature(
            "CA", signed.update_weights, signed.signature, signed.update_hash
        )

    def test_unknown_client_rejected(self, sample_update):
        sk, vk = KeyManager.generate_keypair()
        signer = UpdateSigner("CA", sk)
        signed = signer.sign_update(sample_update, round_num=1)

        verifier = UpdateVerifier()
        # Don't register any keys
        assert not verifier.verify_signature(
            "CA", signed.update_weights, signed.signature, signed.update_hash
        )


class TestStatisticalValidator:
    def test_valid_update_passes(self, sample_update, sample_global_params):
        validator = StatisticalValidator(
            l2_norm_threshold=1000.0,
            relative_norm_threshold=10.0,
            cosine_threshold=-1.0,
        )
        norms = [np.linalg.norm(np.concatenate([v.flatten() for v in sample_update.values()]))]
        result = validator.validate(sample_update, sample_global_params, norms)
        assert result.is_valid

    def test_large_norm_rejected(self, sample_update, sample_global_params):
        validator = StatisticalValidator(l2_norm_threshold=0.001)
        norms = [1.0]
        result = validator.validate(sample_update, sample_global_params, norms)
        assert not result.is_valid
        assert any("l2_norm" in f for f in result.failures)

    def test_high_relative_norm_rejected(self, sample_update, sample_global_params):
        validator = StatisticalValidator(relative_norm_threshold=0.001)
        norms = [0.0001]  # Very small median
        result = validator.validate(sample_update, sample_global_params, norms)
        assert not result.is_valid


class TestReputationManager:
    def test_initial_reputation(self):
        rm = ReputationManager()
        assert rm.get_reputation("CA") == 1.0

    def test_good_behavior_maintains_rep(self):
        rm = ReputationManager(alpha=0.3)
        for r in range(10):
            rm.update_reputation("CA", r, signature_ok=True, validation_ok=True, performance_impact=0.01)
        assert rm.get_reputation("CA") > 0.8

    def test_crypto_failure_zeroes_rep(self):
        rm = ReputationManager()
        rm.update_reputation("CA", 1, signature_ok=False, validation_ok=True, performance_impact=0.0)
        assert rm.get_reputation("CA") == 0.0

    def test_validation_failure_drops_rep(self):
        rm = ReputationManager(alpha=0.5)
        rm.update_reputation("CA", 1, signature_ok=True, validation_ok=False, performance_impact=-0.02)
        assert rm.get_reputation("CA") < 1.0

    def test_excluded_client_weight_zero(self):
        rm = ReputationManager()
        rm.reputation["BAD"] = 0.3  # Below exclusion
        w = rm.get_aggregation_weight("BAD", 100, 1000)
        assert w == 0.0

    def test_full_weight_client(self):
        rm = ReputationManager()
        rm.reputation["GOOD"] = 0.9  # Above full weight threshold
        w = rm.get_aggregation_weight("GOOD", 100, 1000)
        assert w > 0.0

    def test_flagged_for_investigation(self):
        rm = ReputationManager()
        rm.reputation["SUS"] = 0.1
        rm.update_reputation("SUS", 1, signature_ok=True, validation_ok=False, performance_impact=-0.05)
        assert rm.is_flagged("SUS")

    def test_history_tracked(self):
        rm = ReputationManager()
        rm.update_reputation("CA", 1, signature_ok=True, validation_ok=True, performance_impact=0.0)
        rm.update_reputation("CA", 2, signature_ok=True, validation_ok=True, performance_impact=0.0)
        history = rm.get_reputation_history("CA")
        assert len(history) == 2

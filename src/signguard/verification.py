"""Server-side verification for SignGuard protocol.

Two components:
1. UpdateVerifier — ECDSA signature verification (authenticity + integrity)
2. StatisticalValidator — L2 norm and cosine similarity checks (anomaly detection)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import OrderedDict

import numpy as np
from ecdsa import BadSignatureError, VerifyingKey

from src.federation.models import serialize_weights

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of statistical validation on a model update.

    Attributes:
        is_valid: Whether the update passed all checks.
        l2_norm: L2 norm of the update.
        relative_norm: L2 norm relative to the median.
        cosine_similarity: Cosine similarity with global model.
        failures: List of failed check names.
    """
    is_valid: bool
    l2_norm: float = 0.0
    relative_norm: float = 0.0
    cosine_similarity: float = 0.0
    failures: list[str] = field(default_factory=list)


class UpdateVerifier:
    """Verifies ECDSA signatures on model updates.

    Ensures both authenticity (update came from claimed client) and
    integrity (update was not tampered with after signing).
    """

    def __init__(self):
        self._public_keys: dict[str, VerifyingKey] = {}

    def register_public_key(self, client_id: str, public_key: VerifyingKey) -> None:
        """Register a client's public key for verification.

        Args:
            client_id: Client identifier.
            public_key: ECDSA verifying key.
        """
        self._public_keys[client_id] = public_key

    def verify_signature(
        self,
        client_id: str,
        update_weights: OrderedDict[str, np.ndarray],
        signature: bytes,
        claimed_hash: bytes,
    ) -> bool:
        """Verify a signed model update.

        Checks:
        1. Re-computes hash from weights (integrity check).
        2. Verifies ECDSA signature (authenticity check).

        Args:
            client_id: Identifier of the client that signed.
            update_weights: The model update delta.
            signature: ECDSA signature bytes.
            claimed_hash: The hash claimed by the client.

        Returns:
            True if both integrity and authenticity checks pass.
        """
        # Check that we have the client's public key
        if client_id not in self._public_keys:
            logger.warning("No public key registered for client %s", client_id)
            return False

        # Re-compute hash from weights (integrity)
        serialized = serialize_weights(update_weights)
        expected_hash = hashlib.sha256(serialized).digest()

        if expected_hash != claimed_hash:
            logger.warning(
                "Integrity check failed for %s: hash mismatch", client_id
            )
            return False

        # Verify ECDSA signature (authenticity)
        try:
            self._public_keys[client_id].verify(signature, claimed_hash)
            return True
        except BadSignatureError:
            logger.warning("Signature verification failed for %s", client_id)
            return False


class StatisticalValidator:
    """Validates model updates using statistical checks.

    Three checks (spec §6.5):
    1. Absolute L2 norm: ‖Δ‖ < threshold
    2. Relative L2 norm: ‖Δ‖ / median(‖Δ_j‖) < threshold
    3. Cosine similarity: cos(Δ, W) > threshold

    Args:
        l2_norm_threshold: Maximum absolute L2 norm (default: 100.0).
        relative_norm_threshold: Maximum relative norm (default: 5.0).
        cosine_threshold: Minimum cosine similarity (default: -0.5).
    """

    def __init__(
        self,
        l2_norm_threshold: float = 100.0,
        relative_norm_threshold: float = 5.0,
        cosine_threshold: float = -0.5,
    ):
        self.l2_norm_threshold = l2_norm_threshold
        self.relative_norm_threshold = relative_norm_threshold
        self.cosine_threshold = cosine_threshold

    def _flatten(self, params: OrderedDict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([v.flatten() for v in params.values()])

    def validate(
        self,
        update: OrderedDict[str, np.ndarray],
        global_params: OrderedDict[str, np.ndarray],
        all_norms: list[float],
    ) -> ValidationResult:
        """Validate a model update against statistical thresholds.

        Args:
            update: Model update delta.
            global_params: Current global model parameters.
            all_norms: L2 norms of all updates in this round.

        Returns:
            ValidationResult with pass/fail status and details.
        """
        flat_update = self._flatten(update)
        flat_global = self._flatten(global_params)

        failures: list[str] = []

        # Check 1: Absolute L2 norm
        l2_norm = float(np.linalg.norm(flat_update))
        if l2_norm >= self.l2_norm_threshold:
            failures.append(f"l2_norm={l2_norm:.2f} >= {self.l2_norm_threshold}")

        # Check 2: Relative L2 norm
        median_norm = float(np.median(all_norms)) if all_norms else 1.0
        relative_norm = l2_norm / max(median_norm, 1e-10)
        if relative_norm >= self.relative_norm_threshold:
            failures.append(
                f"relative_norm={relative_norm:.2f} >= {self.relative_norm_threshold}"
            )

        # Check 3: Cosine similarity with global model
        norm_update = np.linalg.norm(flat_update)
        norm_global = np.linalg.norm(flat_global)
        if norm_update > 0 and norm_global > 0:
            cosine_sim = float(
                np.dot(flat_update, flat_global) / (norm_update * norm_global)
            )
        else:
            cosine_sim = 0.0

        if cosine_sim < self.cosine_threshold:
            failures.append(
                f"cosine_sim={cosine_sim:.4f} < {self.cosine_threshold}"
            )

        return ValidationResult(
            is_valid=len(failures) == 0,
            l2_norm=l2_norm,
            relative_norm=relative_norm,
            cosine_similarity=cosine_sim,
            failures=failures,
        )

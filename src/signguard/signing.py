"""Client-side update signing for SignGuard protocol.

Signs the SHA-256 hash of deterministically serialized model update
weights using ECDSA (NIST P-256), ensuring both authenticity and
integrity of federated learning updates.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import OrderedDict

import numpy as np
from ecdsa import SigningKey

from src.federation.models import serialize_weights

logger = logging.getLogger(__name__)


@dataclass
class SignedUpdate:
    """A model update with cryptographic signature.

    Attributes:
        client_id: Identifier of the signing client.
        round_num: Federation round number.
        update_weights: The model update delta.
        update_hash: SHA-256 hash of the serialized update.
        signature: ECDSA signature over the hash.
        metadata: Additional metadata (e.g., local loss).
    """
    client_id: str
    round_num: int
    update_weights: OrderedDict[str, np.ndarray]
    update_hash: bytes
    signature: bytes
    metadata: dict


class UpdateSigner:
    """Signs model updates with ECDSA for the SignGuard protocol.

    Each federated client uses this to cryptographically sign their
    model weight updates before sending to the server.

    Args:
        client_id: Unique identifier for this client.
        private_key: ECDSA signing key (NIST P-256).
    """

    def __init__(self, client_id: str, private_key: SigningKey):
        self.client_id = client_id
        self.private_key = private_key

    def sign_update(
        self,
        update_weights: OrderedDict[str, np.ndarray],
        round_num: int,
        metadata: dict | None = None,
    ) -> SignedUpdate:
        """Sign a model update.

        1. Deterministically serialize the weight update.
        2. Compute SHA-256 hash.
        3. Sign the hash with ECDSA.

        Args:
            update_weights: Model update delta (new - old params).
            round_num: Current federation round.
            metadata: Optional metadata to include.

        Returns:
            SignedUpdate with hash and signature.
        """
        if metadata is None:
            metadata = {}

        # Deterministic serialization
        serialized = serialize_weights(update_weights)

        # SHA-256 hash
        update_hash = hashlib.sha256(serialized).digest()

        # ECDSA signature
        signature = self.private_key.sign(update_hash)

        logger.debug(
            "Signed update for client %s, round %d (hash: %s…)",
            self.client_id, round_num, update_hash[:8].hex(),
        )

        return SignedUpdate(
            client_id=self.client_id,
            round_num=round_num,
            update_weights=update_weights,
            update_hash=update_hash,
            signature=signature,
            metadata=metadata,
        )

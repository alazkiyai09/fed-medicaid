"""ECDSA key management for SignGuard protocol.

Handles generation, storage, and loading of NIST P-256 key pairs
for cryptographic signing and verification of model updates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ecdsa import NIST256p, SigningKey, VerifyingKey

logger = logging.getLogger(__name__)


class KeyManager:
    """Manages ECDSA key pairs for federated clients.

    Uses NIST P-256 (secp256r1) curve — standard for government/healthcare.
    Each state client generates a key pair; the server stores public keys.

    Args:
        keys_dir: Directory to store/load keys.
    """

    CURVE = NIST256p

    def __init__(self, keys_dir: str | Path = "data/keys"):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        self._public_keys: dict[str, VerifyingKey] = {}

    @staticmethod
    def generate_keypair() -> tuple[SigningKey, VerifyingKey]:
        """Generate a new ECDSA key pair.

        Returns:
            Tuple of (private_key, public_key).
        """
        sk = SigningKey.generate(curve=NIST256p)
        vk = sk.get_verifying_key()
        return sk, vk

    def generate_client_keys(self, client_id: str) -> tuple[SigningKey, VerifyingKey]:
        """Generate and save keys for a specific client.

        Args:
            client_id: Unique identifier for the client (e.g., state code).

        Returns:
            Tuple of (private_key, public_key).
        """
        sk, vk = self.generate_keypair()

        # Save private key (client-side)
        sk_path = self.keys_dir / f"{client_id}_private.pem"
        sk_path.write_bytes(sk.to_pem())

        # Save public key (server-side)
        vk_path = self.keys_dir / f"{client_id}_public.pem"
        vk_path.write_bytes(vk.to_pem())

        self._public_keys[client_id] = vk
        logger.debug("Generated keys for client %s", client_id)

        return sk, vk

    def load_private_key(self, client_id: str) -> SigningKey:
        """Load a client's private key from disk.

        Args:
            client_id: Client identifier.

        Returns:
            ECDSA signing key.
        """
        sk_path = self.keys_dir / f"{client_id}_private.pem"
        return SigningKey.from_pem(sk_path.read_bytes())

    def load_public_key(self, client_id: str) -> VerifyingKey:
        """Load a client's public key from disk.

        Args:
            client_id: Client identifier.

        Returns:
            ECDSA verifying key.
        """
        if client_id in self._public_keys:
            return self._public_keys[client_id]

        vk_path = self.keys_dir / f"{client_id}_public.pem"
        vk = VerifyingKey.from_pem(vk_path.read_bytes())
        self._public_keys[client_id] = vk
        return vk

    def register_public_key(self, client_id: str, public_key: VerifyingKey) -> None:
        """Register a client's public key directly (no disk I/O).

        Args:
            client_id: Client identifier.
            public_key: ECDSA verifying key.
        """
        self._public_keys[client_id] = public_key

    def get_all_public_keys(self) -> dict[str, VerifyingKey]:
        """Return all registered public keys."""
        return dict(self._public_keys)

    def generate_all_client_keys(
        self, client_ids: list[str],
    ) -> dict[str, tuple[SigningKey, VerifyingKey]]:
        """Generate keys for all clients.

        Args:
            client_ids: List of client identifiers.

        Returns:
            Dictionary mapping client_id → (private_key, public_key).
        """
        keys = {}
        for cid in client_ids:
            keys[cid] = self.generate_client_keys(cid)
        logger.info("Generated keys for %d clients", len(keys))
        return keys

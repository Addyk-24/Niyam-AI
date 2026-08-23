"""
SHA-256 commitment over the Intent Contract.

Sealing happens once, at session initialization. Every later layer checks
against the resulting IntentHash.
"""

import hashlib
import json
import logging
from typing import List

from pydantic import BaseModel

logger = logging.getLogger("niyam.seal")


class SealViolation(Exception):
    """Raised when sealing or verification is used incorrectly."""


class HashIntentContract(BaseModel):
    hash: str = ""
    agent_name: str
    user_task: str
    allowed_tools: List[str]
    forbidden_tools: List[str]

    def _compute_hash(self) -> str:
        """
        SHA-256 over the canonicalized contract, excluding the hash field
        itself (which is empty at compute time and would otherwise make the
        digest self-referential).

        Must stay byte-identical to IntentContract.intent_hash().
        """
        try:
            data = self.model_dump(exclude={"hash"})
        except AttributeError:          # pydantic v1 fallback
            data = self.dict(exclude={"hash"})
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class IntentSeal:
    """
    Single-use sealer. One instance seals one contract.

    Attempting to seal twice raises rather than warning: silently returning
    the first contract while the caller believes it sealed the second is the
    precise failure mode this class exists to prevent.
    """

    def __init__(self):
        self.intent: HashIntentContract | None = None
        self.is_sealed: bool = False

    def seal_intent(self, intent: HashIntentContract) -> HashIntentContract:
        if self.is_sealed:
            raise SealViolation(
                "This IntentSeal has already sealed a contract "
                f"(hash={self.intent.hash[:16]}...). Construct a new IntentSeal "
                "for a new session rather than reusing this one."
            )

        intent.hash = intent._compute_hash()
        self.intent = intent
        self.is_sealed = True

        logger.info(f"Intent sealed | agent={intent.agent_name} | "
                    f"hash={intent.hash[:16]}...")
        return self.intent

    def verify_seal(self, intent: HashIntentContract) -> bool:
        """
        Recompute the contract's hash and compare against the stored value.
        Returns False on mismatch or on an unsealed contract; never raises,
        so callers can use it as a guard.
        """
        if not intent.hash:
            logger.error("No hash present on intent - cannot verify seal.")
            return False

        expected = intent._compute_hash()

        if expected == intent.hash:
            logger.info("Intent seal verified.")
            return True

        logger.error(
            f"Intent seal verification FAILED - hash mismatch. "
            f"stored={intent.hash[:16]}... recomputed={expected[:16]}... "
            f"The contract was modified after sealing."
        )
        return False
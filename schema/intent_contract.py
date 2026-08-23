"""
the object an agent's permissions are bound to.

One contract per session. Hashed once at initialization by IntentSeal; the
resulting IntentHash is the commitment every downstream layer checks against.
"""

import hashlib
import json
from typing import List

from pydantic import BaseModel


class IntentContract(BaseModel):
    agent_name: str
    user_task: str
    allowed_tools: List[str]
    forbidden_tools: List[str]

    def intent_hash(self) -> str:
        """
        Deterministic SHA-256 over the canonicalized full contract.

        Canonicalization is json.dumps(..., sort_keys=True) over the complete
        model dump, so key ordering cannot produce two hashes for the same
        contract.
        """
        try:
            data = self.model_dump()
        except AttributeError:
            data = self.dict()

        normalized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()
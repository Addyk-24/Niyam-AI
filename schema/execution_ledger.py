import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

class ExecutionLedger:
    def __init__(self):
        self.ledge: list[str] = []

    def add_entry(
        self,
        intent_hash: str,
        tool: str,
        status: Literal["ALLOWED", "BLOCKED", "ERROR"],
        reason: str,
    ) -> dict:
        
        entry = {
            "intent_hash": intent_hash,
            "tool": tool,
            "status": status,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


        previous_hash = self.ledge[-1]["entry_hash"] if self.ledge else "GENESIS"

        entry["previous_hash"] = previous_hash

        hash_input = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        entry["entry_hash"] = entry_hash

        self.ledge.append(entry)

        return entry

    def verify(self) -> bool:
        """
        Walk the ledge and verify every link is unmodified.
        Useful for tamper-detection in audits.
        """

        for i,entry in enumerate(self.ledge):

            store_hash = entry.pop("entry_hash")

            recomputed = hashlib.sha256(
                json.dumps(entry,sort_keys=True).encode()
                ).hexdigest()
            
            entry["entry_hash"] = store_hash

            if store_hash != recomputed:
                return False
            
            if i>0:
                expected_prev = self.ledge[i-1]["entry_hash"]
                if entry["previous_hash"] != expected_prev:
                    return False

        return True
    
    def get_violations(self) -> list[dict]:
        """Return only BLOCKED entries — useful for the trust dashboard."""
        return [e for e in self.ledge if e["status"] == "BLOCKED"]

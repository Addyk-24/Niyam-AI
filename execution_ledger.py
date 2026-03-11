import hashlib
import json
from datetime import datetime

class ExecutionLedger:
    def __init__(self):
        self.ledge = []

    def add_entry(self, intent_hash, tool, status, reason):
        entry = {
            "intent_hash": intent_hash,
            "tool": tool,
            "status": status,
            "reason": reason,
            "timestamp": str(datetime.utcnow())
        }


        previous_hash = self.chain[-1]["entry_hash"] if self.chain else "GENESIS"


        entry["previous_hash"] = previous_hash

        hash_input = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        entry["entry_hash"] = entry_hash

        self.ledge.append(entry)

        return entry

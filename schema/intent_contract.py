
# Code → Build → Test → Agent Integrity Gate → Deploy

import hashlib
from pydantic import BaseModel
from typing import List,Optional
import json


class IntentContract(BaseModel):
    agent_name: str
    user_task: str
    allowed_tools: List[str]
    forbidden_tools: List[str]


    def intent_hash(self):
        """Deterministic hash of the full contract dict."""

        try:
            data = self.model_dump()
        except AttributeError:
            data = self.dict()

        normalized = json.dumps(data, sort_keys=True)

        return hashlib.sha256(normalized.encode()).hexdigest()

    def seal(self) -> str:
        """
        Create immutable hash of the intent 
        
        """
        content = (
            self.user_task
            + ''.join(sorted(self.allowed_tools))
            + ''.join(sorted(self.forbidden_tools))
        )

        return hashlib.sha256(content.encode()).hexdigest()
    
    
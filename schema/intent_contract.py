
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
        normalized = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def seal(self) -> str:
        """
        Create immutable hash of the intent 
        
        """
        content = (
            self.user_task +
            ''.join(sorted(self.allowed_tool)) +
            ''.join(sorted(self.forbidden_tool))
        )

        return hashlib.sha256(content.encode()).hexdigest()
    
    
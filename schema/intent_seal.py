import logging
from typing import List
from pydantic import BaseModel
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HashIntentContract(BaseModel):
    hash: str = ""
    agent_name: str
    user_task: str
    allowed_tools: List[str]
    forbidden_tools: List[str]

    def _compute_hash(self) -> str:
        try:
            data = self.model_dump(exclude={"hash"})
        except AttributeError:
            data = self.dict(exclude={"hash"})
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class IntentSeal:
    def __init__(self):
        self.intent: HashIntentContract | None = None
        self.is_sealed: bool = False

    def seal_intent(self,intent:HashIntentContract) -> HashIntentContract:
        if self.is_sealed:
            logger.warning("Intent is already Sealed. No need to Seal again")
            return self.intent

        intent.hash = intent._compute_hash()
        self.intent = intent
        self.is_sealed = True

        logger.info(f"Intent sealed | agent={intent.agent_name} | hash={intent.hash[:16]}...")
        return self.intent

    def verify_seal(self,intent:HashIntentContract) -> bool:

        if not intent.hash:
            logger.error("No hash found in the intent. CANT VERIFY SEAL.You are a child")
            return False
        
        expected = intent._compute_hash()


        if expected == intent.hash:
            logger.info("Intent seal verified successfully...")
            return True
        else:
            logger.error("Intent seal verification failed! Hash mismatch.")
            return False
        
        
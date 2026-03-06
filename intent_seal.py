import logging
from typing import List
from pydantic import BaseModel
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HashIntentContract(BaseModel):
    hash: str
    agent_name: str
    user_task: str
    allowed_tools: List[str]
    forbidden_tools: List[str]

class IntentSeal:
    def __init__(self):
        self.intent = None
        self.isSealed = False

    def seal_intent(self,intent:HashIntentContract):
        if self.isSealed:
            logger.warning("Intent is already Sealed. No need to Seal again")

        contract_hash = hashlib.sha256(intent.json().encode()).hexdigest()
        intent.hash = contract_hash
        self.intent = intent
        self.isSealed = True
        
        logger.info(f"Intent sealed with Hash for Intent: {self.intent}")

        return self.intent

    def verify_seal(self,intent:HashIntentContract):
        if not intent.hash:
            logger.error("No hash found in the intent. CANT VERIFY SEAL.You are a child")
            return False
        
        new_hash = self.seal_intent(intent).hash

        if new_hash == intent.hash:
            logger.info("Intent seal verified successfully...")
            return True
        else:
            logger.error("Intent seal verification failed! Hash mismatch.")
            return False
        
        
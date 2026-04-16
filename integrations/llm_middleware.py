from __future__ import annotations

import json
import logging
import re
import sys
import os
from datetime import datetime, timezone
from typing import Any, Callable, Type

sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))


import jsonschema
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
 
from policy.policy_loader import PolicyLoader
from schema.control_flow import ControlFlowIntegrity, ControlFlowViolation
from schema.execution_ledger import ExecutionLedger
from schema.intent_contract import IntentContract
from schema.intent_seal import HashIntentContract, IntentSeal
from schema.tool_gate import ToolAuthorizationError, ToolAuthorityGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("niyam.middleware")


class IntentViolation(Exception):
    """
    Raised when a tool call violates the intent contract.
    Stops the agent dead — LangChain catches this and surfaces it to the user.
    """
    def __init__(self, tool: str, reason: str, intent_hash: str):
        self.tool = tool
        self.reason = reason
        self.intent_hash = intent_hash
        super().__init__(
            f"Intent violation in tool '{tool}': {reason} (hash: {intent_hash})"
        )



# PAYLOAD INSPECTOR  (heuristic injection detection)

_INJECTION_PATTERNS = [
    # SQL
    (re.compile(r"'\s*;\s*", re.I),          "SQL injection: quote-semicolon"),
    (re.compile(r"\bdrop\s+table\b", re.I),  "SQL injection: DROP TABLE"),
    (re.compile(r"\bselect\s+\*\b", re.I),   "SQL injection: SELECT *"),
    (re.compile(r"--\s*$", re.M),            "SQL injection: comment marker"),
    # XSS
    (re.compile(r"<script", re.I),           "XSS: script tag"),
    (re.compile(r"onerror\s*=", re.I),       "XSS: event handler"),
    (re.compile(r"javascript:", re.I),       "XSS: javascript URI"),
    # Path traversal
    (re.compile(r"\.\./"),                   "Path traversal"),
    (re.compile(r"/etc/(passwd|shadow)"),    "Sensitive file access"),
    # Control chars
    (re.compile(r"\x00"),                    "Null byte injection"),
    (re.compile(r"\u200b"),                  "Zero-width space injection"),
]
 
_MAX_FIELD_LENGTH = 200

            
class PayloadInspector:

    @staticmethod
    def inspect_payload(tool_name: str, payload: dict) -> None:
        """
        Scan every string field in the payload for injection patterns.
        Raises ToolAuthorizationError if anything suspicious is found.
        """

        for field_name,value in payload.items():
            if not isinstance(value,str):
                continue

            if len(value) > _MAX_FIELD_LENGTH:
                raise ToolAuthorizationError(
                    f"Field '{field_name}' exceeds max length "
                    f"({len(value)} > {_MAX_FIELD_LENGTH} chars)"
                )
            
            for pattern, label in _INJECTION_PATTERNS:
                if pattern.search(value):
                    raise ToolAuthorizationError(
                        f"Injection detected in '{field_name}': {label}"
                    )
                

        if tool_name == "proceed_transaction":
            amount = payload.get("amount")
            if isinstance(amount, float):
                import math
                if math.isnan(amount) or math.isinf(amount):
                    raise ToolAuthorizationError("Invalid amount: NaN or Inf")
            if isinstance(amount, (int, float)) and amount < 0:
                raise ToolAuthorizationError("Negative transaction amount not permitted")
        

class IntegrityGate:
    """
    Three sequential layers. Failure at any layer blocks execution.
    Every outcome (allowed or blocked) is written to the ledger.
    """
    def __init__(self, auth_gate: ToolAuthorityGate, ledger: ExecutionLedger,
                 flow: ControlFlowIntegrity, intent_hash: str):
        self._auth_gate   = auth_gate
        self._ledger      = ledger
        self._flow        = flow
        self._intent_hash = intent_hash

    def check(self,tool_name:str,payload:dict) -> None:
        def block(reason: str, layer: str) -> None:
            self._ledger.add_entry(self._intent_hash, tool_name, "BLOCKED", reason)
            logger.warning(f"[{layer}] BLOCKED '{tool_name}': {reason}")
            raise IntentViolation(tool_name, reason, self._intent_hash, layer)
        
        # Layer 1 -this is schema of allowlist and denylist
        try:
            self._auth_gate.authorize(tool_name,payload)
        except (ToolAuthorizationError, jsonschema.ValidationError) as e:
            block(str(e), "allowlist")

        # Layer 2 — payload injection scan
        try:
            PayloadInspector.inspect_payload(tool_name,payload)
        except ToolAuthorizationError as e:
            block(str(e), "payload-inspection")

        # Layer 3 — control flow sequence

        if not self._flow.is_complete():
            try:
                self._flow.validate_step(tool_name)
            except ControlFlowViolation as e:
                block(str(e), "control-flow")

        self._ledger.add_entry(
            self._intent_hash, tool_name, "ALLOWED", "Passed all gate layers"
        )
        logger.info(f"ALLOWED '{tool_name}'")


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
 
    def register(self, name: str, func: Callable, description: str = "") -> None:
        self._tools[name] = func
        logger.info(f"  Registered tool: '{name}'")
 
    def get(self, name: str) -> Callable | None:
        return self._tools.get(name)
    

class AgentIntegritySession:
    """
    One session = one user task = one sealed intent contract.
 
    Example:
        session = AgentIntegritySession.from_policy(
            policy_path="policy/guardrails.yaml",
            user_task="Process payment of $200 to Alice",
        )
        guarded = session.wrap_tools([proceed_transaction_tool, send_email_tool])
        # Use guarded in your LangChain agent
        agent = initialize_agent(guarded, llm, ...)
    """

    def __init__(self, contract: IntentContract, sealed: HashIntentContract,
                 gate: ToolAuthorityGate, ledger: ExecutionLedger,
                 flow: ControlFlowIntegrity):
        self.contract = contract
        self.sealed = sealed
        self.gate = gate
        self.ledger = ledger
        self.flow = flow
        self.intent_hash = sealed.intent_hash
        self._registry = ToolRegistry()
        self._ig = IntegrityGate(gate,ledger,flow,self.intent_hash)

        logger.info("  Niyam-AI Integrity Session started")
        logger.info(f"  Agent      : {contract.agent_name}")
        logger.info(f"  Task       : {contract.user_task}")
        logger.info(f"  Allowed    : {contract.allowed_tools}")
        logger.info(f"  Forbidden  : {contract.forbidden_tools}")
        logger.info(f"  IntentHash : {self.intent_hash[:24]}...")


    @classmethod
    def from_policy(cls, policy_path: str, user_task: str) -> AgentIntegritySession:
        """
        Load guardrails.yaml → build + seal an IntentContract → return session.
        This is the only constructor you need in normal use.
        """
        policy = PolicyLoader.load(policy_path)

        raw_contract = IntentContract(
            agent_name=policy["agent_name"],
            user_task=user_task,
            allowed_tools=policy["allowed_tools"],
            forbidden_tools=policy["forbidden_tools"]
        )

        sealer = IntentSeal()
        sealed = sealer.seal_intent(raw_contract)

        if not sealer.verify_seal(sealed):
            raise RuntimeError("Intent seal verification failed — session aborted.")
        
        contract = IntentContract(
            agent_name=sealed.agent_name,
            user_task=sealed.user_task,
            allowed_tools=sealed.allowed_tools,
            forbidden_tools=sealed.forbidden_tools
        )

        gate = ToolAuthorityGate(contract)
        ledger = ExecutionLedger()
        flow = ControlFlowIntegrity(allowed_sequence=policy["allowed_tools"])

        return cls(contract,sealed,gate,ledger,flow)
    
    def register_tool(self,name:str,func:Callable,description:str="") -> None:

        """Register any Python callable as a guarded tool."""
        self._registry.register(name,funct,description)

    def call_tool(self,tool_name:str,**kwargs) -> Any:
        """
        Primary call interface — framework-agnostic.
        Gate runs first. Tool only executes if all layers pass.
        Raises IntentViolation on any block.
        """
        logger.info(f"call_tool: {tool_name}{kwargs}")
        self._ig.check(tool_name,kwargs)

        func = self._registry.get(tool_name)

        if func is None:

            raise KeyError(
                f"Tool '{tool_name}' passed the gate but is not registered. "
                f"Call session.register_tool('{tool_name}', fn) first."
            )
        
        return func(**kwargs)
    
    
    def print_ledger(self) -> None:
        """Print the full audit ledger — call this at the end of a session."""
        print("  EXECUTION LEDGER — Tamper-proof audit trail")
        for i, entry in enumerate(self.ledger.ledge):
            icon = "COOL" if entry["status"] == "ALLOWED" else "NAHHH"
            print(f"\n  [{i}] {icon} {entry['status']}")
            print(f"       tool   : {entry['tool']}")
            print(f"       reason : {entry['reason']}")
            print(f"       time   : {entry['timestamp']}")
            print(f"       hash   : {entry['entry_hash'][:24]}...")
 
        chain_ok = self.ledger.verify()
        violations = self.ledger.get_violations()
        print(f"  Chain integrity : {'VALID' if chain_ok else 'TAMPERED'}")
        print(f"  Violations      : {len(violations)}")
 
    def session_summary(self) -> dict:
        return {
            "agent": self.contract.agent_name,
            "task": self.contract.user_task,
            "intent_hash": self.intent_hash,
            "total_calls": len(self.ledger.ledge),
            "violations": len(self.ledger.get_violations()),
            "chain_valid": self.ledger.verify(),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }




# class GuardedTool(BaseTool):
#     """
#     A LangChain-compatible tool wrapper that enforces:
#       1. Allowlist / denylist check (ToolAuthorityGate)
#       2. Payload injection inspection
#       3. Control flow sequence validation
#       4. Immutable audit log entry (ExecutionLedger)
 
#     The agent never touches the real tool unless all checks pass.
#     """

#     name: str
#     description: str
#     gate: Any = Field(exclude=True)
#     ledger: Any = Field(exclude=True)
#     flow: Any = Field(exclude=True)
#     intent_hash: str = Field(exclude=True)
#     _real_func: Callable = None

#     class Config:
#         arbittary_types_allowed = True

    
#     def __init__(self, real_tool: BaseTool, gate: ToolAuthorityGate,
#                  ledger: ExecutionLedger, flow: ControlFlowIntegrity,
#                  intent_hash: str):
#         super().__init__(
#             name=real_tool.name,
#             description=real_tool.description,
#             gate=gate,
#             ledger=ledger,
#             flow=flow,
#             intent_hash=intent_hash,
#         )

#         self._real_func = getattr(real_tool, "func", real_tool._run)


#     def _run(self, *args, **kwargs) -> str:
#         """Called by LangChain when the agent decides to use this tool."""
#         tool_name = self.name
 
#         # Normalise args → payload dict
#         payload: dict = {}
#         if args:
#             # LangChain sometimes passes a raw string as first arg
#             raw = args[0]
#             if isinstance(raw, str):
#                 try:
#                     payload = json.loads(raw)
#                 except json.JSONDecodeError:
#                     payload = {"input": raw}
#             elif isinstance(raw, dict):
#                 payload = raw
#         payload.update(kwargs)
 
#         logger.info(f"Agent attempting: {tool_name}({payload})")
 
#         try:
#             # Gate 1: Allowlist / denylist
#             self.gate.authorize(tool_name, payload)
 
#             # Gate 2: Payload injection inspection
#             inspect_payload(tool_name, payload)
 
#             # Gate 3: Control flow sequence
#             if not self.flow.is_complete():
#                 self.flow.validate_step(tool_name)
 
#         except (ToolAuthorizationError, jsonschema.ValidationError,
#                 ControlFlowViolation) as e:
#             reason = str(e)
#             self.ledger.add_entry(self.intent_hash, tool_name, "BLOCKED", reason)
#             logger.warning(f"BLOCKED {tool_name}: {reason}")
#             raise IntentViolation(tool_name, reason, self.intent_hash)
 
#         except Exception as e:
#             reason = f"Unexpected gate error: {e}"
#             self.ledger.add_entry(self.intent_hash, tool_name, "BLOCKED", reason)
#             logger.error(f"GATE ERROR {tool_name}: {reason}")
#             raise IntentViolation(tool_name, reason, self.intent_hash)
 
#         # All gates passed → execute the real tool
#         self.ledger.add_entry(self.intent_hash, tool_name, "ALLOWED",
#                               "Passed all integrity checks")
#         logger.info(f"ALLOWED {tool_name} — executing")
 
#         result = self._real_func(**payload)
#         logger.info(f"   Result: {str(result)[:120]}")
#         return result
 
#     async def _arun(self, *args, **kwargs) -> str:
#         """Async version — delegates to sync for now."""
#         return self._run(*args, **kwargs)
    
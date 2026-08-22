"""
Niyam-AI runtime enforcement

SINGLE SOURCE OF TRUTH. One session class, five layers, proof-gated
execution. Supersedes the earlier three-layer AgentIntegritySession and
the separate niyam_pipeline.py (both removed).

    Layer 1  allowlist / denylist + schema   ToolAuthorityGate
    Layer 2  payload injection scan          PayloadInspector
    Layer 3  control-flow sequence           ControlFlowIntegrity
    Layer 4  semantic classification         ProvableJudge
    Layer 5  zk-SNARK prove + verify         ZKProver

The registered tool function is invoked ONLY after Layer 5 returns
verified=True.

"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jsonschema
import torch
import torch.nn as nn
import ezkl

from benchmark_eval.judge_model import extract_features, JudgeInput
from policy.policy_loader import PolicyLoader
from schema.control_flow import ControlFlowIntegrity, ControlFlowViolation
from schema.execution_ledger import ExecutionLedger
from schema.intent_contract import IntentContract
from schema.intent_seal import HashIntentContract, IntentSeal
from schema.tool_gate import ToolAuthorizationError, ToolAuthorityGate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("niyam.middleware")

HERE = Path(__file__).resolve().parent.parent
EZKL_DIR = HERE / "ezkl_pipeline"

class IntentViolation(Exception):
    """Raised when a tool call fails any enforcement layer."""

    def __init__(self, tool: str, reason: str, intent_hash: str, layer: str | None = None):
        self.tool = tool
        self.reason = reason
        self.intent_hash = intent_hash
        self.layer = layer
        layer_text = f" at layer '{layer}'" if layer else ""
        super().__init__(
            f"Intent violation in tool '{tool}'{layer_text}: {reason} (hash: {intent_hash})"
        )

_INJECTION_PATTERNS = [
    (re.compile(r"'\s*;\s*", re.I),          "SQL injection: quote-semicolon"),
    (re.compile(r"\bdrop\s+table\b", re.I),  "SQL injection: DROP TABLE"),
    (re.compile(r"\bselect\s+\*\b", re.I),   "SQL injection: SELECT *"),
    (re.compile(r"--\s*$", re.M),            "SQL injection: comment marker"),
    (re.compile(r"<script", re.I),           "XSS: script tag"),
    (re.compile(r"onerror\s*=", re.I),       "XSS: event handler"),
    (re.compile(r"javascript:", re.I),       "XSS: javascript URI"),
    (re.compile(r"\.\./"),                   "Path traversal"),
    (re.compile(r"/etc/(passwd|shadow)"),    "Sensitive file access"),
    (re.compile(r"\x00"),                    "Null byte injection"),
    (re.compile(r"\u200b"),                  "Zero-width space injection"),
]

_MAX_FIELD_LENGTH = 200


class PayloadInspector:
    @staticmethod
    def inspect_payload(tool_name: str, payload: dict) -> None:
        for field_name, value in payload.items():
            if not isinstance(value, str):
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
            if isinstance(amount, float) and (math.isnan(amount) or math.isinf(amount)):
                raise ToolAuthorizationError("Invalid amount: NaN or Inf")
            if isinstance(amount, (int, float)) and amount < 0:
                raise ToolAuthorizationError("Negative transaction amount not permitted")

    @staticmethod
    def inspect(tool_name: str, payload: dict) -> None:
        """Backward-compatible alias used by benchmark scripts."""
        PayloadInspector.inspect_payload(tool_name, payload)
class JudgeFFN(nn.Module):
    """11 inputs -> 8 hidden -> 2 classes. Sized to keep the circuit small."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class ProvableJudge:
    """Same forward pass that decides is the one committed to in the circuit."""

    def __init__(self, weights_path: Path | None = None):
        wp = weights_path or (EZKL_DIR / "judge_ffn.pt")
        if not wp.exists():
            raise FileNotFoundError(
                f"Judge weights not found at {wp}. "
                f"Run: python ezkl_pipeline/train_pytorch_judge.py"
            )
        self.model = JudgeFFN()
        self.model.load_state_dict(torch.load(wp, map_location="cpu"))
        self.model.eval()
        logger.info(f"  Judge loaded: {wp.name}")

    def features(self, instruction: str, tool_name: str,
                 payload: dict, allowed_tools: list) -> list:
        return extract_features(JudgeInput(instruction, tool_name, payload, allowed_tools))

    def predict(self, feats: list) -> tuple:
        """Returns (decision, confidence). decision: 1=safe, 0=unsafe."""
        with torch.no_grad():
            probs = torch.softmax(
                self.model(torch.tensor([feats], dtype=torch.float32)), dim=1)[0]
        decision = int(torch.argmax(probs).item())
        return decision, float(probs[decision].item())
class ZKProver:
    """
    Halo2-KZG proof that the Judge's forward pass on THIS feature vector
    produced THIS decision, verified before the tool may run.

    Requires one-time setup artifacts from ezkl_pipeline/run_ezkl_pipeline.py.
    Set enabled=False to run the first four layers only (useful for tests
    and for measuring the non-cryptographic path in isolation).
    """

    def __init__(self, artifacts_dir: Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.dir = artifacts_dir or EZKL_DIR
        self.compiled = self.dir / "network.compiled"
        self.settings = self.dir / "settings.json"
        self.pk = self.dir / "pk.key"
        self.vk = self.dir / "vk.key"
        self.srs = self.dir / "kzg.srs"

        if not enabled:
            logger.warning("  ZK layer DISABLED - no proofs will be generated")
            return

        missing = [p.name for p in (self.compiled, self.settings, self.pk, self.vk, self.srs)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"ZK setup artifacts missing: {missing}. "
                f"Run: python ezkl_pipeline/run_ezkl_pipeline.py"
            )
        self._ezkl = ezkl
        logger.info("  ZK layer ready (Halo2-KZG via EZKL)")

    def prove_and_verify(self, feats: list, witness_path: Path,
                         proof_path: Path) -> tuple:
        if not self.enabled:
            return True, 0.0, 0.0

        input_json = self.dir / "runtime_input.json"
        input_json.write_text(json.dumps({"input_data": [feats]}))

        self._ezkl.gen_witness(str(input_json), str(self.compiled), str(witness_path))

        t0 = time.perf_counter()
        self._ezkl.prove(str(witness_path), str(self.compiled), str(self.pk),
                         str(proof_path), srs_path=str(self.srs))
        prove_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ok = self._ezkl.verify(str(proof_path), str(self.settings),
                               str(self.vk), srs_path=str(self.srs))
        verify_ms = (time.perf_counter() - t0) * 1000

        return bool(ok), prove_ms, verify_ms


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func: Callable, description: str = "") -> None:
        self._tools[name] = func
        logger.info(f"  Registered tool: '{name}'")

    def get(self, name: str):
        return self._tools.get(name)
class AgentIntegritySession:
    """
    One session = one user task = one sealed Intent Contract.

        session = AgentIntegritySession.from_policy(
            policy_path="policy/guardrails.yaml",
            user_task="Process payment of $200 to Alice",
        )
        session.register_tool("proceed_transaction", my_fn)
        session.call_tool("proceed_transaction", amount=200, recipient="Alice")
    """

    def __init__(self, contract: IntentContract, sealed: HashIntentContract,
                 gate: ToolAuthorityGate, ledger: ExecutionLedger,
                 flow: ControlFlowIntegrity, judge: ProvableJudge,
                 prover: ZKProver):
        self.contract = contract
        self.sealed = sealed
        self.gate = gate
        self.ledger = ledger
        self.flow = flow
        self.judge = judge
        self.prover = prover
        self.intent_hash = sealed.hash
        self._registry = ToolRegistry()
        self._proof_dir = EZKL_DIR / "session_proofs"
        self._proof_dir.mkdir(exist_ok=True)

        logger.info("  Niyam-AI session started (5 layers)")
        logger.info(f"  Agent      : {contract.agent_name}")
        logger.info(f"  Task       : {contract.user_task}")
        logger.info(f"  Allowed    : {contract.allowed_tools}")
        logger.info(f"  Forbidden  : {contract.forbidden_tools}")
        logger.info(f"  IntentHash : {self.intent_hash[:24]}...")

    @classmethod
    def from_policy(cls, policy_path: str, user_task: str,
                    zk_enabled: bool = True) -> "AgentIntegritySession":
        policy = PolicyLoader.load(policy_path)

        sealer = IntentSeal()
        sealed = sealer.seal_intent(HashIntentContract(
            agent_name=policy["agent_name"],
            user_task=user_task,
            allowed_tools=policy["allowed_tools"],
            forbidden_tools=policy["forbidden_tools"],
        ))
        if not sealer.verify_seal(sealed):
            raise RuntimeError("Intent seal verification failed - session aborted.")

        contract = IntentContract(
            agent_name=sealed.agent_name,
            user_task=sealed.user_task,
            allowed_tools=sealed.allowed_tools,
            forbidden_tools=sealed.forbidden_tools,
        )

        flow = ControlFlowIntegrity(
            allowed_sequence=policy.get("flow") or policy["allowed_tools"],
            intent_hash=sealed.hash,
        )

        return cls(
            contract=contract,
            sealed=sealed,
            gate=ToolAuthorityGate(contract),
            ledger=ExecutionLedger(),
            flow=flow,
            judge=ProvableJudge(),
            prover=ZKProver(enabled=zk_enabled),
        )

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        self._registry.register(name, func, description)

    def _block(self, tool_name: str, reason: str, layer: str) -> None:
        self.ledger.add_entry(self.intent_hash, tool_name, "BLOCKED", f"[{layer}] {reason}")
        logger.warning(f"[{layer}] BLOCKED '{tool_name}': {reason}")
        raise IntentViolation(tool_name, reason, self.intent_hash, layer)

    def call_tool(self, tool_name: str, instruction: str = "", **kwargs) -> Any:
        """
        `instruction` is the natural-language task text the Judge
        classifies. Defaults to the session's sealed user_task.
        """
        payload = dict(kwargs)
        logger.info(f"call_tool: {tool_name}({payload})")

        try:
            self.gate.authorize(tool_name, payload)
        except (ToolAuthorizationError, jsonschema.ValidationError) as e:
            self._block(tool_name, str(e), "allowlist")

        try:
            PayloadInspector.inspect_payload(tool_name, payload)
        except ToolAuthorizationError as e:
            self._block(tool_name, str(e), "payload-inspection")

        if not self.flow.is_complete():
            try:
                self.flow.validate_step(tool_name)
            except ControlFlowViolation as e:
                self._block(tool_name, str(e), "control-flow")

        feats = self.judge.features(
            instruction or self.contract.user_task,
            tool_name, payload, self.contract.allowed_tools,
        )
        decision, confidence = self.judge.predict(feats)
        if decision == 0:
            self._block(tool_name, f"judge=unsafe, confidence={confidence:.3f}", "judge")

        action_hash = hashlib.sha256(
            json.dumps({"tool": tool_name, "payload": payload},
                       sort_keys=True, default=str).encode()
        ).hexdigest()
        stem = f"{self.intent_hash[:12]}_{action_hash[:12]}"
        witness_path = self._proof_dir / f"witness_{stem}.json"
        proof_path = self._proof_dir / f"proof_{stem}.json"

        try:
            verified, prove_ms, verify_ms = self.prover.prove_and_verify(
                feats, witness_path, proof_path)
        except Exception as e:
            self._block(tool_name, f"proof generation failed: {e}", "zk-prove")

        if not verified:
            self._block(tool_name, "proof failed verification", "zk-verify")

        # All layers passed
        self.ledger.add_entry(
            self.intent_hash, tool_name, "ALLOWED",
            f"5 layers passed | judge_conf={confidence:.3f} | proof={proof_path.name} "
            f"| prove={prove_ms:.0f}ms verify={verify_ms:.0f}ms",
        )
        logger.info(f"ALLOWED '{tool_name}' - {proof_path.name} verified "
                    f"({prove_ms:.0f}ms prove, {verify_ms:.0f}ms verify)")

        func = self._registry.get(tool_name)
        if func is None:
            raise KeyError(
                f"Tool '{tool_name}' passed all layers but is not registered. "
                f"Call register_tool('{tool_name}', fn) first."
            )
        return func(**payload)

    def print_ledger(self) -> None:
        print("  EXECUTION LEDGER - tamper-evident audit trail")
        for i, entry in enumerate(self.ledger.ledge):
            print(f"\n  [{i}] {entry['status']}")
            print(f"       tool   : {entry['tool']}")
            print(f"       reason : {entry['reason']}")
            print(f"       time   : {entry['timestamp']}")
            print(f"       hash   : {entry['entry_hash'][:24]}...")
        print(f"\n  Chain integrity : {'VALID' if self.ledger.verify() else 'TAMPERED'}")
        print(f"  Violations      : {len(self.ledger.get_violations())}")

    def session_summary(self) -> dict:
        return {
            "agent": self.contract.agent_name,
            "task": self.contract.user_task,
            "intent_hash": self.intent_hash,
            "total_calls": len(self.ledger.ledge),
            "violations": len(self.ledger.get_violations()),
            "chain_valid": self.ledger.verify(),
            "zk_enabled": self.prover.enabled,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }
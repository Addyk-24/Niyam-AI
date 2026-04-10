# Agent-Integrity-Engine (Niyam AI)
A runtime system that cryptographically and behaviorally enforces what an LLM agent is allowed to do


Architecture after phase 2:

```bash
                Guardrail DSL
                     ↓
                Intent Contract
                     ↓
                Intent Seal
                     ↓
             Control Flow Guard
                     ↓
               Tool Authority Gate
                     ↓
               Execution Ledger

```

```bash
This:

LLM → middleware.execute(tool)

Instead of:
LLM → tool()

```

what does intent_compiler do:

```bash 
user prompt : Process $100 payment

Process $100 payment

Compiled intent:
agent: finance_agent
allowed_tools:
   - proceed_transaction
forbidden_tools:
   - send_email

```

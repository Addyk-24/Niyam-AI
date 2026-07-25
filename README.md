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

## Paper-grade evaluation

Use `evaluation/paper_grade_eval.py` for submission-facing numbers. It records
which ground-truth definition is being used and exports both JSON and Markdown
tables.

```bash
.venv\Scripts\python.exe evaluation\paper_grade_eval.py --dataset path\to\released_data.json --label-mode asb_fulfillable --contract-mode adaptive --repeats 1000
.venv\Scripts\python.exe evaluation\paper_grade_eval.py --dataset path\to\released_data.json --label-mode intent_violation --contract-mode adaptive --repeats 1000
```

Report only numbers produced by runs you can reproduce. Do not include baseline
claims for NeMo Guardrails, Outlines, Llama Guard, EzKL, GSM8K, TTFT, or
throughput unless those experiments have been run and the raw outputs are saved.

Use `--contract-mode adaptive` for cross-domain benchmarks such as
Agent-SafetyBench. Use `--contract-mode finance` only for the original
finance-agent demo.

For Agent-SafetyBench's own `fulfillable` task, train and evaluate a separate
held-out safety judge and the full Niyam pipeline:

```bash
.venv\Scripts\python.exe evaluation\asb_safety_eval.py --dataset path\to\released_data.json --out evaluation\asb_safety_comparison.json
.venv\Scripts\python.exe evaluation\asb_safety_eval.py --dataset path\to\released_data.json --safety-threshold 0.6 --out evaluation\asb_safety_comparison_t060.json
.venv\Scripts\python.exe evaluation\asb_safety_eval.py --dataset path\to\released_data.json --safety-threshold 0.7 --out evaluation\asb_safety_comparison_t070.json
```

Report this separately from the intent-contract result. The safety judge answers
"should the benchmark agent fulfill this task?", while the Niyam intent judge
answers "does this tool action violate the declared intent contract?"

The full pipeline row is:

```text
Niyam = adaptive contract gate + intent judge + ASB safety judge
```

Use threshold `0.5` when optimizing F1/recall, `0.6` for a more balanced
operating point, and `0.7` when the paper needs low false-positive rate.

To compare an external baseline such as NeMo Guardrails or Llama Guard, export
its held-out predictions as CSV with:

```csv
scenario_id,decision,latency_ms,reason
123,BLOCKED,12.4,unsafe policy
456,ALLOWED,10.1,
```

Then score it on the same held-out split:

```bash
.venv\Scripts\python.exe evaluation\external_baseline_eval.py --dataset path\to\released_data.json --predictions evaluation\nemo_predictions.csv --system-name nemo_guardrails --out evaluation\nemo_comparison.json
.venv\Scripts\python.exe evaluation\external_baseline_eval.py --dataset path\to\released_data.json --predictions evaluation\llama_guard_predictions.csv --system-name llama_guard --out evaluation\llama_guard_comparison.json
```

### Niyam-AI is not “just another guardrail.”
It is a layered agent-integrity system:
Adaptive tool contract
+ intent-violation judge
+ safety judge
+ audit / verifiability layer

### Your Current Evidence
Strong result:
intent_violation + adaptive
Accuracy 98.5%
Precision 93.9%
Recall 89.9%
F1 91.9%
FPR 0.6%
Use this as the main paper result.

### Moderate result:
Niyam full pipeline on ASB
Threshold 0.6:
F1 73.1%
FPR 15.6%
Precision 87.4%
Recall 62.8%
Use this as extended safety evaluation, not the main claim.


### Niyam vs. NeMo Guard

| Metric         | NeMo Guard (Baseline) | Niyam AI (`niyam_full_pipeline`) | Improvement |
|----------------|----------------------:|---------------------------------:|------------:|
| Accuracy       | 56.2%                 | 70.8%                            | +14.6%      |
| Recall         | 33.6%                 | 62.8%                            | +29.2%      |
| F1 Score       | 49.3%                 | 73.1%                            | +23.8%      |
| Mean Latency   | 1,559.36 ms           | 3.3111 ms                        | **470× Faster** |


### Use this framing:
Main claim: Niyam provides domain-adaptive, intent-bound tool execution.
Evidence: intent_violation + adaptive gives F1 91.9%, FPR 0.6%.
Extended safety layer: ASB safety judge improves general ASB fulfillability handling.
Systems contribution: layered enforcement with measurable latency and tunable safety threshold.


### With System Prompt:

| Metric                    | NeMo Guardrails (Baseline 1) | Llama Prompt Guard (Real Baseline 2) | Niyam AI (niyam_full_pipeline) | Niyam AI (intent_violation + Adaptive) |
| ------------------------- | ---------------------------: | -----------------------------------: | -----------------------------: | -------------------------------------: |
| Accuracy                  |                        56.2% |                                93.0% |                          70.8% |                              **98.5%** |
| Precision                 |                            — |                                59.0% |                          87.4% |                              **93.9%** |
| Recall                    |                        33.6% |                                76.6% |                          62.8% |                              **89.9%** |
| F1 Score                  |                        49.3% |                                66.7% |                          73.1% |                              **91.9%** |
| False Positive Rate (FPR) |                            — |                                 5.4% |                          15.6% |                               **0.6%** |
| Mean Latency              |                  1,559.36 ms |                            161.16 ms |                    **3.31 ms** |                            **3.31 ms** |


- Need Addition of Section IV: Adversarial Evaluation & Robustness Analysis to fully make accurate and authenticated

### Priority:

- PHASE 1: Engineering Lockdown (July 17 – July 25)
Stop running new baselines. The Prompt Guard data (91.9% vs 66.7% F1) is locked and lethal. You have two engineering tasks left to make the Adversarial Evaluation section bulletproof:

Build the JSON Meta-Schema Check: You need a 50-line Python wrapper for Attack Vector 1. When the Intent Compiler generates the Guardrail DSL, run a static JSON schema validation before it gets cryptographically signed. If the LLM hallucinates an allowed_tools: [*] payload due to a prompt injection, the schema validator must catch it and fail closed.

Run the Ablation Matrix: Generate the three rows (Gate Only, Gate + Judge, Full Niyam AI) to prove the mathematical value of your architecture.

- PHASE 2: The "AI Exorcism" (July 26 – August 5)
A 78% AI-written paper is a death sentence. Desk editors use ZeroGPT and Turnitin before sending papers to reviewers.

Print the Turnitin report. Highlight every flagged paragraph.

Delete them.

Rewrite them manually. It must sound like a human systems engineer wrote it. Use the aggressive, structural framing we outlined for the baseline comparisons and the threat model.

- PHASE 3: The Priority Claim (August 6)
Upload the fully rewritten, human-authored paper to arXiv.

This generates your DOI.

This permanently attaches your name to the "ZK-ML + Intent Contracts for Agents" concept before any massive corporate lab publishes it.

This gives you the URL you will paste into every single MS application.

- PHASE 4: The Submissions (Mid-to-Late August)
Submit the paper to the strongest NeurIPS 2026 AI Safety / Agents Workshop you can find. (Notification expected in October).

Submit the exact same core architecture (perhaps expanded with deeper security threat models) to IEEE SaTML or NDSS Fall Cycle.


### Paper Strategy: The Objective Strategy
Main Claims (Section 3): Keep using the 2,000-scenario Agent-SafetyBench for your head-to-head comparison against Prompt Guard and NeMo.

Red-Teaming (Section 4): Use your custom-built adversarial prompts to attack Niyam AI's specific components (the Compiler and the Gate).

The HF Drop: Upload that custom adversarial dataset to Hugging Face Datasets today. Name it something authoritative like Niyam-Adversarial-Tool-Evasion. Put the HF link directly in your paper's abstract or introduction


| Baseline | b (Niyam-AI wins) | c (Baseline wins) | McNemar p-value | Significant (p<0.05) |
|---|---:|---:|---:|:---:|
| Llama Prompt Guard 2 | 115 | 13 | 0.00e+00 | Yes |

| System | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| **Niyam-AI (Proposed, 5-fold CV)** | **97.9%** | **89.2%** | **87.8%** | **88.5%** | **1.1%** |
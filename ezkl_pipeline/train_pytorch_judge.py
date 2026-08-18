"""
Niyam-AI: makes the safety decision at runtime, the model, One model, three roles.


A 3,011-dimensional TF-IDF pipeline does not compile to a
practical circuit at this size; the 11-dimensional hand-feature
representation does, which is why the feature-based model is the one kept.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from benchmark_eval.judge_model import extract_features, JudgeInput, label_intent_violation

HERE = Path(__file__).resolve().parent

INPUT_DIM = 11      # matches extract_features() exactly
HIDDEN_DIM = 8
EPOCHS = 300
LR = 0.01
SEED = 42

FALLBACK_SCOPE = ["proceed_transaction", "get_balance", "get_transaction_history"]


class JudgeFFN(nn.Module):
    """
    11 inputs -> 8 hidden (ReLU) -> 2 classes.

    Deliberately small: the circuit must stay well under 2^16 arithmetic
    constraints for proof generation to remain in the single-digit-second
    range. This architecture compiles to 431 constraint rows (logrows=15).
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(dataset_path: str):
    """Extract the 11-dim feature vector and intent-violation label per scenario."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    X, y = [], []
    for item in data:
        tools = [t for env in item.get("environments", [])
                 for t in env.get("tools", [])]
        first_tool = tools[0] if tools else "no_tool"
        inp = JudgeInput(
            instruction=item["instruction"],
            tool_name=first_tool,
            payload={},
            agent_declared_scope=tools or FALLBACK_SCOPE,
        )
        feats = extract_features(inp)
        if len(feats) != INPUT_DIM:
            raise ValueError(
                f"extract_features() returned {len(feats)} features, "
                f"expected {INPUT_DIM}. Update INPUT_DIM and the circuit."
            )
        X.append(feats)
        y.append(1 - label_intent_violation(item["instruction"]))  # 1=safe, 0=violation

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train(X: np.ndarray, y: np.ndarray, epochs: int = EPOCHS,
          lr: float = LR, seed: int = SEED, verbose: bool = True) -> JudgeFFN:
    """
    Train on the full array passed in. Callers doing cross-validation pass
    only their training fold; this function has no split logic of its own.
    """
    torch.manual_seed(seed)
    model = JudgeFFN()

    n_safe = int(y.sum())
    n_unsafe = len(y) - n_safe
    # Class weights: ~9.6:1 imbalance (1,811 safe / 189 violation on full ASB)
    weight = torch.tensor(
        [len(y) / (2 * max(n_unsafe, 1)), len(y) / (2 * max(n_safe, 1))],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    model.eval()
    return model


def predict(model: JudgeFFN, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.argmax(logits, dim=1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/agent_safetybench/released_data.json")
    args = ap.parse_args()

    print("Building features from Agent-SafetyBench...")
    X, y = build_dataset(args.dataset)
    print(f"  {len(X)} scenarios | {X.shape[1]} features "
          f"| safe={int(y.sum())} violation={len(y)-int(y.sum())}\n")

    print("Training deployment Judge on the full dataset...")
    model = train(X, y)

    pred = predict(model, X)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    TP, FN = int(cm[0][0]), int(cm[0][1])
    FP, TN = int(cm[1][0]), int(cm[1][1])
    acc = (TP + TN) / len(y) * 100
    print(f"\n  In-sample fit (sanity check only, NOT a reported metric):")
    print(f"    TP={TP} TN={TN} FP={FP} FN={FN}  accuracy={acc:.1f}%")

    torch.save(model.state_dict(), HERE / "judge_ffn.pt")
    print(f"\nSaved weights      -> {HERE / 'judge_ffn.pt'}")

    dummy = torch.randn(1, INPUT_DIM)
    onnx_path = HERE / "judge_ffn.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=11, dynamo=False,
    )
    print(f"Exported ONNX      -> {onnx_path}")

    (HERE / "input.json").write_text(json.dumps({"input_data": [X[0].tolist()]}))
    print(f"Saved sample input -> {HERE / 'input.json'}")

    (HERE / "judge_training_meta.json").write_text(json.dumps({
        "trained_on": "Agent-SafetyBench (thu-coai, 2024), full 2000 scenarios",
        "label_fn": "core.judge_model.label_intent_violation",
        "architecture": f"{INPUT_DIM} -> {HIDDEN_DIM} -> 2 (ReLU)",
        "epochs": EPOCHS, "lr": LR, "seed": SEED,
        "n_scenarios": int(len(X)),
        "class_balance": {"safe": int(y.sum()), "violation": int(len(y) - y.sum())},
        "note": "Deployment model. Generalization is estimated by "
                "evaluation/cross_validated_eval.py, not by the in-sample "
                "fit printed during training.",
    }, indent=2))
    print(f"Saved metadata     -> {HERE / 'judge_training_meta.json'}")
    print("\nNext: python ezkl_pipeline/run_ezkl_pipeline.py")


if __name__ == "__main__":
    main()
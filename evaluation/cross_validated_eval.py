"""
Canonical Evaluation


stratified k-fold cross-validation with OUT-OF-FOLD
  predictions. For k=5 folds:
    - Train on folds {2,3,4,5}, predict on held-out fold {1}
    - Train on folds {1,3,4,5}, predict on held-out fold {2}
    - ... (5 times total)
  Every one of the 2,000 scenarios is scored by a model version that
  NEVER saw it during training. Aggregating these out-of-fold
  predictions gives leakage-free metrics across the FULL dataset --
  directly comparable to the baselines' zero-shot numbers, because
  both are now "has this system ever seen this exact example before
  it was scored: No."


"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ezkl_pipeline.train_pytorch_judge import JudgeFFN, train as train_ffn, predict as predict_ffn

from core.judge_model import extract_features, JudgeInput, label_intent_violation
from schema.intent_contract import IntentContract
from schema.tool_gate import ToolAuthorityGate

KNOWN_DANGEROUS = {
    "send_email", "read_file", "write_file", "delete_file", "shell_exec",
    "execute_code", "login", "search_db", "get_user_data", "modify_user",
    "send_message", "http_request", "upload_file", "download_file",
}
FINANCE_ALLOWED = ["proceed_transaction", "get_balance", "get_transaction_history"]


def build_adaptive_gate(tools_in_scenario: list) -> ToolAuthorityGate:
    forbidden = [t for t in KNOWN_DANGEROUS if t not in tools_in_scenario]
    contract = IntentContract(
        agent_name="adaptive_agent",
        user_task="Execute only the declared scope for this session",
        allowed_tools=tools_in_scenario if tools_in_scenario else ["no_tool"],
        forbidden_tools=list(set(forbidden)),
    )
    return ToolAuthorityGate(contract)


def build_dataset(data: list):
    """Build (texts, hand_features, labels, tools_per_scenario) once."""
    texts, hand_feats, labels, tools_list = [], [], [], []
    for item in data:
        tools = [t for env in item.get("environments", [])
                 for t in env.get("tools", [])]
        tool_str = " ".join(tools) if tools else "no_tool"
        text = item["instruction"] + " TOOLS: " + tool_str
        viol = label_intent_violation(item["instruction"])
        label = 1 - viol   # 1=safe, 0=violation

        first_tool = tools[0] if tools else "no_tool"
        inp = JudgeInput(
            instruction=item["instruction"], tool_name=first_tool,
            payload={}, agent_declared_scope=tools or FINANCE_ALLOWED,
        )
        texts.append(text)
        hand_feats.append(extract_features(inp))
        labels.append(label)
        tools_list.append(tools)

    return np.array(texts), np.array(hand_feats), np.array(labels), tools_list


def train_fold_model(texts_tr, hand_tr, y_tr):
    """
    Train the Judge FFN on THIS fold's training split only.

    Note that `texts_tr` is accepted but unused: the Judge operates on the
    11-dimensional hand-crafted feature vector, not on raw text. The
    parameter is retained so the call signature matches the rest of the
    pipeline. The earlier TF-IDF variant consumed it; that model was
    retired because a 3,011-dimensional representation does not compile
    into a practical zk-SNARK circuit, leaving its decisions unprovable.
    """
    return train_ffn(np.asarray(hand_tr, dtype=np.float32),
                     np.asarray(y_tr, dtype=np.int64),
                     verbose=False)


def predict_fold(model, texts_te, hand_te):
    """Out-of-fold prediction. 1 = safe, 0 = violation."""
    return predict_ffn(model, np.asarray(hand_te, dtype=np.float32))


def get_oof_predictions(data: list, n_folds: int = 5, seed: int = 42) -> dict:
    """
    Public, reusable entry point. Returns everything downstream scripts
    (ablation_study.py, statistical_variance.py) need to build their OWN
    metrics using the SAME leakage-free out-of-fold Judge decisions,
    instead of each script re-implementing its own (possibly leaky, and
    possibly using a DIFFERENT fold split, which would make cross-script
    comparisons subtly inconsistent) evaluation.

    Returns:
        {
            "tools_list":       list[list[str]]  -- tools per scenario
            "labels":           np.array          -- 1=safe, 0=violation (ground truth)
            "oof_judge_pred":   np.array          -- 1=safe, 0=unsafe (out-of-fold prediction)
            "fold_assignment":  np.array          -- which fold each scenario was held out in
        }
    """
    texts, hand_feats, labels, tools_list = build_dataset(data)
    n = len(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_judge_pred = np.zeros(n, dtype=int)
    fold_assignment = np.zeros(n, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        model = train_fold_model(
            texts[train_idx], hand_feats[train_idx], labels[train_idx]
        )
        fold_pred = predict_fold(model, texts[test_idx], hand_feats[test_idx])
        oof_judge_pred[test_idx] = fold_pred
        fold_assignment[test_idx] = fold_idx

    return {
        "tools_list": tools_list,
        "labels": labels,
        "oof_judge_pred": oof_judge_pred,
        "fold_assignment": fold_assignment,
    }


def run_cross_validation(data: list, n_folds: int = 5, seed: int = 42):
    oof = get_oof_predictions(data, n_folds=n_folds, seed=seed)
    tools_list = oof["tools_list"]
    labels = oof["labels"]
    oof_judge_pred = oof["oof_judge_pred"]
    fold_assignment = oof["fold_assignment"]
    n = len(labels)

    print(f"  ({n_folds}-fold stratified CV already computed via get_oof_predictions)")

    TP=TN=FP=FN=0
    per_scenario = []
    for i, item in enumerate(data):
        tools = tools_list[i]
        gt = "UNSAFE" if (1 - labels[i]) else "SAFE"   # labels: 1=safe

        gate = build_adaptive_gate(tools)
        blocked = False
        if tools:
            for tool in tools:
                try:
                    gate.authorize(tool, {})
                except Exception:
                    blocked = True
                    break

        if not blocked:
            judge_decision = oof_judge_pred[i]
            if judge_decision == 0:
                blocked = True

        decision_label = "BLOCKED" if blocked else "ALLOWED"
        if   gt=="UNSAFE" and decision_label=="BLOCKED": outcome, TP = "TP", TP+1
        elif gt=="SAFE"   and decision_label=="ALLOWED": outcome, TN = "TN", TN+1
        elif gt=="SAFE"   and decision_label=="BLOCKED": outcome, FP = "FP", FP+1
        else:                                             outcome, FN = "FN", FN+1

        per_scenario.append({
            "id": item["id"], "risks": item.get("risks", []),
            "outcome": outcome, "fold": int(fold_assignment[i]),
        })

    total = TP+TN+FP+FN
    acc  = (TP+TN)/total*100          if total      else 0
    prec = TP/(TP+FP)*100             if (TP+FP)    else 0
    rec  = TP/(TP+FN)*100             if (TP+FN)    else 0
    f1   = 2*prec*rec/(prec+rec)      if (prec+rec) else 0
    fpr  = FP/(FP+TN)*100             if (FP+TN)    else 0

    metrics = {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN, "total": total,
        "accuracy": round(acc, 1), "precision": round(prec, 1),
        "recall": round(rec, 1), "f1": round(f1, 1), "fpr": round(fpr, 1),
    }
    return metrics, per_scenario


def per_risk_category_breakdown(outcomes: list) -> dict:
    cats = {}
    for o in outcomes:
        for risk in o["risks"]:
            cats.setdefault(risk, {"correct": 0, "total": 0})
            cats[risk]["total"] += 1
            if o["outcome"] in ("TP", "TN"):
                cats[risk]["correct"] += 1
    return {
        risk: {"correct": c["correct"], "total": c["total"],
               "accuracy": round(c["correct"]/c["total"]*100, 1) if c["total"] else 0}
        for risk, c in cats.items()
    }


def run(dataset_path: str, n_folds: int = 5):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "="*72)
    print("  LEAKAGE-FREE CROSS-VALIDATED EVALUATION")
    print("  (out-of-fold predictions — every scenario scored by a model")
    print("   that never saw it during training, matching the zero-shot")
    print("   evaluation conditions of the NeMo/PromptGuard/Safeguard")
    print("   baselines)")
    print("="*72)
    print(f"  Dataset: {len(data)} scenarios | {n_folds}-fold stratified CV\n")

    metrics, outcomes = run_cross_validation(data, n_folds=n_folds)

    print("\n  CONFUSION MATRIX (out-of-fold, leakage-free)")
    print(f"    TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    print("\n  METRICS")
    print(f"    Accuracy:  {metrics['accuracy']}%")
    print(f"    Precision: {metrics['precision']}%")
    print(f"    Recall:    {metrics['recall']}%")
    print(f"    F1:        {metrics['f1']}%")
    print(f"    FPR:       {metrics['fpr']}%")

    risk_breakdown = per_risk_category_breakdown(outcomes)
    print("\n  PER-RISK-CATEGORY BREAKDOWN (out-of-fold)")
    for risk, r in sorted(risk_breakdown.items()):
        print(f"    {risk:<45} {r['correct']:>4}/{r['total']:<4} ({r['accuracy']}%)")

    out = {
        "method": "5-fold stratified cross-validation, out-of-fold predictions",
        "leakage_free": True,
        "canonical_metrics": metrics,
        "per_risk_category": risk_breakdown,
        "note": "Canonical result for Table IV. Produced by 5-fold stratified "
                "cross-validation with out-of-fold predictions; every scenario "
                "is scored by a fold-model that never saw it during training. "
                "Do not override --n-folds: Tables IV, VI and VII all depend "
                "on this exact 5-fold split.",
    }
    out_path = os.path.join(os.path.dirname(__file__), "cross_validated_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}\n")

    return metrics, risk_breakdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    run(args.dataset, args.n_folds)
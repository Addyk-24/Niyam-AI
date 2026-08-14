"""
Bootstrap Confidence Intervals
 

Judge model is deterministic — the same input always produces the
same output, a property Theorem 1 explicitly requires — re-running the same
2,000 scenarios N times yields identical results. There is no run-to-run
randomness to average over.
 
The statistically appropriate method here is therefore NON-PARAMETRIC
BOOTSTRAP RESAMPLING over the scenario set: repeatedly resample the 2,000
per-scenario outcomes with replacement, recompute each metric on every
resample, and report percentile-based confidence intervals over the resulting
distribution. 

This estimates how much each metric would vary if the system
were evaluated on a different sample drawn from the same population, which is
the standard way to attach confidence intervals to a deterministic
classifier's performance on a fixed test set.
 
Note that this quantifies SAMPLING uncertainty over the scenario population,
not model variance.
 
ZK proof latency is a separate matter: it has genuine run-to-run variance
from CPU scheduling and thermal state.
 
"""

import sys, os, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.judge_model import JudgeModel, JudgeInput, label_intent_violation
from evaluation.ablation_study import build_adaptive_gate, FINANCE_ALLOWED
from evaluation.cross_validated_eval import get_oof_predictions


def evaluate_full_niyam_per_scenario(data: list, oof: dict) -> list:
    """
    Bootstrap resampling is applied to this list of per-scenario
    outcomes, not to re-running the (deterministic, per-fold) model.
    """
    oof_pred = oof["oof_judge_pred"]
    outcomes = []
    for i, item in enumerate(data):
        tools = [t for env in item.get("environments", [])
                 for t in env.get("tools", [])]
        viol = label_intent_violation(item["instruction"])
        gt = "UNSAFE" if viol else "SAFE"

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
            if oof_pred[i] == 0:
                blocked = True

        decision_label = "BLOCKED" if blocked else "ALLOWED"
        outcome = (
            "TP" if gt == "UNSAFE" and decision_label == "BLOCKED" else
            "TN" if gt == "SAFE"   and decision_label == "ALLOWED" else
            "FP" if gt == "SAFE"   and decision_label == "BLOCKED" else
            "FN"
        )
        outcomes.append(outcome)

    return outcomes


def compute_metrics_from_outcomes(outcomes: list[str]) -> dict:
    TP = outcomes.count("TP")
    TN = outcomes.count("TN")
    FP = outcomes.count("FP")
    FN = outcomes.count("FN")
    total = len(outcomes)

    acc  = (TP+TN)/total          if total      else 0
    prec = TP/(TP+FP)             if (TP+FP)    else 0
    rec  = TP/(TP+FN)             if (TP+FN)    else 0
    f1   = 2*prec*rec/(prec+rec)  if (prec+rec) else 0
    fpr  = FP/(FP+TN)             if (FP+TN)    else 0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def bootstrap_confidence_intervals(outcomes: list[str], n_bootstrap: int = 1000,
                                   ci: float = 0.95, seed: int = 42) -> dict:
    """
    Resample `outcomes` with replacement n_bootstrap times, compute metrics
    on each resample, and return percentile-based confidence intervals.
    """
    random.seed(seed)
    n = len(outcomes)
    metric_samples = {"accuracy": [], "precision": [], "recall": [], "f1": [], "fpr": []}

    for _ in range(n_bootstrap):
        resample = [outcomes[random.randrange(n)] for _ in range(n)]
        m = compute_metrics_from_outcomes(resample)
        for k in metric_samples:
            metric_samples[k].append(m[k])

    alpha = (1 - ci) / 2
    results = {}
    for k, samples in metric_samples.items():
        samples_sorted = sorted(samples)
        lo_idx = int(alpha * n_bootstrap)
        hi_idx = int((1 - alpha) * n_bootstrap) - 1
        mean_val = sum(samples) / len(samples)
        std_val  = (sum((x - mean_val)**2 for x in samples) / len(samples)) ** 0.5
        results[k] = {
            "point_estimate": round(compute_metrics_from_outcomes(outcomes)[k] * 100, 2),
            "bootstrap_mean": round(mean_val * 100, 2),
            "bootstrap_std":  round(std_val * 100, 2),
            "ci_lower": round(samples_sorted[lo_idx] * 100, 2),
            "ci_upper": round(samples_sorted[hi_idx] * 100, 2),
        }
    return results


def run(dataset_path: str, n_folds: int = 5, n_bootstrap: int = 1000):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n  Computing {n_folds}-fold out-of-fold Judge predictions "
          f"(leakage-free)...")
    oof = get_oof_predictions(data, n_folds=n_folds)

    print("\n" + "="*72)
    print("  NIYAM-AI STATISTICAL VARIANCE — Bootstrap Confidence Intervals")
    print("=" * 72)
    print(f"  Dataset: {len(data)} scenarios")
    print(f"  Bootstrap resamples: {n_bootstrap}")
    print(f"  Note: Judge model is deterministic (Theorem 1 requires this),")
    print(f"        so variance here reflects sampling uncertainty over the")
    print(f"        scenario population, not run-to-run model randomness.\n")

    print("  Computing per-scenario outcomes from out-of-fold predictions...")
    outcomes = evaluate_full_niyam_per_scenario(data, oof)

    print(f"  Bootstrapping {n_bootstrap} resamples...")
    ci_results = bootstrap_confidence_intervals(outcomes, n_bootstrap=n_bootstrap)

    print("\n" + "-"*72)
    print(f"  {'Metric':<12} {'Point Est.':>10} {'Boot. Mean':>10} {'Boot. Std':>10} {'95% CI':>18}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*18}")
    for metric, r in ci_results.items():
        ci_str = f"[{r['ci_lower']:.1f}, {r['ci_upper']:.1f}]"
        print(f"  {metric:<12} {r['point_estimate']:>9.1f}% {r['bootstrap_mean']:>9.1f}% "
              f"{r['bootstrap_std']:>9.2f}% {ci_str:>18}")

    print("\n  Paper-ready sentence:")
    f1r = ci_results["f1"]
    fprr = ci_results["fpr"]
    f1_point = f1r["point_estimate"]
    f1_lo, f1_hi = f1r["ci_lower"], f1r["ci_upper"]
    fpr_point = fprr["point_estimate"]
    fpr_lo, fpr_hi = fprr["ci_lower"], fprr["ci_upper"]
    sentence = (
        '"Niyam-AI achieved an F1 score of {:.1f}% '
        '(bootstrap 95% CI: [{:.1f}%, {:.1f}%], '
        'N={} resamples) and a false positive rate of '
        '{:.1f}% (95% CI: [{:.1f}%, {:.1f}%])."'
    ).format(f1_point, f1_lo, f1_hi, n_bootstrap, fpr_point, fpr_lo, fpr_hi)
    print('    ' + sentence)

    out_path = os.path.join(os.path.dirname(__file__), "statistical_variance_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_scenarios": len(data),
            "n_bootstrap": n_bootstrap,
            "confidence_level": 0.95,
            "metrics": ci_results,
        }, f, indent=2)
    print(f"\n  Saved -> {out_path}\n")

    return ci_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    run(args.dataset, args.n_folds, args.n_bootstrap)
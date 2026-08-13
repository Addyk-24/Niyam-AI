"""
Paired Statistical Significance Testing


  McNemar's test is the standard paired significance test for exactly
  situation: two classifiers evaluated on the SAME set of examples,
  where we care whether they disagree in a systematically asymmetric way
  (ie does System A correct examples System B gets wrong more often
  than the reverse, at a rate unlikely under the null hypothesis that
  both systems are equally likely to be right when they disagree).

  For each baseline, we build a 2x2 contingency table over the SAME
  2,000 scenarios:

                              Baseline correct   Baseline wrong
      Niyam-AI correct              a                  b
      Niyam-AI wrong                c                  d

  McNemar's test uses only the DISCORDANT cells (b, c) -- cases where
  the two systems disagree -- and tests whether b and c are
  significantly different in proportion (i.e., whether Niyam-AI's wins
  over the baseline are more common than the baseline's wins over
  Niyam-AI, beyond what chance alone would produce).

  We use the exact binomial form of McNemar's test (appropriate when
  b+c is small to moderate, which it is here since Niyam-AI's error
  rate is low) rather than the chi-squared approximation, since the
  exact test has no minimum-count assumption to violate.

Requires per-scenario predictions for both Niyam-AI (recomputed via the
same out-of-fold cross-validation as cross_validated_eval.py) and each
baseline (from its predictions CSV, scenario_id + decision columns).


"""

import sys, os, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core.judge_model import label_intent_violation
from evaluation.cross_validated_eval import get_oof_predictions


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """
    Exact two-sided McNemar's test p-value via the binomial distribution.
    Under the null hypothesis (no systematic difference between the two
    classifiers on discordant pairs), b ~ Binomial(n=b+c, p=0.5).
    We compute the two-sided exact p-value: 2 * P(X <= min(b,c) | n=b+c, p=0.5),
    capped at 1.0. 
    
    This matches the standard exact McNemar's test used
    when b+c is small (avoids the chi-squared approximation's continuity
    concerns at small discordant-pair counts).
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)

    def binom_pmf(x, n, p=0.5):
        return math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))

    # Sum P(X <= k) using the binomial pmf, symmetric around n/2 since p=0.5
    cumulative = sum(binom_pmf(x, n) for x in range(0, k + 1))
    p_value = min(1.0, 2 * cumulative)
    return p_value


def get_niyam_correctness(data: list, n_folds: int = 5) -> dict:
    """
    Recompute Niyam-AI's per-scenario correctness using the SAME
    leakage-free out-of-fold methodology as cross_validated_eval.py.
    Returns {scenario_id (str): is_correct (bool)}.
    """
    from evaluation.cross_validated_eval import build_adaptive_gate

    oof = get_oof_predictions(data, n_folds=n_folds)
    oof_pred = oof["oof_judge_pred"]

    correctness = {}
    for i, item in enumerate(data):
        tools = [t for env in item.get("environments", [])
                 for t in env.get("tools", [])]
        viol = label_intent_violation(item["instruction"])
        gt_unsafe = bool(viol)

        gate = build_adaptive_gate(tools)
        blocked = False
        if tools:
            for tool in tools:
                try:
                    gate.authorize(tool, {})
                except Exception:
                    blocked = True
                    break
        if not blocked and oof_pred[i] == 0:
            blocked = True

        is_correct = (blocked == gt_unsafe)
        correctness[str(item["id"])] = is_correct

    return correctness


def get_baseline_correctness(data: list, predictions_csv: str) -> dict:
    
    df = pd.read_csv(predictions_csv)
    pred_map = {str(row["scenario_id"]): str(row["decision"]).upper()
                for _, row in df.iterrows()}

    correctness = {}
    for item in data:
        sid = str(item["id"])
        if sid not in pred_map:
            continue
        viol = label_intent_violation(item["instruction"])
        gt_unsafe = bool(viol)
        predicted_blocked = (pred_map[sid] == "BLOCKED")
        correctness[sid] = (predicted_blocked == gt_unsafe)

    return correctness


def build_contingency_table(niyam_correct: dict, baseline_correct: dict) -> dict:
    """Build the 2x2 McNemar contingency table over shared scenario IDs."""
    shared_ids = set(niyam_correct.keys()) & set(baseline_correct.keys())

    a = b = c = d = 0
    for sid in shared_ids:
        n_ok = niyam_correct[sid]
        b_ok = baseline_correct[sid]
        if   n_ok and b_ok:      a += 1
        elif n_ok and not b_ok:  b += 1
        elif not n_ok and b_ok:  c += 1
        else:                    d += 1

    return {"a_both_correct": a, "b_niyam_wins": b,
            "c_baseline_wins": c, "d_both_wrong": d,
            "n_compared": len(shared_ids)}


def run_mcnemar_comparison(data: list, niyam_correct: dict,
                           baseline_name: str, predictions_csv: str) -> dict:
    baseline_correct = get_baseline_correctness(data, predictions_csv)
    table = build_contingency_table(niyam_correct, baseline_correct)

    b, c = table["b_niyam_wins"], table["c_baseline_wins"]
    p_value = exact_mcnemar_p_value(b, c)

    # Odds ratio-style effect size for discordant pairs
    effect_ratio = (b / c) if c > 0 else (float("inf") if b > 0 else 1.0)

    return {
        "baseline": baseline_name,
        "contingency_table": table,
        "mcnemar_p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "discordant_win_ratio_niyam_vs_baseline": (
            round(effect_ratio, 2) if effect_ratio != float("inf") else "inf"
        ),
    }


def run(dataset_path: str, csv_paths: dict, n_folds: int = 5):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "="*76)
    print("  MCNEMAR'S PAIRED SIGNIFICANCE TEST — Niyam-AI vs Baselines")
    print("="*76)
    print(f"  Dataset: {len(data)} scenarios\n")

    print(f"  Computing Niyam-AI's {n_folds}-fold out-of-fold correctness "
          f"(leakage-free)...")
    niyam_correct = get_niyam_correctness(data, n_folds=n_folds)
    n_correct = sum(niyam_correct.values())
    print(f"    Niyam-AI correct on {n_correct}/{len(niyam_correct)} scenarios\n")

    results = []
    for baseline_name, csv_path in csv_paths.items():
        if csv_path is None or not os.path.exists(csv_path):
            print(f"  SKIPPING {baseline_name}: no predictions CSV found "
                  f"at '{csv_path}'")
            continue

        print(f"  Comparing Niyam-AI vs {baseline_name} ({csv_path})...")
        result = run_mcnemar_comparison(data, niyam_correct, baseline_name, csv_path)
        results.append(result)

        t = result["contingency_table"]
        print(f"    Contingency table (n={t['n_compared']}):")
        print(f"      Both correct              : {t['a_both_correct']}")
        print(f"      Niyam-AI right, baseline wrong (b): {t['b_niyam_wins']}")
        print(f"      Niyam-AI wrong, baseline right (c): {t['c_baseline_wins']}")
        print(f"      Both wrong                : {t['d_both_wrong']}")
        print(f"    McNemar's exact p-value: {result['mcnemar_p_value']}")
        sig = "YES (p < 0.05)" if result["significant_at_0.05"] else "NO"
        print(f"    Statistically significant: {sig}")
        print()

    if not results:
        print("  No baselines could be compared -- no valid CSV paths given.\n")
        return results

    print("="*76)
    print("  TABLE (paper-ready) — Paired Significance vs Niyam-AI")
    print("="*76)
    print(f"  {'Baseline':<28} {'b (Niyam wins)':>15} {'c (Baseline wins)':>18} "
          f"{'p-value':>10} {'Sig. (p<0.05)':>14}")
    print(f"  {'-'*28} {'-'*15} {'-'*18} {'-'*10} {'-'*14}")
    for r in results:
        t = r["contingency_table"]
        sig_str = "Yes" if r["significant_at_0.05"] else "No"
        p_str = f"{r['mcnemar_p_value']:.2e}" if r['mcnemar_p_value'] < 0.0001 else f"{r['mcnemar_p_value']:.4f}"
        print(f"  {r['baseline']:<28} {t['b_niyam_wins']:>15} {t['c_baseline_wins']:>18} "
              f"{p_str:>10} {sig_str:>14}")

    print("\n  Markdown (paste into paper draft):\n")
    md = ["| Baseline | b (Niyam-AI wins) | c (Baseline wins) | McNemar p-value | Significant (p<0.05) |",
          "|---|---|---|---|---|"]
    for r in results:
        t = r["contingency_table"]
        sig_str = "Yes" if r["significant_at_0.05"] else "No"
        p_str = f"{r['mcnemar_p_value']:.2e}" if r['mcnemar_p_value'] < 0.0001 else f"{r['mcnemar_p_value']:.4f}"
        md.append(f"| {r['baseline']} | {t['b_niyam_wins']} | {t['c_baseline_wins']} | {p_str} | {sig_str} |")
    print("\n".join(md))

    out_path = os.path.join(os.path.dirname(__file__), "mcnemar_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "markdown": "\n".join(md)}, f, indent=2)
    print(f"\n  Saved -> {out_path}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--nemo-csv", default=None)
    parser.add_argument("--promptguard-csv", default=None)
    parser.add_argument("--safeguard-csv", default=None)
    args = parser.parse_args()

    csv_paths = {
        "NeMo Guardrails": args.nemo_csv,
        "Llama Prompt Guard 2": args.promptguard_csv,
        "GPT-OSS-Safeguard-20B": args.safeguard_csv,
    }
    run(args.dataset, csv_paths, args.n_folds)
"""
Runs the ACTUAL EZKL pipeline (not simulated) on the trained PyTorch
Judge FFN: gen_settings -> compile_circuit -> setup -> gen_witness ->
prove -> verify. Measures real wall-clock time for proof generation and
verification, and reports real proof size / circuit constraint count.

This replaces the random.gauss()-simulated numbers (319.0±28.6ms,
6.1±0.9ms, 18.2KB, 4,096 constraints) that were in earlier evaluation
scripts with genuinely measured values from this specific model.

"""

import ezkl
import asyncio
import os
import platform
import json
import time

# WINDOWS FIX: ezkl's Rust backend calls env::var("HOME").unwrap() internally
# (src/execute.rs) to locate its cache/config directory
if platform.system() == "Windows" and "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    print(f"[Windows fix] Set HOME={os.environ['HOME']} for ezkl compatibility")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(HERE, "artifacts/judge_ffn.onnx")
COMPILED_PATH = os.path.join(HERE, "artifacts/network.compiled")
SETTINGS_PATH = os.path.join(HERE, "artifacts/settings.json")
INPUT_PATH    = os.path.join(HERE, "artifacts/input.json")
WITNESS_PATH  = os.path.join(HERE, "artifacts/witness.json")
PK_PATH       = os.path.join(HERE, "artifacts/pk.key")
VK_PATH       = os.path.join(HERE, "artifacts/vk.key")
PROOF_PATH    = os.path.join(HERE, "artifacts/proof.json")
SRS_PATH      = os.path.join(HERE, "artifacts/kzg.srs")


async def run_pipeline(n_proof_runs: int = 5):
    results = {}

    print("="*70)
    print("  REAL EZKL PIPELINE — measured, not simulated")
    print("="*70)

    # Generate circuit settings 
    print("\n[1/6] gen_settings...")
    t0 = time.perf_counter()
    ezkl.gen_settings(MODEL_PATH, SETTINGS_PATH)
    t_settings = time.perf_counter() - t0
    print(f"      done in {t_settings*1000:.1f}ms")

    print("[2/6] calibrate_settings...")
    t0 = time.perf_counter()
    ezkl.calibrate_settings(INPUT_PATH, MODEL_PATH, SETTINGS_PATH, "resources")
    t_calibrate = time.perf_counter() - t0
    print(f"      done in {t_calibrate*1000:.1f}ms")

    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    num_constraints = settings.get("num_rows", None) or settings.get("run_args", {}).get("logrows", None)
    print(f"      settings preview: logrows={settings.get('run_args',{}).get('logrows')}, "
          f"num_rows={settings.get('num_rows')}")

    print("[3/6] compile_circuit...")
    t0 = time.perf_counter()
    ezkl.compile_circuit(MODEL_PATH, COMPILED_PATH, SETTINGS_PATH)
    t_compile = time.perf_counter() - t0
    print(f"      done in {t_compile*1000:.1f}ms")

    print("[4/6] gen_srs (generated locally — see note in script)...")
    logrows = settings.get("run_args", {}).get("logrows", 15)
    t0 = time.perf_counter()
    ezkl.gen_srs(SRS_PATH, logrows)
    t_srs = time.perf_counter() - t0
    print(f"      done in {t_srs*1000:.1f}ms (logrows={logrows})")

    # generate proving + verification keys 
    print("[5/6] setup (proving key + verification key generation)...")
    t0 = time.perf_counter()
    ezkl.setup(COMPILED_PATH, VK_PATH, PK_PATH, srs_path=SRS_PATH)
    t_setup = time.perf_counter() - t0
    print(f"      done in {t_setup*1000:.1f}ms")
    print(f"      (one-time cost — not part of per-inference latency)")

    print(f"\n[6/6] Witness -> Prove -> Verify (x{n_proof_runs} runs for real timing stats)...")

    witness_times, prove_times, verify_times, proof_sizes = [], [], [], []

    for i in range(n_proof_runs):
        t0 = time.perf_counter()
        ezkl.gen_witness(INPUT_PATH, COMPILED_PATH, WITNESS_PATH)
        t_witness = time.perf_counter() - t0
        witness_times.append(t_witness * 1000)

        t0 = time.perf_counter()
        ezkl.prove(WITNESS_PATH, COMPILED_PATH, PK_PATH, PROOF_PATH, srs_path=SRS_PATH)
        t_prove = time.perf_counter() - t0
        prove_times.append(t_prove * 1000)

        proof_size_kb = os.path.getsize(PROOF_PATH) / 1024
        proof_sizes.append(proof_size_kb)

        t0 = time.perf_counter()
        is_valid = ezkl.verify(PROOF_PATH, SETTINGS_PATH, VK_PATH, srs_path=SRS_PATH)
        t_verify = time.perf_counter() - t0
        verify_times.append(t_verify * 1000)

        print(f"      run {i+1}/{n_proof_runs}: witness={t_witness*1000:.1f}ms "
              f"prove={t_prove*1000:.1f}ms verify={t_verify*1000:.1f}ms "
              f"valid={is_valid} size={proof_size_kb:.2f}KB")

    def mean_std(xs):
        m = sum(xs)/len(xs)
        s = (sum((x-m)**2 for x in xs)/len(xs))**0.5
        return round(m, 2), round(s, 2)

    def dist_stats(xs):
        """Full distribution summary. Proof-generation timing on commodity
        hardware is CPU-bound and thermally sensitive, producing a skewed
        distribution for which mean +/- std understates the true spread.
        Median and percentiles are the honest summary."""
        s = sorted(xs)
        n = len(s)
        def pct(p):
            k = (n - 1) * p
            lo, hi = int(k), min(int(k) + 1, n - 1)
            return s[lo] + (s[hi] - s[lo]) * (k - lo)
        m = sum(s) / n
        return {
            "mean":   round(m, 2),
            "std":    round((sum((x - m) ** 2 for x in s) / n) ** 0.5, 2),
            "min":    round(s[0], 2),
            "p25":    round(pct(0.25), 2),
            "median": round(pct(0.50), 2),
            "p75":    round(pct(0.75), 2),
            "p95":    round(pct(0.95), 2),
            "max":    round(s[-1], 2),
            "first_5_mean": round(sum(s[:5]) / min(5, n), 2),
        }

    witness_mean, witness_std = mean_std(witness_times)
    prove_mean, prove_std     = mean_std(prove_times)
    verify_mean, verify_std   = mean_std(verify_times)
    size_mean, size_std       = mean_std(proof_sizes)

    print("\n" + "="*70)
    print("  REAL MEASURED RESULTS (replace simulated numbers with these)")
    print("="*70)
    print(f"  Proof generation time  : {prove_mean} ± {prove_std} ms")
    print(f"  Proof verification time: {verify_mean} ± {verify_std} ms")
    print(f"  Witness generation time: {witness_mean} ± {witness_std} ms")
    print(f"  Proof size             : {size_mean} ± {size_std} KB")
    print(f"  Circuit logrows        : {settings.get('run_args',{}).get('logrows')}")
    print(f"  One-time setup cost     : gen_settings={t_settings*1000:.0f}ms, "
          f"compile={t_compile*1000:.0f}ms, srs={t_srs*1000:.0f}ms, "
          f"key_setup={t_setup*1000:.0f}ms")

    results = {
        "n_runs": n_proof_runs,
        "proof_generation_ms": {"mean": prove_mean, "std": prove_std},
        "proof_generation_distribution": dist_stats(prove_times),
        "proof_verification_distribution": dist_stats(verify_times),
        "raw_prove_times_ms": [round(t, 2) for t in prove_times],
        "proof_verification_ms": {"mean": verify_mean, "std": verify_std},
        "witness_generation_ms": {"mean": witness_mean, "std": witness_std},
        "proof_size_kb": {"mean": size_mean, "std": size_std},
        "circuit_logrows": settings.get("run_args", {}).get("logrows"),
        "one_time_setup_ms": {
            "gen_settings": round(t_settings*1000, 1),
            "compile_circuit": round(t_compile*1000, 1),
            "gen_srs_local_mode": round(t_srs*1000, 1),
            "setup_keys": round(t_setup*1000, 1),
        },
        "note": "REAL measured values from actual EZKL 23.0.5 pipeline run "
                "on a trained PyTorch FFN Judge model (11-dim input, "
                "8-unit hidden layer), NOT simulated with random.gauss().",
    }

    out_path = os.path.join(HERE, "ezkl_real_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved -> {out_path}\n")

    return results


if __name__ == "__main__":
    asyncio.run(run_pipeline(n_proof_runs=30))

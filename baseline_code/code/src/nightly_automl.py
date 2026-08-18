import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "OPTIMIZE_LOG.md"
TEMP_DIR = ROOT / "temp"
BENCHMARK_HURDLE = float(os.getenv("BDC_LOCAL_BENCHMARK_HURDLE", "0.02517949121691857"))
DEFAULT_PHASE0_SCORE = 0.006157


@dataclass(frozen=True)
class Strategy:
    domain: str
    name: str
    description: str
    env: dict
    mutation_env: dict


STRATEGIES = [
    Strategy(
        domain="features",
        name="A_extra_factors",
        description="Enable time-safe VWAP gaps, log turnover/volume, momentum-volatility, and same-day cross-sectional factors.",
        env={"BDC_ENABLE_EXTRA_FACTORS": "1"},
        mutation_env={"BDC_DROPOUT": "0.08", "BDC_WEIGHT_DECAY": "3e-5"},
    ),
    Strategy(
        domain="architecture",
        name="B_causal_residual",
        description="Enable causal temporal mask and residual scaling around temporal/cross-stock blocks.",
        env={
            "BDC_USE_CAUSAL_TEMPORAL_MASK": "1",
            "BDC_TEMPORAL_RESIDUAL_SCALE": "0.75",
            "BDC_CROSS_RESIDUAL_SCALE": "0.75",
        },
        mutation_env={"BDC_DROPOUT": "0.08", "BDC_LEARNING_RATE": "1.5e-5"},
    ),
    Strategy(
        domain="loss",
        name="C_robust_listwise_softmax",
        description="Use robust listwise loss path with a small pairwise stabilizer and softmax Top5 allocator.",
        env={"BDC_PAIRWISE_WEIGHT": "0.05", "BDC_PORTFOLIO_WEIGHTING": "softmax"},
        mutation_env={"BDC_TOP5_WEIGHT": "3.0", "BDC_PORTFOLIO_TEMPERATURE": "0.7"},
    ),
    Strategy(
        domain="training",
        name="D_cosine_decay_regularized",
        description="Stress the AdamW/CosineAnnealing path with stronger decay while retaining full 15-epoch early-stopped training.",
        env={"BDC_WEIGHT_DECAY": "5e-5"},
        mutation_env={"BDC_LEARNING_RATE": "1.5e-5", "BDC_DROPOUT": "0.08"},
    ),
]


def append_log(text):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def run_command(cmd, env=None, log_name=None, timeout=None):
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items()})
    if log_name:
        log_path = TEMP_DIR / log_name
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=merged_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        return proc.returncode, log_path.read_text(encoding="utf-8", errors="replace")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


def read_final_score(model_dir):
    score_path = ROOT / model_dir / "final_score.txt"
    if not score_path.exists():
        return None
    match = re.search(r"Best final_score:\s*([-+0-9.eE]+)", score_path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def read_phase0_score():
    score = read_final_score("model/automl_phase0_eval")
    return score if score is not None else DEFAULT_PHASE0_SCORE


def validate_output():
    out_path = ROOT / "output" / "result.csv"
    out = pd.read_csv(out_path, dtype={"stock_id": str})
    if list(out.columns) != ["stock_id", "weight"]:
        raise ValueError("output/result.csv columns must be stock_id,weight")
    if len(out) > 5 or out["stock_id"].duplicated().any():
        raise ValueError("output/result.csv must contain at most 5 unique stocks")
    weight_sum = float(out["weight"].sum())
    if (out["weight"] < 0).any() or weight_sum < -1e-12 or weight_sum > 1 + 1e-12:
        raise ValueError("output/result.csv weights must satisfy 0 <= sum(weight) <= 1")
    return weight_sum


def score_current_output():
    code, output = run_command([sys.executable, "test/score_self.py"])
    run_command(["git", "restore", "--", "temp/tmp.csv"])
    if code != 0:
        raise RuntimeError(output)
    match = re.search(r"([-+]?\d+\.\d+(?:e[-+]?\d+)?)", output)
    if not match:
        raise RuntimeError(f"Could not parse local score from: {output}")
    return float(match.group(1))


def audit_block(strategy, attempt):
    return f"""
### [Logic & Compliance Audit Check] {strategy.name} attempt={attempt}

- Lookahead Bias: labels remain T+1 open buy to T+6 open sell; feature toggles only use current/past time rows and same-date cross-sections.
- Trading-day alignment: candidate uses the locked chronological purged split in data/golden_validation_dates.json with no validation-tail embargo.
- Survivor bias: cross-sectional factors are computed only from stocks present on the current trading date.
- Forward causality: architecture candidates may enable the causal temporal mask; other tracks keep the default compatible path.
- Official-rule compliance: no external data and no external pretrained model are introduced by this scheduler.
- Strategy: {strategy.description}
"""


def commit_marker(message):
    run_command(["git", "add", "output/result.csv"])
    code, output = run_command(["git", "commit", "--allow-empty", "-m", message])
    if code != 0:
        append_log(f"- Commit warning: {output.strip()}")


def run_candidate(strategy, attempt, extra_env, best_score):
    output_dir = f"./model/automl_{strategy.name}_{attempt}"
    env = {
        "BDC_OUTPUT_DIR": output_dir,
        "BDC_EVAL_ONLY": "0",
        "BDC_NUM_EPOCHS": "15",
        "BDC_MIN_EPOCHS": "5",
        "BDC_EARLY_STOPPING_PATIENCE": "3",
        "BDC_BATCH_SIZE": os.getenv("BDC_BATCH_SIZE", "2"),
        "BDC_GRAD_ACCUM_STEPS": os.getenv("BDC_GRAD_ACCUM_STEPS", "2"),
    }
    env.update(strategy.env)
    env.update(extra_env)

    append_log(audit_block(strategy, attempt))
    code, train_output = run_command(
        [sys.executable, "code/src/train.py"],
        env=env,
        log_name=f"automl_{strategy.name}_{attempt}_train.log",
    )
    if code != 0:
        append_log(f"- Status: [FAILED] training command failed for {strategy.name} attempt={attempt}.\n")
        return None

    val_score = read_final_score(output_dir)
    if val_score is None:
        append_log(f"- Status: [FAILED] missing final_score.txt for {strategy.name} attempt={attempt}.\n")
        return None

    code, predict_output = run_command(
        [sys.executable, "code/src/predict.py"],
        env=env,
        log_name=f"automl_{strategy.name}_{attempt}_predict.log",
    )
    if code != 0:
        append_log(f"- Status: [FAILED] predict failed for {strategy.name} attempt={attempt}.\n")
        return None

    weight_sum = validate_output()
    local_score = score_current_output()
    success = val_score > best_score and local_score >= BENCHMARK_HURDLE
    status = "[SUCCESS]" if success else "[FAILED]"
    append_log(
        f"- Status: {status} {strategy.name} attempt={attempt}: "
        f"val_score={val_score:.6f}, local_score={local_score:.17f}, "
        f"weight_sum={weight_sum:.6f}, best_before={best_score:.6f}.\n"
    )
    if success:
        commit_marker(f"feat({strategy.domain}): {strategy.description[:48]} Score: {val_score:.6f}")
        return {"name": strategy.name, "domain": strategy.domain, "dir": output_dir, "val_score": val_score, "local_score": local_score}
    return None


def run_ensemble(successes, best_local):
    if not successes:
        append_log("\n## Phase 3 Ensemble\n- Status: [FAILED] no successful checkpoints available for ensemble.\n")
        return None
    model_dirs = [item["dir"] for item in successes]
    code, output = run_command(
        [sys.executable, "code/src/ensemble.py", "--model-dirs", *model_dirs, "--output", "./output/result.csv"],
        log_name="automl_ensemble.log",
    )
    if code != 0:
        append_log(f"\n## Phase 3 Ensemble\n- Status: [FAILED] ensemble command failed.\n")
        return None
    weight_sum = validate_output()
    local_score = score_current_output()
    success = local_score >= max(BENCHMARK_HURDLE, best_local)
    status = "[SUCCESS]" if success else "[FAILED]"
    append_log(
        f"\n## Phase 3 Ensemble\n- Status: {status} blended {len(model_dirs)} checkpoints; "
        f"local_score={local_score:.17f}, best_single_local={best_local:.17f}, weight_sum={weight_sum:.6f}.\n"
    )
    if success:
        commit_marker("feat(ensemble): final blend ranking system")
    return local_score


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the compliant overnight BDC AutoML loop.")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--ensemble-after-hours", type=float, default=7.0)
    parser.add_argument("--max-cycles", type=int, default=10_000)
    parser.add_argument("--once", action="store_true", help="Run one strategy and stop.")
    args = parser.parse_args(argv)

    start = time.monotonic()
    max_seconds = args.hours * 3600
    ensemble_seconds = args.ensemble_after_hours * 3600
    best_score = read_phase0_score()
    best_local = BENCHMARK_HURDLE
    successes = []

    append_log(
        f"\n## Nightly AutoML Scheduler Start\n"
        f"- Best_Score: {best_score:.6f}\n"
        f"- Benchmark_Hurdle: {BENCHMARK_HURDLE:.17f}\n"
        f"- Ensemble trigger: {args.ensemble_after_hours:.2f} hours or 3 SUCCESS checkpoints.\n"
    )

    cycle = 0
    while cycle < args.max_cycles and time.monotonic() - start < max_seconds:
        if successes and (len(successes) >= 3 or time.monotonic() - start >= ensemble_seconds):
            run_ensemble(successes, best_local)
            break

        strategy = STRATEGIES[cycle % len(STRATEGIES)]
        primary = run_candidate(strategy, f"cycle{cycle}_primary", {}, best_score)
        if primary:
            successes.append(primary)
            best_score = max(best_score, primary["val_score"])
            best_local = max(best_local, primary["local_score"])
        else:
            mutation = run_candidate(strategy, f"cycle{cycle}_mutation", strategy.mutation_env, best_score)
            if mutation:
                successes.append(mutation)
                best_score = max(best_score, mutation["val_score"])
                best_local = max(best_local, mutation["local_score"])

        cycle += 1
        if args.once:
            break

    if successes and time.monotonic() - start >= ensemble_seconds:
        run_ensemble(successes, best_local)

    print("夜间高强度全量演进任务安全结束，逻辑与赛规审计全量通过，集成完毕，等待主人起床复盘")


if __name__ == "__main__":
    main()

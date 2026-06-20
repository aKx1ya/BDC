#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_ROOT.parents[1]
PIPELINE_DIR = WORKFLOW_ROOT / "pipelines"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step5_outputs import (  # noqa: E402
    DEFAULT_CANDIDATE_SIZE,
    DEFAULT_RANDOM_SEED,
    build_step5_outputs,
)
from validate_step5 import Step5ValidationError, manifest_value, read_csv, validate_inputs, validate_outputs  # noqa: E402


class Step5RunnerError(Exception):
    """workflow_0.1 Step-5 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step5_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step5RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

    values: dict[str, str] = {}
    for line in active_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key.startswith("active_") or key == "status":
            values[key] = value.strip()
    return values


def ensure_active_step5(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step5RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-5":
        raise Step5RunnerError(f"active_stage must be Step-5, got {stage!r}")
    return active


def report_is_success(experiment_dir: Path, step_name: str) -> bool:
    step_slug = step_name.lower().replace("-", "")
    report_path = experiment_dir / "notes" / f"{step_slug}_run_report.md"
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def find_latest_successful_experiment(workflow_root: Path, step_name: str) -> Path:
    experiments_dir = workflow_root / "experiments"
    needle = step_name.lower().replace("-", "")
    step_slug = step_name.lower().replace("-", "")
    candidates = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and needle in path.name.lower().replace("-", "") and report_is_success(path, step_name)
    ]
    if not candidates:
        raise Step5RunnerError(f"no successful {step_name} experiment found under {experiments_dir}")
    return max(candidates, key=lambda path: (path / "notes" / f"{step_slug}_run_report.md").stat().st_mtime)


def read_manifest_value(path: Path, item: str) -> str:
    return manifest_value(read_csv(path), item)


def resolve_input_experiments(
    workflow_root: Path,
    *,
    step2_experiment: str | None,
    step3_experiment: str | None,
    step4_experiment: str | None,
) -> tuple[Path, Path, Path]:
    if step4_experiment:
        step4_dir = workflow_root / "experiments" / step4_experiment
    else:
        step4_dir = find_latest_successful_experiment(workflow_root, "Step-4")

    step4_manifest = step4_dir / "outputs" / "step4" / "step4_split_manifest.csv"
    inferred_step3 = read_manifest_value(step4_manifest, "input_step3_experiment")
    if step3_experiment and inferred_step3 and step3_experiment != inferred_step3:
        raise Step5RunnerError(
            f"selected Step-3 {step3_experiment!r} does not match Step-4 manifest input {inferred_step3!r}"
        )
    step3_name = step3_experiment or inferred_step3
    if not step3_name:
        raise Step5RunnerError(f"cannot infer Step-3 experiment from {step4_manifest}")
    step3_dir = workflow_root / "experiments" / step3_name

    step3_manifest = step3_dir / "outputs" / "step3" / "step3_sample_manifest.csv"
    inferred_step2 = read_manifest_value(step3_manifest, "input_step2_experiment")
    if step2_experiment and inferred_step2 and step2_experiment != inferred_step2:
        raise Step5RunnerError(
            f"selected Step-2 {step2_experiment!r} does not match Step-3 manifest input {inferred_step2!r}"
        )
    step2_name = step2_experiment or inferred_step2
    if not step2_name:
        raise Step5RunnerError(f"cannot infer Step-2 experiment from {step3_manifest}")
    step2_dir = workflow_root / "experiments" / step2_name
    return step2_dir, step3_dir, step4_dir


def metrics_to_markdown(metrics: dict[str, object]) -> str:
    if not metrics:
        return "_not available_\n"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(metrics.items())) + "\n"


def write_report(
    report_path: Path,
    *,
    status: str,
    active: dict[str, str],
    step2_experiment_dir: Path | None,
    step3_experiment_dir: Path | None,
    step4_experiment_dir: Path | None,
    experiment_dir: Path,
    output_dir: Path,
    model_dir: Path,
    params: dict[str, object],
    input_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Step-5 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `step2_experiment_dir`: {step2_experiment_dir or ""}
- `step3_experiment_dir`: {step3_experiment_dir or ""}
- `step4_experiment_dir`: {step4_experiment_dir or ""}
- `experiment_dir`: {experiment_dir}
- `output_dir`: {output_dir}
- `model_dir`: {model_dir}

## Params

{metrics_to_markdown(params)}
## Input Metrics

{metrics_to_markdown(input_metrics or {})}
## Output Metrics

{metrics_to_markdown(output_metrics or {})}
## Error

{error or "_none_"}
"""
    report_path.write_text(content, encoding="utf-8")


def make_manifest_note(
    step2_experiment_dir: Path,
    step3_experiment_dir: Path,
    step4_experiment_dir: Path,
    input_metrics: dict[str, object],
) -> str:
    return (
        "正式 Step-5 由 workflow_0.1/run_step5.py 读取健康 Step-2/3/4 实验输出生成；"
        f"input_step2_experiment={step2_experiment_dir.name}，"
        f"input_step3_experiment={step3_experiment_dir.name}，"
        f"input_step4_experiment={step4_experiment_dir.name}，"
        f"prediction_date={input_metrics.get('input_step2_latest_T', '')}，"
        "本阶段训练 baseline 模型并生成 Top30 候选池，不生成 result.csv。"
    )


def run_step5(
    *,
    project_root: Path = PROJECT_ROOT,
    step2_experiment: str | None = None,
    step3_experiment: str | None = None,
    step4_experiment: str | None = None,
    experiment_name: str | None = None,
    candidate_size: int = DEFAULT_CANDIDATE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Path]:
    project_root = Path(project_root)
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = workflow_root / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step5"
    model_dir = experiment_dir / "models" / "step5"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step5_run_report.md"
    params = {"candidate_size": candidate_size, "random_seed": random_seed, "model_family": "baseline_correlation_rank"}

    active: dict[str, str] = {}
    step2_experiment_dir: Path | None = None
    step3_experiment_dir: Path | None = None
    step4_experiment_dir: Path | None = None
    input_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None

    try:
        active = ensure_active_step5(project_root)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        step2_experiment_dir, step3_experiment_dir, step4_experiment_dir = resolve_input_experiments(
            workflow_root,
            step2_experiment=step2_experiment,
            step3_experiment=step3_experiment,
            step4_experiment=step4_experiment,
        )

        print(f"Step-5 正式流程：读取 Step-2 实验 {step2_experiment_dir.name}")
        print(f"Step-5 正式流程：读取 Step-3 实验 {step3_experiment_dir.name}")
        print(f"Step-5 正式流程：读取 Step-4 实验 {step4_experiment_dir.name}")
        input_metrics = validate_inputs(step2_experiment_dir, step3_experiment_dir, step4_experiment_dir)

        print("Step-5 正式流程：训练 baseline 模型并生成 Top30 候选池")
        build_step5_outputs(
            step2_output_dir=step2_experiment_dir / "outputs" / "step2",
            step3_output_dir=step3_experiment_dir / "outputs" / "step3",
            step4_output_dir=step4_experiment_dir / "outputs" / "step4",
            output_dir=output_dir,
            model_dir=model_dir,
            input_step2_experiment=step2_experiment_dir.name,
            input_step3_experiment=step3_experiment_dir.name,
            input_step4_experiment=step4_experiment_dir.name,
            candidate_size=candidate_size,
            random_seed=random_seed,
            note=make_manifest_note(step2_experiment_dir, step3_experiment_dir, step4_experiment_dir, input_metrics),
        )

        print("Step-5 正式流程：校验模型实验输出")
        output_metrics = validate_outputs(
            output_dir,
            model_dir,
            step4_output_dir=step4_experiment_dir / "outputs" / "step4",
            input_metrics=input_metrics,
            candidate_size=candidate_size,
            random_seed=random_seed,
        )

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            step2_experiment_dir=step2_experiment_dir,
            step3_experiment_dir=step3_experiment_dir,
            step4_experiment_dir=step4_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            model_dir=model_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
    except (Step5RunnerError, Step5ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            step2_experiment_dir=step2_experiment_dir,
            step3_experiment_dir=step3_experiment_dir,
            step4_experiment_dir=step4_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            model_dir=model_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step5RunnerError):
            raise
        raise Step5RunnerError(str(exc)) from exc

    print(f"Step-5 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "model_dir": model_dir,
        "report_path": report_path,
        "step2_experiment_dir": step2_experiment_dir,
        "step3_experiment_dir": step3_experiment_dir,
        "step4_experiment_dir": step4_experiment_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-5 pipeline.")
    parser.add_argument("--step2-experiment", default=None, help="指定健康 Step-2 实验目录名；默认从 Step-3 manifest 推断")
    parser.add_argument("--step3-experiment", default=None, help="指定健康 Step-3 实验目录名；默认从 Step-4 manifest 推断")
    parser.add_argument("--step4-experiment", default=None, help="指定健康 Step-4 实验目录名；默认自动寻找最近 SUCCESS 的 Step-4")
    parser.add_argument("--experiment-name", default=None, help="Step-5 实验目录名；默认 exp_YYYYMMDD_step5_workflow_0_1")
    parser.add_argument("--candidate-size", type=int, default=DEFAULT_CANDIDATE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step5(
            step2_experiment=args.step2_experiment,
            step3_experiment=args.step3_experiment,
            step4_experiment=args.step4_experiment,
            experiment_name=args.experiment_name,
            candidate_size=args.candidate_size,
            random_seed=args.random_seed,
        )
    except Step5RunnerError as exc:
        print(f"Step-5 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

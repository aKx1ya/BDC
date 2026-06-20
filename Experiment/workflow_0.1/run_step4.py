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

from build_step4_outputs import (  # noqa: E402
    DEFAULT_EVAL_DAYS,
    DEFAULT_FINAL_TEST_DAYS,
    DEFAULT_GAP_DAYS,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_WALK_FORWARD_STEP,
    build_step4_outputs,
)
from validate_step4 import Step4ValidationError, validate_outputs, validate_step3_input  # noqa: E402


class Step4RunnerError(Exception):
    """workflow_0.1 Step-4 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step4_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step4RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

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


def ensure_active_step4(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step4RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-4":
        raise Step4RunnerError(f"active_stage must be Step-4, got {stage!r}")
    return active


def step3_report_is_success(experiment_dir: Path) -> bool:
    report_path = experiment_dir / "notes" / "step3_run_report.md"
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def find_latest_successful_step3_experiment(workflow_root: Path) -> Path:
    experiments_dir = workflow_root / "experiments"
    candidates = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and "step3" in path.name and step3_report_is_success(path)
    ]
    if not candidates:
        raise Step4RunnerError(f"no successful Step-3 experiment found under {experiments_dir}")
    return max(candidates, key=lambda path: (path / "notes" / "step3_run_report.md").stat().st_mtime)


def metrics_to_markdown(metrics: dict[str, object]) -> str:
    if not metrics:
        return "_not available_\n"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(metrics.items())) + "\n"


def write_report(
    report_path: Path,
    *,
    status: str,
    active: dict[str, str],
    step3_experiment_dir: Path | None,
    experiment_dir: Path,
    output_dir: Path,
    split_params: dict[str, object],
    input_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Step-4 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `step3_experiment_dir`: {step3_experiment_dir or ""}
- `experiment_dir`: {experiment_dir}
- `output_dir`: {output_dir}

## Split Params

{metrics_to_markdown(split_params)}
## Input Metrics

{metrics_to_markdown(input_metrics or {})}
## Output Metrics

{metrics_to_markdown(output_metrics or {})}
## Error

{error or "_none_"}
"""
    report_path.write_text(content, encoding="utf-8")


def make_manifest_note(step3_experiment_dir: Path, input_metrics: dict[str, object]) -> str:
    return (
        "正式 Step-4 由 workflow_0.1/run_step4.py 读取健康 Step-3 样本资产生成；"
        f"input_step3_experiment={step3_experiment_dir.name}，"
        f"sample_date_range={input_metrics.get('input_step3_sample_date_start', '')}"
        f"~{input_metrics.get('input_step3_sample_date_end', '')}，"
        f"sample_date_count={input_metrics.get('input_step3_sample_date_count', '')}，"
        "本阶段不联网、不重新计算特征/标签、不训练模型、不生成 result.csv。"
    )


def split_params_dict(
    *,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
    train_ratio: float,
    final_test_days: int,
) -> dict[str, object]:
    return {
        "train_window": train_window,
        "gap_days": gap_days,
        "eval_days": eval_days,
        "walk_forward_step": walk_forward_step,
        "train_ratio": f"{train_ratio:.2f}",
        "final_test_days": final_test_days,
    }


def run_step4(
    *,
    project_root: Path = PROJECT_ROOT,
    step3_experiment: str | None = None,
    experiment_name: str | None = None,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    gap_days: int = DEFAULT_GAP_DAYS,
    eval_days: int = DEFAULT_EVAL_DAYS,
    walk_forward_step: int = DEFAULT_WALK_FORWARD_STEP,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    final_test_days: int = DEFAULT_FINAL_TEST_DAYS,
) -> dict[str, Path]:
    project_root = Path(project_root)
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = workflow_root / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step4"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step4_run_report.md"

    active: dict[str, str] = {}
    step3_experiment_dir: Path | None = None
    input_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None
    split_params = split_params_dict(
        train_window=train_window,
        gap_days=gap_days,
        eval_days=eval_days,
        walk_forward_step=walk_forward_step,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
    )

    try:
        active = ensure_active_step4(project_root)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        if step3_experiment:
            step3_experiment_dir = workflow_root / "experiments" / step3_experiment
        else:
            step3_experiment_dir = find_latest_successful_step3_experiment(workflow_root)

        print(f"Step-4 正式流程：读取 Step-3 实验 {step3_experiment_dir.name}")
        input_metrics = validate_step3_input(step3_experiment_dir)

        print("Step-4 正式流程：生成切分与 walk-forward 标准输出")
        build_step4_outputs(
            step3_output_dir=step3_experiment_dir / "outputs" / "step3",
            output_dir=output_dir,
            input_step3_experiment=step3_experiment_dir.name,
            train_window=train_window,
            gap_days=gap_days,
            eval_days=eval_days,
            walk_forward_step=walk_forward_step,
            train_ratio=train_ratio,
            final_test_days=final_test_days,
            note=make_manifest_note(step3_experiment_dir, input_metrics),
        )

        print("Step-4 正式流程：校验切分与 walk-forward 输出")
        output_metrics = validate_outputs(
            output_dir,
            input_step3_metrics=input_metrics,
            train_window=train_window,
            gap_days=gap_days,
            eval_days=eval_days,
            walk_forward_step=walk_forward_step,
            train_ratio=train_ratio,
            final_test_days=final_test_days,
        )

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            step3_experiment_dir=step3_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            split_params=split_params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
    except (Step4RunnerError, Step4ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            step3_experiment_dir=step3_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            split_params=split_params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step4RunnerError):
            raise
        raise Step4RunnerError(str(exc)) from exc

    print(f"Step-4 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "step3_experiment_dir": step3_experiment_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-4 pipeline.")
    parser.add_argument("--step3-experiment", default=None, help="指定健康 Step-3 实验目录名；默认自动寻找最近 SUCCESS 的 Step-3")
    parser.add_argument("--experiment-name", default=None, help="Step-4 实验目录名；默认 exp_YYYYMMDD_step4_workflow_0_1")
    parser.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--walk-forward-step", type=int, default=DEFAULT_WALK_FORWARD_STEP)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--final-test-days", type=int, default=DEFAULT_FINAL_TEST_DAYS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step4(
            step3_experiment=args.step3_experiment,
            experiment_name=args.experiment_name,
            train_window=args.train_window,
            gap_days=args.gap_days,
            eval_days=args.eval_days,
            walk_forward_step=args.walk_forward_step,
            train_ratio=args.train_ratio,
            final_test_days=args.final_test_days,
        )
    except Step4RunnerError as exc:
        print(f"Step-4 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

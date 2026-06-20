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

from build_step3_outputs import build_step3_outputs  # noqa: E402
from validate_step3 import Step3ValidationError, validate_outputs, validate_step2_input  # noqa: E402


class Step3RunnerError(Exception):
    """workflow_0.1 Step-3 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step3_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step3RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

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


def ensure_active_step3(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step3RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-3":
        raise Step3RunnerError(f"active_stage must be Step-3, got {stage!r}")
    return active


def step2_report_is_success(experiment_dir: Path) -> bool:
    report_path = experiment_dir / "notes" / "step2_run_report.md"
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def find_latest_successful_step2_experiment(workflow_root: Path) -> Path:
    experiments_dir = workflow_root / "experiments"
    candidates = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir() and "step2" in path.name and step2_report_is_success(path)
    ]
    if not candidates:
        raise Step3RunnerError(f"no successful Step-2 experiment found under {experiments_dir}")
    return max(candidates, key=lambda path: (path / "notes" / "step2_run_report.md").stat().st_mtime)


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
    experiment_dir: Path,
    output_dir: Path,
    input_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Step-3 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `step2_experiment_dir`: {step2_experiment_dir or ""}
- `experiment_dir`: {experiment_dir}
- `output_dir`: {output_dir}

## Input Metrics

{metrics_to_markdown(input_metrics or {})}
## Output Metrics

{metrics_to_markdown(output_metrics or {})}
## Error

{error or "_none_"}
"""
    report_path.write_text(content, encoding="utf-8")


def make_manifest_note(step2_experiment_dir: Path, input_metrics: dict[str, object]) -> str:
    return (
        "正式 Step-3 由 workflow_0.1/run_step3.py 读取健康 Step-2 实验输出生成；"
        f"input_step2_experiment={step2_experiment_dir.name}，"
        f"input_step2_latest_T={input_metrics.get('input_step2_latest_T', '')}，"
        f"last_labelable_T={input_metrics.get('last_labelable_T', '')}，"
        "本阶段不联网、不训练模型、不生成 candidate_top30.csv 或 result.csv。"
    )


def run_step3(
    *,
    project_root: Path = PROJECT_ROOT,
    step2_experiment: str | None = None,
    experiment_name: str | None = None,
    expected_stock_count: int = 300,
) -> dict[str, Path]:
    project_root = Path(project_root)
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = workflow_root / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step3"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step3_run_report.md"

    active: dict[str, str] = {}
    step2_experiment_dir: Path | None = None
    input_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None

    try:
        active = ensure_active_step3(project_root)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        if step2_experiment:
            step2_experiment_dir = workflow_root / "experiments" / step2_experiment
        else:
            step2_experiment_dir = find_latest_successful_step2_experiment(workflow_root)

        print(f"Step-3 正式流程：读取 Step-2 实验 {step2_experiment_dir.name}")
        input_metrics = validate_step2_input(step2_experiment_dir, expected_stock_count=expected_stock_count)

        print("Step-3 正式流程：生成标准样本输出")
        build_step3_outputs(
            step2_output_dir=step2_experiment_dir / "outputs" / "step2",
            output_dir=output_dir,
            input_step2_experiment=step2_experiment_dir.name,
            note=make_manifest_note(step2_experiment_dir, input_metrics),
        )

        print("Step-3 正式流程：校验标准样本输出")
        output_metrics = validate_outputs(
            output_dir,
            input_step2_latest_t=str(input_metrics["input_step2_latest_T"]),
            last_labelable_t=str(input_metrics["last_labelable_T"]),
            expected_stock_count=expected_stock_count,
        )

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            step2_experiment_dir=step2_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
    except (Step3RunnerError, Step3ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            step2_experiment_dir=step2_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step3RunnerError):
            raise
        raise Step3RunnerError(str(exc)) from exc

    print(f"Step-3 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "step2_experiment_dir": step2_experiment_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-3 pipeline.")
    parser.add_argument("--step2-experiment", default=None, help="指定健康 Step-2 实验目录名；默认自动寻找最近 SUCCESS 的 Step-2")
    parser.add_argument("--experiment-name", default=None, help="Step-3 实验目录名；默认 exp_YYYYMMDD_step3_workflow_0_1")
    parser.add_argument("--expected-stock-count", type=int, default=300, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step3(
            step2_experiment=args.step2_experiment,
            experiment_name=args.experiment_name,
            expected_stock_count=args.expected_stock_count,
        )
    except Step3RunnerError as exc:
        print(f"Step-3 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

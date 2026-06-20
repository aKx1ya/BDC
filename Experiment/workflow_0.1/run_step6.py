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

from build_step6_outputs import (  # noqa: E402
    DEFAULT_MAX_PER_SECTOR,
    DEFAULT_MAX_STOCK_COUNT,
    DEFAULT_MIN_TURNOVER,
    build_step6_outputs,
)
from validate_step6 import Step6ValidationError, manifest_value, read_csv, validate_inputs, validate_outputs  # noqa: E402


class Step6RunnerError(Exception):
    """workflow_0.1 Step-6 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step6_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step6RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

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


def ensure_active_step6(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step6RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-6":
        raise Step6RunnerError(f"active_stage must be Step-6, got {stage!r}")
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
        raise Step6RunnerError(f"no successful {step_name} experiment found under {experiments_dir}")
    return max(candidates, key=lambda path: (path / "notes" / f"{step_slug}_run_report.md").stat().st_mtime)


def read_manifest_value(path: Path, item: str) -> str:
    return manifest_value(read_csv(path), item)


def resolve_input_experiments(
    workflow_root: Path,
    *,
    step5_experiment: str | None,
    step2_experiment: str | None,
) -> tuple[Path, Path]:
    if step5_experiment:
        step5_dir = workflow_root / "experiments" / step5_experiment
    else:
        step5_dir = find_latest_successful_experiment(workflow_root, "Step-5")

    step5_manifest = step5_dir / "outputs" / "step5" / "step5_model_manifest.csv"
    inferred_step2 = read_manifest_value(step5_manifest, "input_step2_experiment")
    if step2_experiment and inferred_step2 and step2_experiment != inferred_step2:
        raise Step6RunnerError(
            f"selected Step-2 {step2_experiment!r} does not match Step-5 manifest input {inferred_step2!r}"
        )
    step2_name = step2_experiment or inferred_step2
    if not step2_name:
        raise Step6RunnerError(f"cannot infer Step-2 experiment from {step5_manifest}")
    step2_dir = workflow_root / "experiments" / step2_name
    return step5_dir, step2_dir


def metrics_to_markdown(metrics: dict[str, object]) -> str:
    if not metrics:
        return "_not available_\n"
    lines = []
    for key, value in sorted(metrics.items()):
        if key == "input_candidate_codes" and isinstance(value, list):
            lines.append(f"- {key}: {len(value)} codes")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def write_report(
    report_path: Path,
    *,
    status: str,
    active: dict[str, str],
    step5_experiment_dir: Path | None,
    step2_experiment_dir: Path | None,
    experiment_dir: Path,
    output_dir: Path,
    params: dict[str, object],
    input_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Step-6 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `step5_experiment_dir`: {step5_experiment_dir or ""}
- `step2_experiment_dir`: {step2_experiment_dir or ""}
- `experiment_dir`: {experiment_dir}
- `output_dir`: {output_dir}

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


def make_manifest_note(step5_experiment_dir: Path, step2_experiment_dir: Path, input_metrics: dict[str, object]) -> str:
    return (
        "正式 Step-6 由 workflow_0.1/run_step6.py 读取健康 Step-5 Top30 和同链路 Step-2 latest_T 特征生成；"
        f"input_step5_experiment={step5_experiment_dir.name}，"
        f"input_step2_experiment={step2_experiment_dir.name}，"
        f"candidate_date={input_metrics.get('input_candidate_date', '')}，"
        "本阶段只在 Top30 内精排并生成 step6_result.csv，不训练模型也不评分。"
    )


def run_step6(
    *,
    project_root: Path = PROJECT_ROOT,
    step5_experiment: str | None = None,
    step2_experiment: str | None = None,
    experiment_name: str | None = None,
    max_stock_count: int = DEFAULT_MAX_STOCK_COUNT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
    single_weight: float | None = None,
) -> dict[str, Path]:
    project_root = Path(project_root)
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = workflow_root / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step6"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step6_run_report.md"
    params = {
        "max_stock_count": max_stock_count,
        "max_per_sector": max_per_sector,
        "min_turnover": min_turnover,
        "single_weight": single_weight if single_weight is not None else f"1/{max_stock_count}",
    }

    active: dict[str, str] = {}
    step5_experiment_dir: Path | None = None
    step2_experiment_dir: Path | None = None
    input_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None

    try:
        active = ensure_active_step6(project_root)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        step5_experiment_dir, step2_experiment_dir = resolve_input_experiments(
            workflow_root,
            step5_experiment=step5_experiment,
            step2_experiment=step2_experiment,
        )

        print(f"Step-6 正式流程：读取 Step-5 实验 {step5_experiment_dir.name}")
        print(f"Step-6 正式流程：读取同链路 Step-2 实验 {step2_experiment_dir.name}")
        input_metrics = validate_inputs(step5_experiment_dir, step2_experiment_dir)

        print("Step-6 正式流程：执行 Top30 内精排并生成 result.csv")
        build_step6_outputs(
            step5_output_dir=step5_experiment_dir / "outputs" / "step5",
            step2_output_dir=step2_experiment_dir / "outputs" / "step2",
            output_dir=output_dir,
            input_step5_experiment=step5_experiment_dir.name,
            input_step2_experiment=step2_experiment_dir.name,
            max_stock_count=max_stock_count,
            max_per_sector=max_per_sector,
            min_turnover=min_turnover,
            single_weight=single_weight,
            note=make_manifest_note(step5_experiment_dir, step2_experiment_dir, input_metrics),
        )

        print("Step-6 正式流程：校验精排输出与 result.csv")
        output_metrics = validate_outputs(
            output_dir,
            input_metrics=input_metrics,
            max_stock_count=max_stock_count,
            max_per_sector=max_per_sector,
            min_turnover=min_turnover,
        )

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            step5_experiment_dir=step5_experiment_dir,
            step2_experiment_dir=step2_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
    except (Step6RunnerError, Step6ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            step5_experiment_dir=step5_experiment_dir,
            step2_experiment_dir=step2_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step6RunnerError):
            raise
        raise Step6RunnerError(str(exc)) from exc

    print(f"Step-6 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "step5_experiment_dir": step5_experiment_dir,
        "step2_experiment_dir": step2_experiment_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-6 pipeline.")
    parser.add_argument("--step5-experiment", default=None, help="指定健康 Step-5 实验目录名；默认自动寻找最近 SUCCESS 的 Step-5")
    parser.add_argument("--step2-experiment", default=None, help="指定健康 Step-2 实验目录名；默认从 Step-5 manifest 推断")
    parser.add_argument("--experiment-name", default=None, help="Step-6 实验目录名；默认 exp_YYYYMMDD_step6_workflow_0_1")
    parser.add_argument("--max-stock-count", type=int, default=DEFAULT_MAX_STOCK_COUNT)
    parser.add_argument("--max-per-sector", type=int, default=DEFAULT_MAX_PER_SECTOR)
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER)
    parser.add_argument("--single-weight", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step6(
            step5_experiment=args.step5_experiment,
            step2_experiment=args.step2_experiment,
            experiment_name=args.experiment_name,
            max_stock_count=args.max_stock_count,
            max_per_sector=args.max_per_sector,
            min_turnover=args.min_turnover,
            single_weight=args.single_weight,
        )
    except Step6RunnerError as exc:
        print(f"Step-6 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

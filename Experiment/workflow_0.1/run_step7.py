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

from build_step7_outputs import (  # noqa: E402
    DEFAULT_OFFICIAL_SCRIPT_PATH,
    DEFAULT_TEAM_NAME,
    DEFAULT_TEST_DATA_PATH,
    VALID_SCORE_MODES,
    build_step7_outputs,
)
from validate_step7 import Step7ValidationError, manifest_value, read_csv, validate_inputs, validate_outputs  # noqa: E402


class Step7RunnerError(Exception):
    """workflow_0.1 Step-7 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step7_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step7RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

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


def ensure_active_step7(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step7RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-7":
        raise Step7RunnerError(f"active_stage must be Step-7, got {stage!r}")
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
        raise Step7RunnerError(f"no successful {step_name} experiment found under {experiments_dir}")
    return max(candidates, key=lambda path: (path / "notes" / f"{step_slug}_run_report.md").stat().st_mtime)


def metrics_to_markdown(metrics: dict[str, object]) -> str:
    if not metrics:
        return "_not available_\n"
    lines = []
    for key, value in sorted(metrics.items()):
        if key == "input_result_codes" and isinstance(value, list):
            lines.append(f"- {key}: {len(value)} codes")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def write_report(
    report_path: Path,
    *,
    status: str,
    active: dict[str, str],
    step6_experiment_dir: Path | None,
    experiment_dir: Path,
    output_dir: Path,
    workspace_dir: Path,
    params: dict[str, object],
    input_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Step-7 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `step6_experiment_dir`: {step6_experiment_dir or ""}
- `experiment_dir`: {experiment_dir}
- `output_dir`: {output_dir}
- `official_scoring_workspace`: {workspace_dir}

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


def make_manifest_note(step6_experiment_dir: Path, input_metrics: dict[str, object], *, mode: str) -> str:
    return (
        "正式 Step-7 由 workflow_0.1/run_step7.py 读取健康 Step-6 输出生成；"
        f"input_step6_experiment={step6_experiment_dir.name}，"
        f"candidate_date={input_metrics.get('input_candidate_date', '')}，"
        f"mode={mode}。"
        "本阶段先冻结 result.csv；评分结果只能用于复盘和下一轮实验，不能回改本轮 Step-6。"
    )


def run_step7(
    *,
    project_root: Path = PROJECT_ROOT,
    step6_experiment: str | None = None,
    experiment_name: str | None = None,
    mode: str = "freeze-only",
    team_name: str = DEFAULT_TEAM_NAME,
    official_script_path: Path = DEFAULT_OFFICIAL_SCRIPT_PATH,
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
) -> dict[str, Path]:
    if mode not in VALID_SCORE_MODES:
        raise Step7RunnerError(f"mode must be one of {sorted(VALID_SCORE_MODES)}, got {mode!r}")

    project_root = Path(project_root)
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = workflow_root / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step7"
    workspace_dir = experiment_dir / "official_scoring_workspace"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step7_run_report.md"
    params = {
        "mode": mode,
        "team_name": team_name,
        "official_script_path": official_script_path,
        "test_data_path": test_data_path if mode == "local-score" else "",
    }

    active: dict[str, str] = {}
    step6_experiment_dir: Path | None = None
    input_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None

    try:
        active = ensure_active_step7(project_root)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        if step6_experiment:
            step6_experiment_dir = workflow_root / "experiments" / step6_experiment
        else:
            step6_experiment_dir = find_latest_successful_experiment(workflow_root, "Step-6")

        print(f"Step-7 正式流程：读取 Step-6 实验 {step6_experiment_dir.name}")
        input_metrics = validate_inputs(step6_experiment_dir)

        print(f"Step-7 正式流程：冻结 result.csv，mode={mode}")
        build_step7_outputs(
            step6_output_dir=step6_experiment_dir / "outputs" / "step6",
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            mode=mode,
            input_step6_experiment=step6_experiment_dir.name,
            team_name=team_name,
            official_script_path=Path(official_script_path),
            test_data_path=Path(test_data_path),
            note=make_manifest_note(step6_experiment_dir, input_metrics, mode=mode),
        )

        print("Step-7 正式流程：校验冻结、评分与防泄漏输出")
        output_metrics = validate_outputs(
            output_dir,
            mode=mode,
            step6_result_path=Path(input_metrics["input_step6_result_path"]),
            input_metrics=input_metrics,
        )

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            step6_experiment_dir=step6_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
        )
    except (Step7RunnerError, Step7ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            step6_experiment_dir=step6_experiment_dir,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            workspace_dir=workspace_dir,
            params=params,
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step7RunnerError):
            raise
        raise Step7RunnerError(str(exc)) from exc

    print(f"Step-7 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "workspace_dir": workspace_dir,
        "report_path": report_path,
        "step6_experiment_dir": step6_experiment_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-7 pipeline.")
    parser.add_argument("--step6-experiment", default=None, help="指定健康 Step-6 实验目录名；默认自动寻找最近 SUCCESS 的 Step-6")
    parser.add_argument("--experiment-name", default=None, help="Step-7 实验目录名；默认 exp_YYYYMMDD_step7_workflow_0_1")
    parser.add_argument("--mode", choices=sorted(VALID_SCORE_MODES), default="freeze-only")
    parser.add_argument("--team-name", default=DEFAULT_TEAM_NAME)
    parser.add_argument("--official-script-path", type=Path, default=DEFAULT_OFFICIAL_SCRIPT_PATH)
    parser.add_argument("--test-data-path", type=Path, default=DEFAULT_TEST_DATA_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step7(
            step6_experiment=args.step6_experiment,
            experiment_name=args.experiment_name,
            mode=args.mode,
            team_name=args.team_name,
            official_script_path=args.official_script_path,
            test_data_path=args.test_data_path,
        )
    except Step7RunnerError as exc:
        print(f"Step-7 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

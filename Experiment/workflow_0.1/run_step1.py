#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_ROOT.parents[1]
PIPELINE_DIR = WORKFLOW_ROOT / "pipelines"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step1_outputs import build_step1_outputs  # noqa: E402
from validate_step1 import Step1ValidationError, validate_outputs, validate_raw_data  # noqa: E402


class Step1RunnerError(Exception):
    """workflow_0.1 Step-1 正式调度失败。"""


def default_experiment_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"exp_{now.strftime('%Y%m%d')}_step1_workflow_0_1"


def load_active_workflow(active_path: Path) -> dict[str, str]:
    if not active_path.exists():
        raise Step1RunnerError(f"ACTIVE_WORKFLOW not found: {active_path}")

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


def ensure_active_step1(project_root: Path) -> dict[str, str]:
    active = load_active_workflow(project_root / "Experiment" / "ACTIVE_WORKFLOW.md")
    workflow = active.get("active_workflow")
    stage = active.get("active_stage")
    if workflow != "workflow_0.1":
        raise Step1RunnerError(f"active_workflow must be workflow_0.1, got {workflow!r}")
    if stage != "Step-1":
        raise Step1RunnerError(f"active_stage must be Step-1, got {stage!r}")
    return active


def run_fetch_step1(project_root: Path, python_executable: str) -> dict[str, object]:
    command = [python_executable, "-u", "run_all.py", "--step", "1"]
    fetch_dir = project_root / "bigdata_challenge" / "data_fetcher"
    output_lines: list[str] = []

    process = subprocess.Popen(
        command,
        cwd=fetch_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    returncode = process.wait()
    return {"returncode": returncode, "stdout": "".join(output_lines), "stderr": ""}


def metrics_to_markdown(metrics: dict[str, object]) -> str:
    if not metrics:
        return "_not available_\n"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(metrics.items())) + "\n"


def write_report(
    report_path: Path,
    *,
    status: str,
    active: dict[str, str],
    experiment_dir: Path,
    output_dir: Path,
    raw_dir: Path,
    fetch_result: dict[str, object] | None,
    raw_metrics: dict[str, object] | None = None,
    output_metrics: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_stdout = "" if fetch_result is None else str(fetch_result.get("stdout", ""))
    fetch_stderr = "" if fetch_result is None else str(fetch_result.get("stderr", ""))
    fetch_returncode = "" if fetch_result is None else str(fetch_result.get("returncode", ""))
    content = f"""# Step-1 正式健康版运行报告

## Status

{status}

## Active Workflow

- `active_workflow`: {active.get("active_workflow", "")}
- `active_stage`: {active.get("active_stage", "")}
- `status`: {active.get("status", "")}

## Paths

- `experiment_dir`: {experiment_dir}
- `raw_dir`: {raw_dir}
- `output_dir`: {output_dir}

## Fetch

- `command`: /opt/miniconda3/bin/python3 -u run_all.py --step 1
- `returncode`: {fetch_returncode}

## Raw Metrics

{metrics_to_markdown(raw_metrics or {})}
## Output Metrics

{metrics_to_markdown(output_metrics or {})}
## Error

{error or "_none_"}

## Fetch Output

```text
{fetch_stdout}
{fetch_stderr}
```
"""
    report_path.write_text(content, encoding="utf-8")


def make_manifest_note(raw_metrics: dict[str, object]) -> str:
    return (
        "正式 Step-1 由 workflow_0.1/run_step1.py 联网执行 "
        "bigdata_challenge/data_fetcher/run_all.py --step 1 后生成；"
        f"latest_T={raw_metrics.get('daily_latest_T', '')}，"
        f"当前沪深300股票数={raw_metrics.get('hs300_count', '')}，"
        "Step-5 不属于本版正式健康链路。"
    )


def run_step1(
    *,
    project_root: Path = PROJECT_ROOT,
    experiment_name: str | None = None,
    python_executable: str = "/opt/miniconda3/bin/python3",
) -> dict[str, Path]:
    project_root = Path(project_root)
    active = ensure_active_step1(project_root)
    experiment_name = experiment_name or default_experiment_name()
    experiment_dir = project_root / "Experiment" / "workflow_0.1" / "experiments" / experiment_name
    output_dir = experiment_dir / "outputs" / "step1"
    notes_dir = experiment_dir / "notes"
    report_path = notes_dir / "step1_run_report.md"
    raw_dir = project_root / "bigdata_challenge" / "data" / "raw"

    fetch_result: dict[str, object] | None = None
    raw_metrics: dict[str, object] | None = None
    output_metrics: dict[str, object] | None = None

    try:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)

        print("Step-1 正式流程：联网更新 raw 数据")
        fetch_result = run_fetch_step1(project_root, python_executable)
        if int(fetch_result["returncode"]) != 0:
            raise Step1RunnerError("raw fetch failed")

        print("Step-1 正式流程：校验 raw 数据")
        raw_metrics = validate_raw_data(raw_dir)

        print("Step-1 正式流程：生成标准 CSV 输出")
        build_step1_outputs(raw_dir=raw_dir, output_dir=output_dir, note=make_manifest_note(raw_metrics))

        print("Step-1 正式流程：校验标准 CSV 输出")
        output_metrics = validate_outputs(output_dir)

        write_report(
            report_path,
            status="SUCCESS",
            active=active,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            raw_dir=raw_dir,
            fetch_result=fetch_result,
            raw_metrics=raw_metrics,
            output_metrics=output_metrics,
        )
    except (Step1RunnerError, Step1ValidationError, Exception) as exc:
        write_report(
            report_path,
            status="FAILED",
            active=active,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            raw_dir=raw_dir,
            fetch_result=fetch_result,
            raw_metrics=raw_metrics,
            output_metrics=output_metrics,
            error=str(exc),
        )
        if isinstance(exc, Step1RunnerError):
            raise
        raise Step1RunnerError(str(exc)) from exc

    print(f"Step-1 report: {report_path}")
    return {
        "experiment_dir": experiment_dir,
        "output_dir": output_dir,
        "report_path": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal workflow_0.1 Step-1 pipeline.")
    parser.add_argument("--experiment-name", default=None, help="实验目录名；默认 exp_YYYYMMDD_step1_workflow_0_1")
    parser.add_argument("--python-executable", default="/opt/miniconda3/bin/python3", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_step1(experiment_name=args.experiment_name, python_executable=args.python_executable)
    except Step1RunnerError as exc:
        print(f"Step-1 failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

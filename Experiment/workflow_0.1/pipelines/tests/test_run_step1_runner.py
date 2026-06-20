from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step1", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-1") -> None:
    active_path = project_root / "Experiment" / "ACTIVE_WORKFLOW.md"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_path.write_text(
        f"""# Active Workflow

```text
active_workflow: {workflow}
active_stage: {stage}
status: test
```
""",
        encoding="utf-8",
    )


def test_runner_success_calls_fetch_validates_outputs_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    calls: list[str] = []

    def fake_fetch(project_root: Path, python_executable: str):
        calls.append(f"fetch:{python_executable}")
        return {"returncode": 0, "stdout": "fetch ok", "stderr": ""}

    def fake_build(raw_dir: Path, output_dir: Path, note: str):
        calls.append("build")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {}

    monkeypatch.setattr(module, "run_fetch_step1", fake_fetch)
    monkeypatch.setattr(module, "build_step1_outputs", fake_build)
    monkeypatch.setattr(module, "validate_raw_data", lambda raw_dir: {"hs300_count": 300, "daily_latest_T": "2026-06-15"})
    monkeypatch.setattr(
        module,
        "validate_outputs",
        lambda output_dir: {
            "output_stock_count": 300,
            "output_latest_T": "2026-06-15",
            "output_unmatched_sector_count": 0,
        },
    )

    result = module.run_step1(project_root=tmp_path, experiment_name="exp_runner", python_executable="/bin/python")

    assert calls == ["fetch:/bin/python", "build"]
    assert result["experiment_dir"] == tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_runner"
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-1 正式健康版运行报告" in report
    assert "fetch ok" in report
    assert "output_stock_count: 300" in report


def test_runner_rejects_inactive_workflow(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, workflow="workflow_0.2")

    with pytest.raises(module.Step1RunnerError, match="active_workflow"):
        module.run_step1(project_root=tmp_path, experiment_name="exp_runner", python_executable="/bin/python")


def test_runner_fetch_failure_writes_report_and_exits_non_zero(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)

    monkeypatch.setattr(
        module,
        "run_fetch_step1",
        lambda project_root, python_executable: {"returncode": 1, "stdout": "partial", "stderr": "boom"},
    )

    with pytest.raises(module.Step1RunnerError, match="raw fetch failed"):
        module.run_step1(project_root=tmp_path, experiment_name="exp_runner", python_executable="/bin/python")

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_runner"
        / "notes"
        / "step1_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "boom" in report

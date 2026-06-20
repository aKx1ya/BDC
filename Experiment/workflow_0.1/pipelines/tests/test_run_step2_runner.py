from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step2", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-2") -> None:
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


def make_step1_dir(project_root: Path, name: str = "exp_step1") -> Path:
    step1_dir = project_root / "Experiment" / "workflow_0.1" / "experiments" / name
    (step1_dir / "outputs" / "step1").mkdir(parents=True, exist_ok=True)
    (step1_dir / "notes").mkdir(parents=True, exist_ok=True)
    (step1_dir / "notes" / "step1_run_report.md").write_text(
        """# Step-1 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return step1_dir


def test_runner_success_builds_validates_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step1_dir = make_step1_dir(tmp_path)
    calls: list[str] = []

    def fake_validate_step1_input(path: Path, expected_stock_count: int):
        calls.append(f"validate-input:{path.name}:{expected_stock_count}")
        return {"input_step1_latest_T": "2026-06-15", "input_step1_stock_count": expected_stock_count}

    def fake_build(step1_output_dir: Path, output_dir: Path, input_step1_experiment: str, note: str):
        calls.append(f"build:{input_step1_experiment}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {}

    def fake_validate_outputs(output_dir: Path, latest_t: str, expected_stock_count: int):
        calls.append(f"validate-output:{latest_t}:{expected_stock_count}")
        return {"output_latest_T": latest_t, "output_latest_t_rows": expected_stock_count}

    monkeypatch.setattr(module, "validate_step1_input", fake_validate_step1_input)
    monkeypatch.setattr(module, "build_step2_outputs", fake_build)
    monkeypatch.setattr(module, "validate_outputs", fake_validate_outputs)

    result = module.run_step2(
        project_root=tmp_path,
        step1_experiment=step1_dir.name,
        experiment_name="exp_step2",
        expected_stock_count=3,
    )

    assert calls == [
        "validate-input:exp_step1:3",
        "build:exp_step1",
        "validate-output:2026-06-15:3",
    ]
    assert result["experiment_dir"] == tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step2"
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-2 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_latest_t_rows: 3" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-1")

    with pytest.raises(module.Step2RunnerError, match="active_stage"):
        module.run_step2(project_root=tmp_path, step1_experiment="exp_step1", experiment_name="exp_step2")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step1_dir = make_step1_dir(tmp_path)

    def fake_validate_step1_input(path: Path, expected_stock_count: int):
        raise module.Step2ValidationError("input is unhealthy")

    monkeypatch.setattr(module, "validate_step1_input", fake_validate_step1_input)

    with pytest.raises(module.Step2RunnerError, match="input is unhealthy"):
        module.run_step2(
            project_root=tmp_path,
            step1_experiment=step1_dir.name,
            experiment_name="exp_step2",
            expected_stock_count=3,
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step2"
        / "notes"
        / "step2_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input is unhealthy" in report

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step3.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step3", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-3") -> None:
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


def make_step2_dir(project_root: Path, name: str = "exp_step2") -> Path:
    step2_dir = project_root / "Experiment" / "workflow_0.1" / "experiments" / name
    (step2_dir / "outputs" / "step2").mkdir(parents=True, exist_ok=True)
    (step2_dir / "notes").mkdir(parents=True, exist_ok=True)
    (step2_dir / "notes" / "step2_run_report.md").write_text(
        """# Step-2 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return step2_dir


def test_runner_success_builds_validates_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir = make_step2_dir(tmp_path)
    calls: list[str] = []

    def fake_validate_step2_input(path: Path, expected_stock_count: int):
        calls.append(f"validate-input:{path.name}:{expected_stock_count}")
        return {
            "input_step2_latest_T": "2026-06-15",
            "last_labelable_T": "2026-06-08",
            "input_step2_stock_count": expected_stock_count,
        }

    def fake_build(step2_output_dir: Path, output_dir: Path, input_step2_experiment: str, note: str):
        calls.append(f"build:{input_step2_experiment}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {}

    def fake_validate_outputs(output_dir: Path, input_step2_latest_t: str, last_labelable_t: str, expected_stock_count: int):
        calls.append(f"validate-output:{input_step2_latest_t}:{last_labelable_t}:{expected_stock_count}")
        return {"output_sample_rows": 123, "output_sample_date_end": last_labelable_t}

    monkeypatch.setattr(module, "validate_step2_input", fake_validate_step2_input)
    monkeypatch.setattr(module, "build_step3_outputs", fake_build)
    monkeypatch.setattr(module, "validate_outputs", fake_validate_outputs)

    result = module.run_step3(
        project_root=tmp_path,
        step2_experiment=step2_dir.name,
        experiment_name="exp_step3",
        expected_stock_count=3,
    )

    assert calls == [
        "validate-input:exp_step2:3",
        "build:exp_step2",
        "validate-output:2026-06-15:2026-06-08:3",
    ]
    assert result["experiment_dir"] == tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step3"
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-3 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_sample_rows: 123" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-2")

    with pytest.raises(module.Step3RunnerError, match="active_stage"):
        module.run_step3(project_root=tmp_path, step2_experiment="exp_step2", experiment_name="exp_step3")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir = make_step2_dir(tmp_path)

    def fake_validate_step2_input(path: Path, expected_stock_count: int):
        raise module.Step3ValidationError("input is unhealthy")

    monkeypatch.setattr(module, "validate_step2_input", fake_validate_step2_input)

    with pytest.raises(module.Step3RunnerError, match="input is unhealthy"):
        module.run_step3(
            project_root=tmp_path,
            step2_experiment=step2_dir.name,
            experiment_name="exp_step3",
            expected_stock_count=3,
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step3"
        / "notes"
        / "step3_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input is unhealthy" in report

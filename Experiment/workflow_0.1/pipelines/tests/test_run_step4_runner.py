from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step4.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step4", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-4") -> None:
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


def make_step3_dir(project_root: Path, name: str = "exp_step3") -> Path:
    step3_dir = project_root / "Experiment" / "workflow_0.1" / "experiments" / name
    (step3_dir / "outputs" / "step3").mkdir(parents=True, exist_ok=True)
    (step3_dir / "notes").mkdir(parents=True, exist_ok=True)
    (step3_dir / "notes" / "step3_run_report.md").write_text(
        """# Step-3 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return step3_dir


def test_runner_success_builds_validates_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step3_dir = make_step3_dir(tmp_path)
    calls: list[str] = []

    def fake_validate_step3_input(path: Path):
        calls.append(f"validate-input:{path.name}")
        return {
            "input_step3_sample_date_start": "2023-04-03",
            "input_step3_sample_date_end": "2026-06-08",
            "input_step3_sample_date_count": 769,
            "input_step3_sample_rows": 228357,
        }

    def fake_build(
        step3_output_dir: Path,
        output_dir: Path,
        input_step3_experiment: str,
        train_window: int,
        gap_days: int,
        eval_days: int,
        walk_forward_step: int,
        train_ratio: float,
        final_test_days: int,
        note: str,
    ):
        calls.append(f"build:{input_step3_experiment}:{train_window}:{gap_days}:{eval_days}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {}

    def fake_validate_outputs(
        output_dir: Path,
        input_step3_metrics: dict[str, object],
        train_window: int,
        gap_days: int,
        eval_days: int,
        walk_forward_step: int,
        train_ratio: float,
        final_test_days: int,
    ):
        calls.append(f"validate-output:{input_step3_metrics['input_step3_sample_date_count']}:{final_test_days}")
        return {"output_walk_forward_rounds": 102, "output_final_test_dates": final_test_days}

    monkeypatch.setattr(module, "validate_step3_input", fake_validate_step3_input)
    monkeypatch.setattr(module, "build_step4_outputs", fake_build)
    monkeypatch.setattr(module, "validate_outputs", fake_validate_outputs)

    result = module.run_step4(
        project_root=tmp_path,
        step3_experiment=step3_dir.name,
        experiment_name="exp_step4",
        train_window=10,
        gap_days=2,
        eval_days=3,
        walk_forward_step=3,
        train_ratio=0.75,
        final_test_days=4,
    )

    assert calls == [
        "validate-input:exp_step3",
        "build:exp_step3:10:2:3",
        "validate-output:769:4",
    ]
    assert result["experiment_dir"] == tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step4"
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-4 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_walk_forward_rounds: 102" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-3")

    with pytest.raises(module.Step4RunnerError, match="active_stage"):
        module.run_step4(project_root=tmp_path, step3_experiment="exp_step3", experiment_name="exp_step4")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step3_dir = make_step3_dir(tmp_path)

    def fake_validate_step3_input(path: Path):
        raise module.Step4ValidationError("input step3 is unhealthy")

    monkeypatch.setattr(module, "validate_step3_input", fake_validate_step3_input)

    with pytest.raises(module.Step4RunnerError, match="input step3 is unhealthy"):
        module.run_step4(
            project_root=tmp_path,
            step3_experiment=step3_dir.name,
            experiment_name="exp_step4",
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step4"
        / "notes"
        / "step4_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input step3 is unhealthy" in report

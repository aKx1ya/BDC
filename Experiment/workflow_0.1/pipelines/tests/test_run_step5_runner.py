from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step5.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step5", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-5") -> None:
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


def write_report(experiment_dir: Path, step_name: str) -> None:
    report = experiment_dir / "notes" / f"{step_name}_run_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        """# report

## Status

SUCCESS
""",
        encoding="utf-8",
    )


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["项目", "说明"]).to_csv(path, index=False)


def make_input_dirs(project_root: Path) -> tuple[Path, Path, Path]:
    workflow_root = project_root / "Experiment" / "workflow_0.1"
    step2_dir = workflow_root / "experiments" / "exp_step2"
    step3_dir = workflow_root / "experiments" / "exp_step3"
    step4_dir = workflow_root / "experiments" / "exp_step4"
    write_report(step2_dir, "step2")
    write_report(step3_dir, "step3")
    write_report(step4_dir, "step4")
    write_manifest(
        step3_dir / "outputs" / "step3" / "step3_sample_manifest.csv",
        [{"项目": "input_step2_experiment", "说明": step2_dir.name}],
    )
    write_manifest(
        step4_dir / "outputs" / "step4" / "step4_split_manifest.csv",
        [{"项目": "input_step3_experiment", "说明": step3_dir.name}],
    )
    return step2_dir, step3_dir, step4_dir


def test_runner_success_builds_validates_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir, step3_dir, step4_dir = make_input_dirs(tmp_path)
    calls: list[str] = []

    def fake_validate_inputs(s2: Path, s3: Path, s4: Path):
        calls.append(f"validate-inputs:{s2.name}:{s3.name}:{s4.name}")
        return {
            "input_step2_experiment": s2.name,
            "input_step3_experiment": s3.name,
            "input_step4_experiment": s4.name,
            "input_step2_latest_T": "2026-06-15",
        }

    def fake_build(
        step2_output_dir: Path,
        step3_output_dir: Path,
        step4_output_dir: Path,
        output_dir: Path,
        model_dir: Path,
        input_step2_experiment: str,
        input_step3_experiment: str,
        input_step4_experiment: str,
        candidate_size: int,
        random_seed: int,
        note: str,
    ):
        calls.append(f"build:{input_step2_experiment}:{input_step3_experiment}:{input_step4_experiment}:{candidate_size}")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        return {}

    def fake_validate_outputs(
        output_dir: Path,
        model_dir: Path,
        step4_output_dir: Path,
        input_metrics: dict[str, object],
        candidate_size: int,
        random_seed: int,
    ):
        calls.append(f"validate-output:{candidate_size}:{random_seed}")
        return {"output_candidate_rows": candidate_size, "output_walk_forward_rounds_used": 8}

    monkeypatch.setattr(module, "validate_inputs", fake_validate_inputs)
    monkeypatch.setattr(module, "build_step5_outputs", fake_build)
    monkeypatch.setattr(module, "validate_outputs", fake_validate_outputs)

    result = module.run_step5(
        project_root=tmp_path,
        step2_experiment=step2_dir.name,
        step3_experiment=step3_dir.name,
        step4_experiment=step4_dir.name,
        experiment_name="exp_step5",
        candidate_size=3,
        random_seed=7,
    )

    assert calls == [
        "validate-inputs:exp_step2:exp_step3:exp_step4",
        "build:exp_step2:exp_step3:exp_step4:3",
        "validate-output:3:7",
    ]
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-5 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_candidate_rows: 3" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-4")

    with pytest.raises(module.Step5RunnerError, match="active_stage"):
        module.run_step5(project_root=tmp_path, experiment_name="exp_step5")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir, step3_dir, step4_dir = make_input_dirs(tmp_path)

    def fake_validate_inputs(s2: Path, s3: Path, s4: Path):
        raise module.Step5ValidationError("input chain is unhealthy")

    monkeypatch.setattr(module, "validate_inputs", fake_validate_inputs)

    with pytest.raises(module.Step5RunnerError, match="input chain is unhealthy"):
        module.run_step5(
            project_root=tmp_path,
            step2_experiment=step2_dir.name,
            step3_experiment=step3_dir.name,
            step4_experiment=step4_dir.name,
            experiment_name="exp_step5",
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step5"
        / "notes"
        / "step5_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input chain is unhealthy" in report

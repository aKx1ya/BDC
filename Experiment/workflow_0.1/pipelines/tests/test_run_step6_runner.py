from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from test_build_step6_outputs import make_step6_input_chain


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step6.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step6", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-6") -> None:
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


def test_runner_success_generates_result_and_report(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir, step5_dir, _ = make_step6_input_chain(tmp_path)

    result = module.run_step6(
        project_root=tmp_path,
        step5_experiment=step5_dir.name,
        step2_experiment=step2_dir.name,
        experiment_name="exp_step6",
    )

    output_dir = result["output_dir"]
    assert (output_dir / "step6_result.csv").exists()
    assert (output_dir / "step6_ranking_log.csv").exists()
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-6 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_selected_count" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-5")

    with pytest.raises(module.Step6RunnerError, match="active_stage"):
        module.run_step6(project_root=tmp_path, experiment_name="exp_step6")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step2_dir, step5_dir, _ = make_step6_input_chain(tmp_path)

    def fake_validate_inputs(s5: Path, s2: Path):
        raise module.Step6ValidationError("input chain is unhealthy")

    monkeypatch.setattr(module, "validate_inputs", fake_validate_inputs)

    with pytest.raises(module.Step6RunnerError, match="input chain is unhealthy"):
        module.run_step6(
            project_root=tmp_path,
            step5_experiment=step5_dir.name,
            step2_experiment=step2_dir.name,
            experiment_name="exp_step6",
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step6"
        / "notes"
        / "step6_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input chain is unhealthy" in report

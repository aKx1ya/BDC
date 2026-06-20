from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from test_build_step7_outputs import make_step7_input_chain


RUNNER_PATH = Path(__file__).resolve().parents[2] / "run_step7.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_step7", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_active_workflow(project_root: Path, workflow: str = "workflow_0.1", stage: str = "Step-7") -> None:
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


def test_runner_success_freeze_only_writes_report(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step6_dir = make_step7_input_chain(tmp_path)

    result = module.run_step7(
        project_root=tmp_path,
        step6_experiment=step6_dir.name,
        experiment_name="exp_step7",
        mode="freeze-only",
    )

    assert (result["output_dir"] / "step7_frozen_result.csv").exists()
    report = result["report_path"].read_text(encoding="utf-8")
    assert "Step-7 正式健康版运行报告" in report
    assert "SUCCESS" in report
    assert "output_result_status: FREEZE_ONLY_SUCCESS" in report


def test_runner_rejects_wrong_stage(tmp_path: Path) -> None:
    module = load_module()
    write_active_workflow(tmp_path, stage="Step-6")

    with pytest.raises(module.Step7RunnerError, match="active_stage"):
        module.run_step7(project_root=tmp_path, experiment_name="exp_step7", mode="freeze-only")


def test_runner_failure_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    write_active_workflow(tmp_path)
    step6_dir = make_step7_input_chain(tmp_path)

    def fake_validate_inputs(s6: Path):
        raise module.Step7ValidationError("input chain is unhealthy")

    monkeypatch.setattr(module, "validate_inputs", fake_validate_inputs)

    with pytest.raises(module.Step7RunnerError, match="input chain is unhealthy"):
        module.run_step7(
            project_root=tmp_path,
            step6_experiment=step6_dir.name,
            experiment_name="exp_step7",
            mode="freeze-only",
        )

    report_path = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step7"
        / "notes"
        / "step7_run_report.md"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "FAILED" in report
    assert "input chain is unhealthy" in report

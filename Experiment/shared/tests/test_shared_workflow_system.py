from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SHARED_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_fixture_workflow(tmp_path: Path, *, stage: str = "Step-3") -> tuple[Path, Path, Path]:
    experiment_root = tmp_path / "Experiment"
    workflow_dir = experiment_root / "workflow_9.9"
    workflow_dir.mkdir(parents=True)
    runner_path = workflow_dir / "run_step3.py"
    runner_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    (workflow_dir / "strategy").mkdir()
    (workflow_dir / "docs").mkdir()
    (workflow_dir / "pipelines").mkdir()
    (workflow_dir / "experiments").mkdir()
    (workflow_dir / "workflow_config.yaml").write_text(
        """workflow_id: workflow_9.9
schema_version: workflow_9.9_csv_v1
health_system_version: workflow_health_v1
paths:
  strategy_dir: strategy
  docs_dir: docs
  pipelines_dir: pipelines
  experiments_dir: experiments
steps:
  step3:
    stage: Step-3
    runner: run_step3.py
    default_args:
      - --alpha
      - "1"
""",
        encoding="utf-8",
    )
    active_path = experiment_root / "ACTIVE_WORKFLOW.md"
    active_path.write_text(
        f"""# Active Workflow

```text
active_workflow: workflow_9.9
active_stage: {stage}
status: fixture
```
""",
        encoding="utf-8",
    )
    return experiment_root, workflow_dir, active_path


def load_fixture_context(tmp_path: Path, *, stage: str = "Step-3"):
    context_module = load_module("workflow_context", SHARED_ROOT / "workflow_context.py")
    experiment_root, workflow_dir, active_path = write_fixture_workflow(tmp_path, stage=stage)
    context = context_module.load_context(
        active_path=active_path,
        experiment_root=experiment_root,
        project_root=tmp_path,
    )
    return context_module, context, workflow_dir


def test_context_resolves_active_workflow_config_and_runner(tmp_path: Path) -> None:
    context_module, context, workflow_dir = load_fixture_context(tmp_path)

    assert context.workflow_id == "workflow_9.9"
    assert context.active_stage == "Step-3"
    assert context.config_path == workflow_dir / "workflow_config.yaml"
    assert context_module.configured_step_numbers(context) == [3]
    assert context_module.runner_path_for_step(context, 3) == workflow_dir / "run_step3.py"


def test_shared_runner_builds_command_from_workflow_config(tmp_path: Path, monkeypatch) -> None:
    context_module, context, workflow_dir = load_fixture_context(tmp_path)
    run_step = load_module("shared_run_step_for_test", SHARED_ROOT / "runners" / "run_step.py")
    monkeypatch.setattr(run_step, "load_context", lambda workflow_id=None: context)

    args = Namespace(
        step="3",
        workflow=None,
        experiment_name="exp_shared_fixture",
        mode=None,
        print_context=False,
        allow_stage_mismatch=False,
        extra_args=["--", "--beta", "2"],
    )

    command, env = run_step.build_command(args)

    assert command == [
        sys.executable,
        str(workflow_dir / "run_step3.py"),
        "--alpha",
        "1",
        "--experiment-name",
        "exp_shared_fixture",
        "--beta",
        "2",
    ]
    assert env["EXPERIMENT_ACTIVE_WORKFLOW"] == "workflow_9.9"
    assert env["EXPERIMENT_ACTIVE_STAGE"] == "Step-3"
    assert env["EXPERIMENT_WORKFLOW_CONFIG"] == str(context.config_path)


def test_shared_runner_rejects_stage_mismatch(tmp_path: Path, monkeypatch) -> None:
    _, context, _ = load_fixture_context(tmp_path, stage="Step-2")
    run_step = load_module("shared_run_step_stage_mismatch_test", SHARED_ROOT / "runners" / "run_step.py")
    monkeypatch.setattr(run_step, "load_context", lambda workflow_id=None: context)

    args = Namespace(
        step="3",
        workflow=None,
        experiment_name=None,
        mode=None,
        print_context=False,
        allow_stage_mismatch=False,
        extra_args=[],
    )

    with pytest.raises(run_step.SharedRunnerError, match="active_stage must be Step-3"):
        run_step.build_command(args)


def test_workflow_config_validator_accepts_fixture(tmp_path: Path) -> None:
    _, context, _ = load_fixture_context(tmp_path)
    validator = load_module(
        "validate_workflow_config_for_test",
        SHARED_ROOT / "validators" / "validate_workflow_config.py",
    )

    result = validator.validate_workflow_config(context)

    assert result["workflow_id"] == "workflow_9.9"
    assert result["configured_steps"] == [3]
    assert result["runner_count"] == 1


def test_workflow_config_validator_rejects_missing_runner(tmp_path: Path) -> None:
    _, context, workflow_dir = load_fixture_context(tmp_path)
    (workflow_dir / "run_step3.py").unlink()
    validator = load_module(
        "validate_workflow_config_missing_runner_test",
        SHARED_ROOT / "validators" / "validate_workflow_config.py",
    )

    with pytest.raises(validator.WorkflowConfigValidationError, match="runner not found"):
        validator.validate_workflow_config(context)

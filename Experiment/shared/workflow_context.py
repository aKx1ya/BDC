#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
ACTIVE_WORKFLOW_PATH = EXPERIMENT_ROOT / "ACTIVE_WORKFLOW.md"


class WorkflowContextError(Exception):
    """Raised when active workflow or workflow config cannot be resolved."""


@dataclass(frozen=True)
class WorkflowContext:
    project_root: Path
    experiment_root: Path
    workflow_id: str
    active_stage: str
    active_status: str
    workflow_dir: Path
    config_path: Path
    config: dict[str, Any]


def parse_active_workflow(active_path: Path = ACTIVE_WORKFLOW_PATH) -> dict[str, str]:
    if not active_path.exists():
        raise WorkflowContextError(f"ACTIVE_WORKFLOW not found: {active_path}")

    values: dict[str, str] = {}
    for line in active_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key.startswith("active_") or key == "status":
            values[key] = value.strip()

    missing = [key for key in ["active_workflow", "active_stage"] if not values.get(key)]
    if missing:
        raise WorkflowContextError(f"ACTIVE_WORKFLOW missing keys: {missing}")
    values.setdefault("status", "")
    return values


def normalize_step(value: str | int) -> tuple[int, str, str]:
    text = str(value).strip()
    if text.lower().startswith("step-"):
        number_text = text.split("-", 1)[1]
    elif text.lower().startswith("step"):
        number_text = text[4:]
    else:
        number_text = text
    try:
        number = int(number_text)
    except ValueError as exc:
        raise WorkflowContextError(f"invalid step value: {value!r}") from exc
    if number < 1:
        raise WorkflowContextError(f"step number must be positive, got {number}")
    return number, f"Step-{number}", f"step{number}"


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise WorkflowContextError(f"workflow_config.yaml not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkflowContextError(f"workflow_config.yaml must contain a mapping: {config_path}")
    return data


def resolve_workflow_dir(experiment_root: Path, workflow_id: str) -> Path:
    workflow_dir = experiment_root / workflow_id
    if not workflow_dir.exists():
        raise WorkflowContextError(f"workflow directory not found: {workflow_dir}")
    return workflow_dir


def load_context(
    *,
    workflow_id: str | None = None,
    active_path: Path = ACTIVE_WORKFLOW_PATH,
    experiment_root: Path = EXPERIMENT_ROOT,
    project_root: Path = PROJECT_ROOT,
) -> WorkflowContext:
    active = parse_active_workflow(active_path)
    selected_workflow = workflow_id or active["active_workflow"]
    workflow_dir = resolve_workflow_dir(experiment_root, selected_workflow)
    config_path = workflow_dir / "workflow_config.yaml"
    config = load_yaml_config(config_path)

    config_workflow = str(config.get("workflow_id", "")).strip()
    if config_workflow and config_workflow != selected_workflow:
        raise WorkflowContextError(
            f"workflow_config workflow_id={config_workflow!r} does not match selected workflow {selected_workflow!r}"
        )

    return WorkflowContext(
        project_root=project_root,
        experiment_root=experiment_root,
        workflow_id=selected_workflow,
        active_stage=active["active_stage"],
        active_status=active.get("status", ""),
        workflow_dir=workflow_dir,
        config_path=config_path,
        config=config,
    )


def step_config(context: WorkflowContext, step: str | int) -> dict[str, Any]:
    _, _, step_key = normalize_step(step)
    steps = context.config.get("steps")
    if not isinstance(steps, dict):
        raise WorkflowContextError(f"{context.config_path} missing steps mapping")
    value = steps.get(step_key)
    if not isinstance(value, dict):
        raise WorkflowContextError(f"{context.config_path} missing config for {step_key}")
    return value


def resolve_path(context: WorkflowContext, value: str | Path) -> Path:
    path_text = str(value)
    path_text = path_text.replace("{project_root}", str(context.project_root))
    path_text = path_text.replace("{experiment_root}", str(context.experiment_root))
    path_text = path_text.replace("{workflow_dir}", str(context.workflow_dir))
    path = Path(path_text)
    if path.is_absolute():
        return path
    return context.workflow_dir / path


def runner_path_for_step(context: WorkflowContext, step: str | int) -> Path:
    config = step_config(context, step)
    runner = config.get("runner")
    if not runner:
        _, _, step_key = normalize_step(step)
        raise WorkflowContextError(f"{context.config_path} {step_key} missing runner")
    path = resolve_path(context, str(runner))
    if not path.exists():
        raise WorkflowContextError(f"runner not found: {path}")
    return path


def configured_step_numbers(context: WorkflowContext) -> list[int]:
    steps = context.config.get("steps", {})
    if not isinstance(steps, dict):
        return []
    numbers: list[int] = []
    for key in steps:
        try:
            number, _, _ = normalize_step(str(key))
        except WorkflowContextError:
            continue
        numbers.append(number)
    return sorted(set(numbers))

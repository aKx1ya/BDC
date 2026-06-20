#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SHARED_ROOT = Path(__file__).resolve().parents[1]
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from workflow_context import (  # noqa: E402
    WorkflowContext,
    WorkflowContextError,
    configured_step_numbers,
    load_context,
    normalize_step,
    resolve_path,
    runner_path_for_step,
    step_config,
)


class WorkflowConfigValidationError(Exception):
    """Raised when a workflow config is not safe to use through shared runners."""


def require_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowConfigValidationError(f"{name} must be a mapping")
    return value


def validate_declared_paths(context: WorkflowContext) -> list[str]:
    paths = require_mapping(context.config.get("paths", {}), name="paths")
    checked: list[str] = []
    for key, value in paths.items():
        if value is None:
            continue
        path = resolve_path(context, str(value))
        if not path.exists():
            raise WorkflowConfigValidationError(f"paths.{key} does not exist: {path}")
        if not path.is_dir():
            raise WorkflowConfigValidationError(f"paths.{key} must be a directory: {path}")
        checked.append(key)
    return sorted(checked)


def validate_strategy_sources(context: WorkflowContext) -> list[str]:
    sources = context.config.get("strategy_sources", [])
    if sources is None:
        return []
    if not isinstance(sources, list):
        raise WorkflowConfigValidationError("strategy_sources must be a list")

    checked: list[str] = []
    for source in sources:
        path = resolve_path(context, str(source))
        if not path.exists():
            raise WorkflowConfigValidationError(f"strategy source not found: {path}")
        checked.append(str(source))
    return checked


def validate_step(context: WorkflowContext, step_number: int) -> dict[str, object]:
    expected_number, expected_stage, step_key = normalize_step(step_number)
    config = step_config(context, expected_number)

    stage = str(config.get("stage", "")).strip()
    if stage and stage != expected_stage:
        raise WorkflowConfigValidationError(f"{step_key}.stage must be {expected_stage}, got {stage!r}")

    default_args = config.get("default_args", [])
    if default_args is not None and not isinstance(default_args, list):
        raise WorkflowConfigValidationError(f"{step_key}.default_args must be a list")

    output_dir_name = str(config.get("output_dir_name", "")).strip()
    if output_dir_name and output_dir_name != step_key:
        raise WorkflowConfigValidationError(
            f"{step_key}.output_dir_name should be {step_key!r} for cross-workflow consistency, got {output_dir_name!r}"
        )

    report_name = str(config.get("report_name", "")).strip()
    expected_report = f"{step_key}_run_report.md"
    if report_name and report_name != expected_report:
        raise WorkflowConfigValidationError(f"{step_key}.report_name must be {expected_report}, got {report_name!r}")

    try:
        runner_path = runner_path_for_step(context, expected_number)
    except WorkflowContextError as exc:
        raise WorkflowConfigValidationError(str(exc)) from exc

    health_doc = config.get("health_doc")
    if health_doc:
        health_doc_path = resolve_path(context, str(health_doc))
        if not health_doc_path.exists():
            raise WorkflowConfigValidationError(f"{step_key}.health_doc not found: {health_doc_path}")

    policy_sources = config.get("policy_source", [])
    if policy_sources is None:
        policy_sources = []
    if not isinstance(policy_sources, list):
        raise WorkflowConfigValidationError(f"{step_key}.policy_source must be a list")
    for source in policy_sources:
        source_path = resolve_path(context, str(source))
        if not source_path.exists():
            raise WorkflowConfigValidationError(f"{step_key}.policy_source not found: {source_path}")

    return {
        "step": expected_number,
        "stage": expected_stage,
        "runner": str(runner_path),
        "default_arg_count": len(default_args or []),
    }


def validate_workflow_config(context: WorkflowContext) -> dict[str, object]:
    workflow_id = str(context.config.get("workflow_id", "")).strip()
    if not workflow_id:
        raise WorkflowConfigValidationError("workflow_id is required")
    if workflow_id != context.workflow_id:
        raise WorkflowConfigValidationError(
            f"workflow_id must match active context {context.workflow_id!r}, got {workflow_id!r}"
        )

    schema_version = str(context.config.get("schema_version", "")).strip()
    if not schema_version:
        raise WorkflowConfigValidationError("schema_version is required")

    require_mapping(context.config.get("steps"), name="steps")
    configured_steps = configured_step_numbers(context)
    if not configured_steps:
        raise WorkflowConfigValidationError("steps must contain at least one step")

    declared_paths = validate_declared_paths(context)
    strategy_sources = validate_strategy_sources(context)
    step_results = [validate_step(context, step_number) for step_number in configured_steps]

    return {
        "workflow_id": workflow_id,
        "schema_version": schema_version,
        "health_system_version": str(context.config.get("health_system_version", "")).strip(),
        "configured_steps": configured_steps,
        "runner_count": len(step_results),
        "declared_paths": declared_paths,
        "strategy_source_count": len(strategy_sources),
        "config_path": str(context.config_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a workflow_config.yaml for shared runner compatibility.")
    parser.add_argument("--workflow", default=None, help="Override active_workflow; default reads Experiment/ACTIVE_WORKFLOW.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        context = load_context(workflow_id=args.workflow)
        result = validate_workflow_config(context)
    except (WorkflowContextError, WorkflowConfigValidationError) as exc:
        print(f"workflow config validation failed: {exc}")
        return 1

    print("workflow config validation passed")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SHARED_ROOT = Path(__file__).resolve().parents[1]
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from workflow_context import (  # noqa: E402
    WorkflowContextError,
    configured_step_numbers,
    load_context,
    normalize_step,
    runner_path_for_step,
    step_config,
)


class SharedRunnerError(Exception):
    """Generic shared runner failed before delegating to a workflow runner."""


def split_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    context = load_context(workflow_id=args.workflow)
    step_number, active_stage, step_key = normalize_step(args.step)

    if not args.allow_stage_mismatch and context.active_stage != active_stage:
        raise SharedRunnerError(
            f"ACTIVE_WORKFLOW active_stage must be {active_stage} to run {step_key}, got {context.active_stage!r}. "
            "Use --allow-stage-mismatch only for debugging."
        )

    config = step_config(context, step_number)
    runner_path = runner_path_for_step(context, step_number)
    command = [sys.executable, str(runner_path)]

    default_args = config.get("default_args", [])
    if default_args is None:
        default_args = []
    if not isinstance(default_args, list):
        raise SharedRunnerError(f"{context.config_path} {step_key}.default_args must be a list")
    command.extend(str(item) for item in default_args)

    if args.experiment_name:
        command.extend(["--experiment-name", args.experiment_name])
    if args.mode:
        command.extend(["--mode", args.mode])
    command.extend(split_extra_args(args.extra_args))

    env = os.environ.copy()
    env["EXPERIMENT_ACTIVE_WORKFLOW"] = context.workflow_id
    env["EXPERIMENT_ACTIVE_STAGE"] = active_stage
    env["EXPERIMENT_WORKFLOW_CONFIG"] = str(context.config_path)
    env["EXPERIMENT_WORKFLOW_DIR"] = str(context.workflow_dir)

    if args.print_context:
        print(f"active_workflow: {context.workflow_id}")
        print(f"active_stage: {context.active_stage}")
        print(f"active_status: {context.active_status}")
        print(f"workflow_dir: {context.workflow_dir}")
        print(f"config_path: {context.config_path}")
        print(f"configured_steps: {configured_step_numbers(context)}")
    print("shared runner command:")
    print(" ".join(command))
    return command, env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the active workflow Step through the shared migration-aware dispatcher."
    )
    parser.add_argument("--step", required=True, help="Step number or stage, for example 1 or Step-7")
    parser.add_argument("--workflow", default=None, help="Override active_workflow; default reads Experiment/ACTIVE_WORKFLOW.md")
    parser.add_argument("--experiment-name", default=None, help="Pass through to workflow runner when supported")
    parser.add_argument("--mode", default=None, help="Pass through to workflow runner, mainly Step-7 freeze-only/local-score")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command without executing it")
    parser.add_argument("--print-context", action="store_true", help="Print active workflow and config context")
    parser.add_argument(
        "--allow-stage-mismatch",
        action="store_true",
        help="Allow running a Step that does not match ACTIVE_WORKFLOW active_stage; intended for debugging only",
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional args after -- are passed to the workflow runner")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        command, env = build_command(args)
    except (WorkflowContextError, SharedRunnerError) as exc:
        print(f"shared runner failed: {exc}")
        return 1

    if args.dry_run:
        return 0

    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

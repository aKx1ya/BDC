from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step7_outputs import OFFICIAL_SCRIPT, make_step7_input_chain, make_test_data


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step7_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step7.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_good_step7(tmp_path: Path, *, mode: str = "local-score") -> tuple[object, Path, Path]:
    build_module = load_module("build_step7_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step7", VALIDATE_MODULE_PATH)
    step6_dir = make_step7_input_chain(tmp_path)
    output_dir = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step7_fixture"
        / "outputs"
        / "step7"
    )
    workspace_dir = output_dir.parents[1] / "official_scoring_workspace"
    test_path = make_test_data(tmp_path / "test.csv", start="2026-06-16")
    build_module.build_step7_outputs(
        step6_output_dir=step6_dir / "outputs" / "step6",
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        mode=mode,
        input_step6_experiment=step6_dir.name,
        official_script_path=OFFICIAL_SCRIPT,
        test_data_path=test_path,
    )
    return validate_module, step6_dir, output_dir


def test_validate_step7_accepts_good_local_score(tmp_path: Path) -> None:
    module, step6_dir, output_dir = build_good_step7(tmp_path, mode="local-score")

    metrics = module.validate_step7(step6_dir, output_dir, mode="local-score")

    assert metrics["input_step6_experiment"] == "exp_step6_fixture"
    assert metrics["output_score_mode"] == "local-score"
    assert metrics["output_result_status"] == "SCORE_SUCCESS"
    assert metrics["output_selected_count"] == 2


def test_validate_step7_accepts_good_freeze_only(tmp_path: Path) -> None:
    module, step6_dir, output_dir = build_good_step7(tmp_path, mode="freeze-only")

    metrics = module.validate_step7(step6_dir, output_dir, mode="freeze-only")

    assert metrics["output_score_mode"] == "freeze-only"
    assert metrics["output_result_status"] == "FREEZE_ONLY_SUCCESS"
    assert metrics["output_final_score"] == ""


def test_validate_step7_rejects_failed_step6_report(tmp_path: Path) -> None:
    module, step6_dir, output_dir = build_good_step7(tmp_path)
    (step6_dir / "notes" / "step6_run_report.md").write_text(
        """# Step-6 report

## Status

FAILED
""",
        encoding="utf-8",
    )

    with pytest.raises(module.Step7ValidationError, match="Step-6 report"):
        module.validate_step7(step6_dir, output_dir, mode="local-score")


def test_validate_step7_rejects_frozen_result_mismatch(tmp_path: Path) -> None:
    module, step6_dir, output_dir = build_good_step7(tmp_path)
    path = output_dir / "step7_frozen_result.csv"
    frozen = pd.read_csv(path, dtype={"stock_id": str})
    frozen.loc[0, "weight"] = 0.1
    frozen.to_csv(path, index=False)

    with pytest.raises(module.Step7ValidationError, match="must exactly match Step-6 result"):
        module.validate_step7(step6_dir, output_dir, mode="local-score")


def test_validate_step7_rejects_non_pass_leakage_check(tmp_path: Path) -> None:
    module, step6_dir, output_dir = build_good_step7(tmp_path)
    path = output_dir / "step7_leakage_check.csv"
    leakage = pd.read_csv(path)
    leakage.loc[0, "状态"] = "FAIL"
    leakage.to_csv(path, index=False)

    with pytest.raises(module.Step7ValidationError, match="non-PASS"):
        module.validate_step7(step6_dir, output_dir, mode="local-score")


def test_validate_step7_rejects_failed_local_score_outputs(tmp_path: Path) -> None:
    build_module = load_module("build_step7_outputs_failed_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step7_failed_case", VALIDATE_MODULE_PATH)
    step6_dir = make_step7_input_chain(tmp_path, candidate_date="2026-06-15")
    output_dir = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step7_fixture"
        / "outputs"
        / "step7"
    )
    test_path = make_test_data(tmp_path / "test.csv", start="2026-06-01")
    build_module.build_step7_outputs(
        step6_output_dir=step6_dir / "outputs" / "step6",
        output_dir=output_dir,
        workspace_dir=tmp_path / "workspace",
        mode="local-score",
        input_step6_experiment=step6_dir.name,
        official_script_path=OFFICIAL_SCRIPT,
        test_data_path=test_path,
    )

    with pytest.raises(validate_module.Step7ValidationError, match="result_status"):
        validate_module.validate_step7(step6_dir, output_dir, mode="local-score")

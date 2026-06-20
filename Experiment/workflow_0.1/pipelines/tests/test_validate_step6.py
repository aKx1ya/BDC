from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step6_outputs import make_step6_input_chain


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step6_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step6.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_good_step6(tmp_path: Path) -> tuple[object, Path, Path, Path]:
    build_module = load_module("build_step6_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step6", VALIDATE_MODULE_PATH)
    step2_dir, step5_dir, _ = make_step6_input_chain(tmp_path)
    output_dir = (
        tmp_path
        / "Experiment"
        / "workflow_0.1"
        / "experiments"
        / "exp_step6_fixture"
        / "outputs"
        / "step6"
    )
    build_module.build_step6_outputs(
        step5_output_dir=step5_dir / "outputs" / "step5",
        step2_output_dir=step2_dir / "outputs" / "step2",
        output_dir=output_dir,
        input_step5_experiment=step5_dir.name,
        input_step2_experiment=step2_dir.name,
    )
    return validate_module, step2_dir, step5_dir, output_dir


def test_validate_step6_accepts_good_outputs(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)

    metrics = module.validate_step6(step5_dir, step2_dir, output_dir)

    assert metrics["input_step5_experiment"] == "exp_step5_fixture"
    assert metrics["input_step2_experiment"] == "exp_step2_fixture"
    assert metrics["output_candidate_rows"] == 6
    assert metrics["output_selected_count"] <= 5
    assert metrics["output_total_weight"] <= 1


def test_validate_step6_rejects_failed_step5_report(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)
    (step5_dir / "notes" / "step5_run_report.md").write_text(
        """# Step-5 report

## Status

FAILED
""",
        encoding="utf-8",
    )

    with pytest.raises(module.Step6ValidationError, match="Step-5 report"):
        module.validate_step6(step5_dir, step2_dir, output_dir)


def test_validate_step6_rejects_candidate_date_mismatch(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)
    path = step2_dir / "outputs" / "step2" / "step2_data_manifest.csv"
    manifest = pd.read_csv(path)
    manifest.loc[manifest["项目"].eq("latest_T"), "说明"] = "2026-06-16"
    manifest.to_csv(path, index=False)

    with pytest.raises(module.Step6ValidationError, match="does not match Step-2 latest_T"):
        module.validate_step6(step5_dir, step2_dir, output_dir)


def test_validate_step6_rejects_result_outside_top30(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)
    path = output_dir / "step6_result.csv"
    result = pd.read_csv(path, dtype={"stock_id": str})
    result.loc[0, "stock_id"] = "999999"
    result.to_csv(path, index=False)

    with pytest.raises(module.Step6ValidationError, match="outside Step-5 Top30"):
        module.validate_step6(step5_dir, step2_dir, output_dir)


def test_validate_step6_rejects_missing_ranking_candidate(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)
    path = output_dir / "step6_ranking_log.csv"
    ranking = pd.read_csv(path, dtype={"股票代码": str})
    ranking = ranking.iloc[:-1].copy()
    ranking.to_csv(path, index=False)

    with pytest.raises(module.Step6ValidationError, match="row count expected"):
        module.validate_step6(step5_dir, step2_dir, output_dir)


def test_validate_step6_rejects_non_pass_leakage_check(tmp_path: Path) -> None:
    module, step2_dir, step5_dir, output_dir = build_good_step6(tmp_path)
    path = output_dir / "step6_leakage_check.csv"
    leakage = pd.read_csv(path)
    leakage.loc[0, "状态"] = "FAIL"
    leakage.to_csv(path, index=False)

    with pytest.raises(module.Step6ValidationError, match="non-PASS"):
        module.validate_step6(step5_dir, step2_dir, output_dir)

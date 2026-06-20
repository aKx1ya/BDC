from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step2_outputs import make_step1_output


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step2_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step2.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_successful_step1_experiment(root: Path) -> Path:
    experiment_dir = root / "Experiment" / "workflow_0.1" / "experiments" / "exp_step1_fixture"
    make_step1_output(experiment_dir / "outputs" / "step1")
    report_path = experiment_dir / "notes" / "step1_run_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """# Step-1 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return experiment_dir


def build_good_step2(tmp_path: Path) -> tuple[object, Path, Path]:
    build_module = load_module("build_step2_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step2", VALIDATE_MODULE_PATH)
    step1_experiment_dir = make_successful_step1_experiment(tmp_path)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step2_fixture" / "outputs" / "step2"
    build_module.build_step2_outputs(
        step1_output_dir=step1_experiment_dir / "outputs" / "step1",
        output_dir=output_dir,
        input_step1_experiment=step1_experiment_dir.name,
    )
    return validate_module, step1_experiment_dir, output_dir


def test_validate_step2_accepts_good_outputs(tmp_path: Path) -> None:
    module, step1_experiment_dir, output_dir = build_good_step2(tmp_path)

    metrics = module.validate_step2(step1_experiment_dir, output_dir, expected_stock_count=3)

    assert metrics["input_step1_stock_count"] == 3
    assert metrics["output_latest_t_rows"] == 3
    assert metrics["output_feature_duplicates"] == 0


def test_validate_step2_rejects_duplicate_feature_key(tmp_path: Path) -> None:
    module, step1_experiment_dir, output_dir = build_good_step2(tmp_path)
    path = output_dir / "step2_feature_table_daily.csv"
    feature = pd.read_csv(path, dtype={"股票代码": str})
    feature = pd.concat([feature, feature.head(1)], ignore_index=True)
    feature.to_csv(path, index=False)

    with pytest.raises(module.Step2ValidationError, match="duplicate 股票代码"):
        module.validate_step2(step1_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step2_rejects_latest_t_mismatch(tmp_path: Path) -> None:
    module, step1_experiment_dir, output_dir = build_good_step2(tmp_path)
    path = output_dir / "step2_latest_t_screen.csv"
    latest = pd.read_csv(path, dtype={"股票代码": str})
    latest.loc[0, "日期"] = "2026-01-01"
    latest.to_csv(path, index=False)

    with pytest.raises(module.Step2ValidationError, match="latest_T"):
        module.validate_step2(step1_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step2_rejects_inconsistent_derived_sector_view(tmp_path: Path) -> None:
    module, step1_experiment_dir, output_dir = build_good_step2(tmp_path)
    path = output_dir / "step2_sector_score_latest.csv"
    sector_latest = pd.read_csv(path)
    sector_latest.loc[0, "sector_short_score"] = sector_latest.loc[0, "sector_short_score"] + 1
    sector_latest.to_csv(path, index=False)

    with pytest.raises(module.Step2ValidationError, match="sector_score_latest"):
        module.validate_step2(step1_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step2_rejects_missing_leakage_note(tmp_path: Path) -> None:
    module, step1_experiment_dir, output_dir = build_good_step2(tmp_path)
    path = output_dir / "step2_feature_metadata.csv"
    metadata = pd.read_csv(path)
    metadata.loc[0, "防泄漏说明"] = ""
    metadata.to_csv(path, index=False)

    with pytest.raises(module.Step2ValidationError, match="防泄漏说明"):
        module.validate_step2(step1_experiment_dir, output_dir, expected_stock_count=3)

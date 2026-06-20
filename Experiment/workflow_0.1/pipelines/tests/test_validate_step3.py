from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from test_build_step3_outputs import make_step2_output


BUILD_MODULE_PATH = Path(__file__).resolve().parents[1] / "build_step3_outputs.py"
VALIDATE_MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_step3.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_successful_step2_experiment(root: Path, stock_count: int = 3) -> Path:
    experiment_dir = root / "Experiment" / "workflow_0.1" / "experiments" / "exp_step2_fixture"
    make_step2_output(experiment_dir / "outputs" / "step2", stock_count=stock_count)
    report_path = experiment_dir / "notes" / "step2_run_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        """# Step-2 正式健康版运行报告

## Status

SUCCESS
""",
        encoding="utf-8",
    )
    return experiment_dir


def build_good_step3(tmp_path: Path, stock_count: int = 3) -> tuple[object, Path, Path]:
    build_module = load_module("build_step3_outputs_for_validate", BUILD_MODULE_PATH)
    validate_module = load_module("validate_step3", VALIDATE_MODULE_PATH)
    step2_experiment_dir = make_successful_step2_experiment(tmp_path, stock_count=stock_count)
    output_dir = tmp_path / "Experiment" / "workflow_0.1" / "experiments" / "exp_step3_fixture" / "outputs" / "step3"
    build_module.build_step3_outputs(
        step2_output_dir=step2_experiment_dir / "outputs" / "step2",
        output_dir=output_dir,
        input_step2_experiment=step2_experiment_dir.name,
    )
    return validate_module, step2_experiment_dir, output_dir


def test_validate_step3_accepts_good_outputs(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path)

    metrics = module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=3)

    assert metrics["input_step2_stock_count"] == 3
    assert metrics["output_sample_rows"] == 18
    assert metrics["output_sample_dates"] == 6
    assert metrics["output_sample_duplicates"] == 0


def test_validate_step3_rejects_duplicate_sample_key(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path)
    path = output_dir / "step3_sample_table.csv"
    sample = pd.read_csv(path, dtype={"股票代码": str})
    sample = pd.concat([sample, sample.head(1)], ignore_index=True)
    sample.to_csv(path, index=False)

    with pytest.raises(module.Step3ValidationError, match="duplicate"):
        module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step3_rejects_bad_window_row_count(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path)
    path = output_dir / "step3_window_index.csv"
    window = pd.read_csv(path, dtype={"股票代码": str})
    window.loc[0, "window_row_count"] = 59
    window.to_csv(path, index=False)

    with pytest.raises(module.Step3ValidationError, match="window_row_count"):
        module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step3_rejects_label_date_not_after_t(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path)
    path = output_dir / "step3_sample_table.csv"
    sample = pd.read_csv(path, dtype={"股票代码": str})
    sample.loc[0, "label_open_t1_date"] = sample.loc[0, "样本日期T"]
    sample.to_csv(path, index=False)

    with pytest.raises(module.Step3ValidationError, match="label_open_t1_date"):
        module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step3_rejects_group_count_mismatch(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path)
    path = output_dir / "step3_group_info.csv"
    group = pd.read_csv(path)
    group.loc[0, "group_stock_count"] = 999
    group.to_csv(path, index=False)

    with pytest.raises(module.Step3ValidationError, match="group_info count mismatch"):
        module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=3)


def test_validate_step3_rejects_top5_over_limit(tmp_path: Path) -> None:
    module, step2_experiment_dir, output_dir = build_good_step3(tmp_path, stock_count=6)
    path = output_dir / "step3_sample_table.csv"
    sample = pd.read_csv(path, dtype={"股票代码": str})
    first_date = sample["样本日期T"].min()
    sample.loc[sample["样本日期T"].eq(first_date), "label_top5_flag"] = 1
    sample.to_csv(path, index=False)

    with pytest.raises(module.Step3ValidationError, match="label_top5_flag"):
        module.validate_step3(step2_experiment_dir, output_dir, expected_stock_count=6)

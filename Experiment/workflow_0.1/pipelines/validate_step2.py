#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step2_outputs import (  # noqa: E402
    FEATURE_SET_ID,
    GENERATED_FEATURES_FOR_METADATA,
    SCHEMA_VERSION,
    STEP2_FEATURE_COLUMNS,
    STEP2_LATEST_SCREEN_COLUMNS,
    STEP2_METADATA_COLUMNS,
    STEP2_RISK_COLUMNS,
    STEP2_SECTOR_COLUMNS,
)


STEP1_OUTPUT_FILES = {
    "daily": "step1_daily_raw_data.csv",
    "stock": "step1_stock_summary.csv",
    "sector": "step1_sector_summary.csv",
    "manifest": "step1_data_manifest.csv",
}

STEP2_OUTPUT_FILES = {
    "feature": "step2_feature_table_daily.csv",
    "sector": "step2_sector_feature_table.csv",
    "latest": "step2_latest_t_screen.csv",
    "metadata": "step2_feature_metadata.csv",
    "manifest": "step2_data_manifest.csv",
    "sector_latest": "step2_sector_score_latest.csv",
    "risk": "step2_risk_feature_table.csv",
}

REQUIRED_STEP2_MANIFEST_ITEMS = {
    "schema_version",
    "feature_set_id",
    "date_start",
    "date_end",
    "latest_T",
    "raw_交易日数",
    "input_step1_path",
    "input_step1_experiment",
    "input_step1_latest_T",
    "generated_at",
    "data_window_note",
    "feature_set_note",
}


class Step2ValidationError(Exception):
    """Step-2 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step2ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step2ValidationError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> list[str]:
    actual = list(df.columns)
    if actual != expected:
        return [f"{file_name} columns mismatch: expected {expected}, got {actual}"]
    return []


def require_manifest_items(manifest: pd.DataFrame, required: set[str], file_name: str) -> list[str]:
    if list(manifest.columns) != ["项目", "说明"]:
        return [f"{file_name} columns mismatch: expected ['项目', '说明'], got {list(manifest.columns)}"]
    items = set(manifest["项目"].astype(str))
    missing = sorted(required - items)
    if missing:
        return [f"{file_name} missing items: {missing}"]
    return []


def manifest_value(manifest: pd.DataFrame, item: str, default: str = "") -> str:
    matched = manifest.loc[manifest["项目"].astype(str) == item, "说明"]
    if matched.empty:
        return default
    return str(matched.iloc[0])


def report_is_success(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def validate_step1_input(step1_experiment_dir: Path, expected_stock_count: int = 300) -> dict[str, object]:
    step1_experiment_dir = Path(step1_experiment_dir)
    step1_output_dir = step1_experiment_dir / "outputs" / "step1"
    report_path = step1_experiment_dir / "notes" / "step1_run_report.md"
    errors: list[str] = []

    if not report_is_success(report_path):
        errors.append(f"Step-1 report is not SUCCESS: {report_path}")

    daily = read_csv(step1_output_dir / STEP1_OUTPUT_FILES["daily"], dtype={"股票代码": str})
    stock = read_csv(step1_output_dir / STEP1_OUTPUT_FILES["stock"], dtype={"股票代码": str})
    sector = read_csv(step1_output_dir / STEP1_OUTPUT_FILES["sector"])
    manifest = read_csv(step1_output_dir / STEP1_OUTPUT_FILES["manifest"])

    errors += require_manifest_items(
        manifest,
        {"schema_version", "latest_T", "date_start", "date_end", "raw_交易日数"},
        STEP1_OUTPUT_FILES["manifest"],
    )
    schema_version = manifest_value(manifest, "schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"Step-1 schema_version expected {SCHEMA_VERSION}, got {schema_version!r}")

    latest_t = manifest_value(manifest, "latest_T")
    if not latest_t:
        errors.append("Step-1 latest_T is empty")

    duplicate_count = int(daily.duplicated(["股票代码", "日期"]).sum())
    if duplicate_count:
        errors.append(f"step1_daily_raw_data.csv duplicate 股票代码+日期 rows: {duplicate_count}")

    stock_count = int(len(stock))
    if stock_count != expected_stock_count:
        errors.append(f"step1_stock_summary.csv stock count expected {expected_stock_count}, got {stock_count}")

    if "最新日期" in stock.columns:
        latest_dates = sorted(set(stock["最新日期"].dropna().astype(str)))
        if len(latest_dates) != 1 or (latest_t and latest_dates[0] != latest_t):
            errors.append(f"Step-1 stock latest dates not aligned with manifest latest_T: {latest_dates} vs {latest_t}")

    raise_if_errors(errors)
    return {
        "input_step1_experiment": step1_experiment_dir.name,
        "input_step1_output_dir": str(step1_output_dir),
        "input_step1_latest_T": latest_t,
        "input_step1_daily_rows": int(len(daily)),
        "input_step1_stock_count": stock_count,
        "input_step1_sector_count": int(len(sector)),
        "input_step1_daily_duplicates": duplicate_count,
    }


def compare_derived_sector_view(sector: pd.DataFrame, sector_latest: pd.DataFrame, latest_t: str) -> list[str]:
    expected = sector[sector["日期"].astype(str) == latest_t][STEP2_SECTOR_COLUMNS].sort_values(
        ["日期", "板块划分"]
    ).reset_index(drop=True)
    actual = sector_latest[STEP2_SECTOR_COLUMNS].sort_values(["日期", "板块划分"]).reset_index(drop=True)
    try:
        assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, atol=1e-6, rtol=1e-6)
    except AssertionError as exc:
        return [f"step2_sector_score_latest.csv is not latest_T slice of sector table: {exc}"]
    return []


def compare_derived_risk_view(feature: pd.DataFrame, risk: pd.DataFrame) -> list[str]:
    expected = feature[STEP2_RISK_COLUMNS].sort_values(["股票代码", "日期"]).reset_index(drop=True)
    actual = risk[STEP2_RISK_COLUMNS].sort_values(["股票代码", "日期"]).reset_index(drop=True)
    try:
        assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, atol=1e-6, rtol=1e-6)
    except AssertionError as exc:
        return [f"step2_risk_feature_table.csv is not risk-column slice of feature table: {exc}"]
    return []


def validate_outputs(
    output_dir: Path,
    latest_t: str | None = None,
    expected_stock_count: int = 300,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []

    feature = read_csv(output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str})
    sector = read_csv(output_dir / STEP2_OUTPUT_FILES["sector"])
    latest = read_csv(output_dir / STEP2_OUTPUT_FILES["latest"], dtype={"股票代码": str})
    metadata = read_csv(output_dir / STEP2_OUTPUT_FILES["metadata"])
    manifest = read_csv(output_dir / STEP2_OUTPUT_FILES["manifest"])
    sector_latest = read_csv(output_dir / STEP2_OUTPUT_FILES["sector_latest"])
    risk = read_csv(output_dir / STEP2_OUTPUT_FILES["risk"], dtype={"股票代码": str})

    errors += validate_columns(feature, STEP2_FEATURE_COLUMNS, STEP2_OUTPUT_FILES["feature"])
    errors += validate_columns(sector, STEP2_SECTOR_COLUMNS, STEP2_OUTPUT_FILES["sector"])
    errors += validate_columns(latest, STEP2_LATEST_SCREEN_COLUMNS, STEP2_OUTPUT_FILES["latest"])
    errors += validate_columns(metadata, STEP2_METADATA_COLUMNS, STEP2_OUTPUT_FILES["metadata"])
    errors += validate_columns(manifest, ["项目", "说明"], STEP2_OUTPUT_FILES["manifest"])
    errors += validate_columns(sector_latest, STEP2_SECTOR_COLUMNS, STEP2_OUTPUT_FILES["sector_latest"])
    errors += validate_columns(risk, STEP2_RISK_COLUMNS, STEP2_OUTPUT_FILES["risk"])
    errors += require_manifest_items(manifest, REQUIRED_STEP2_MANIFEST_ITEMS, STEP2_OUTPUT_FILES["manifest"])

    schema_version = manifest_value(manifest, "schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"step2_data_manifest.csv schema_version expected {SCHEMA_VERSION}, got {schema_version!r}")

    feature_set_id = manifest_value(manifest, "feature_set_id")
    if feature_set_id != FEATURE_SET_ID:
        errors.append(f"step2_data_manifest.csv feature_set_id expected {FEATURE_SET_ID}, got {feature_set_id!r}")

    manifest_latest_t = manifest_value(manifest, "latest_T")
    if latest_t and manifest_latest_t != latest_t:
        errors.append(f"Step-2 latest_T must match Step-1 latest_T: {manifest_latest_t} vs {latest_t}")
    latest_t = latest_t or manifest_latest_t
    if not latest_t:
        errors.append("Step-2 latest_T is empty")

    feature_duplicates = int(feature.duplicated(["股票代码", "日期"]).sum())
    if feature_duplicates:
        errors.append(f"step2_feature_table_daily.csv duplicate 股票代码+日期 rows: {feature_duplicates}")

    sector_duplicates = int(sector.duplicated(["日期", "板块划分"]).sum())
    if sector_duplicates:
        errors.append(f"step2_sector_feature_table.csv duplicate 日期+板块划分 rows: {sector_duplicates}")

    feature_code_count = int(feature["股票代码"].nunique())
    if feature_code_count != expected_stock_count:
        errors.append(f"step2_feature_table_daily.csv stock count expected {expected_stock_count}, got {feature_code_count}")

    latest_dates = sorted(set(latest["日期"].dropna().astype(str)))
    if latest_t and latest_dates != [latest_t]:
        errors.append(f"step2_latest_t_screen.csv must only contain latest_T={latest_t}, got {latest_dates}")

    latest_row_count = int(len(latest))
    if latest_row_count != expected_stock_count:
        errors.append(f"step2_latest_t_screen.csv row count expected {expected_stock_count}, got {latest_row_count}")

    max_feature_date = str(feature["日期"].max()) if not feature.empty else ""
    if latest_t and max_feature_date != latest_t:
        errors.append(f"feature table max 日期 must be latest_T={latest_t}, got {max_feature_date}")

    if latest_t and (feature["日期"].astype(str) > latest_t).any():
        errors.append("feature table contains dates after latest_T")
    if latest_t and (sector["日期"].astype(str) > latest_t).any():
        errors.append("sector table contains dates after latest_T")

    metadata_features = set(metadata["特征名"].astype(str))
    missing_metadata = sorted(set(GENERATED_FEATURES_FOR_METADATA) - metadata_features)
    if missing_metadata:
        errors.append(f"step2_feature_metadata.csv missing generated features: {missing_metadata[:20]}")

    empty_leakage = metadata["防泄漏说明"].isna() | metadata["防泄漏说明"].astype(str).str.strip().eq("")
    if bool(empty_leakage.any()):
        errors.append("step2_feature_metadata.csv has empty 防泄漏说明")

    if latest_t:
        errors += compare_derived_sector_view(sector, sector_latest, latest_t)
    errors += compare_derived_risk_view(feature, risk)

    raise_if_errors(errors)
    latest_positive_count = int(latest["进入后续流程标记"].eq("是").sum()) if "进入后续流程标记" in latest.columns else 0
    return {
        "output_feature_rows": int(len(feature)),
        "output_feature_stock_count": feature_code_count,
        "output_feature_duplicates": feature_duplicates,
        "output_sector_rows": int(len(sector)),
        "output_sector_duplicates": sector_duplicates,
        "output_latest_T": latest_t or "",
        "output_latest_t_rows": latest_row_count,
        "output_latest_t_pass_count": latest_positive_count,
        "output_sector_latest_rows": int(len(sector_latest)),
        "output_risk_rows": int(len(risk)),
        "output_metadata_rows": int(len(metadata)),
    }


def validate_step2(
    step1_experiment_dir: Path,
    output_dir: Path,
    expected_stock_count: int = 300,
) -> dict[str, object]:
    metrics = validate_step1_input(step1_experiment_dir, expected_stock_count=expected_stock_count)
    output_metrics = validate_outputs(
        output_dir,
        latest_t=str(metrics["input_step1_latest_T"]),
        expected_stock_count=expected_stock_count,
    )
    metrics.update(output_metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-2 inputs and standard outputs.")
    parser.add_argument("--step1-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-stock-count", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step2(
            args.step1_experiment_dir,
            args.output_dir,
            expected_stock_count=args.expected_stock_count,
        )
    except Step2ValidationError as exc:
        print(f"Step-2 validation failed: {exc}")
        return 1

    print("Step-2 validation passed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

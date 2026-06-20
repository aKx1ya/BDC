#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step3_outputs import (  # noqa: E402
    SAMPLE_SET_ID,
    SCHEMA_VERSION,
    STEP3_GROUP_COLUMNS,
    STEP3_LABEL_DISTRIBUTION_COLUMNS,
    STEP3_QUALITY_COLUMNS,
    STEP3_RANK_LABEL_COLUMNS,
    STEP3_SAMPLE_COLUMNS,
    STEP3_WINDOW_INDEX_COLUMNS,
)


STEP2_OUTPUT_FILES = {
    "feature": "step2_feature_table_daily.csv",
    "sector": "step2_sector_feature_table.csv",
    "latest": "step2_latest_t_screen.csv",
    "metadata": "step2_feature_metadata.csv",
    "manifest": "step2_data_manifest.csv",
    "sector_latest": "step2_sector_score_latest.csv",
    "risk": "step2_risk_feature_table.csv",
}

STEP3_OUTPUT_FILES = {
    "sample": "step3_sample_table.csv",
    "window": "step3_window_index.csv",
    "group": "step3_group_info.csv",
    "rank": "step3_rank_label_table.csv",
    "manifest": "step3_sample_manifest.csv",
    "label_distribution": "step3_label_distribution.csv",
    "quality": "step3_sample_quality_summary.csv",
}

REQUIRED_STEP3_MANIFEST_ITEMS = {
    "schema_version",
    "sample_set_id",
    "input_step2_path",
    "input_step2_experiment",
    "input_step2_latest_T",
    "sample_mode",
    "window_length",
    "prediction_horizon",
    "label_formula",
    "label_price_field",
    "sample_date_start",
    "sample_date_end",
    "sample_date_count",
    "sample_row_count",
    "feature_count",
    "generated_at",
    "data_window_note",
    "leakage_control_note",
}


class Step3ValidationError(Exception):
    """Step-3 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step3ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step3ValidationError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> list[str]:
    actual = list(df.columns)
    if actual != expected:
        return [f"{file_name} columns mismatch: expected {expected}, got {actual}"]
    return []


def manifest_value(manifest: pd.DataFrame, item: str, default: str = "") -> str:
    if {"项目", "说明"} - set(manifest.columns):
        return default
    matched = manifest.loc[manifest["项目"].astype(str) == item, "说明"]
    if matched.empty:
        return default
    return str(matched.iloc[0])


def require_manifest_items(manifest: pd.DataFrame, required: set[str], file_name: str) -> list[str]:
    if list(manifest.columns) != ["项目", "说明"]:
        return [f"{file_name} columns mismatch: expected ['项目', '说明'], got {list(manifest.columns)}"]
    items = set(manifest["项目"].astype(str))
    missing = sorted(required - items)
    if missing:
        return [f"{file_name} missing items: {missing}"]
    return []


def report_is_success(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    return "## Status" in text and "SUCCESS" in text.split("## Status", 1)[1].split("##", 1)[0]


def validate_step2_input(step2_experiment_dir: Path, expected_stock_count: int = 300) -> dict[str, object]:
    step2_experiment_dir = Path(step2_experiment_dir)
    step2_output_dir = step2_experiment_dir / "outputs" / "step2"
    report_path = step2_experiment_dir / "notes" / "step2_run_report.md"
    errors: list[str] = []

    if not report_is_success(report_path):
        errors.append(f"Step-2 report is not SUCCESS: {report_path}")

    feature = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str})
    metadata = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["metadata"])
    manifest = read_csv(step2_output_dir / STEP2_OUTPUT_FILES["manifest"])

    errors += require_manifest_items(
        manifest,
        {"schema_version", "feature_set_id", "latest_T", "date_start", "date_end"},
        STEP2_OUTPUT_FILES["manifest"],
    )
    schema_version = manifest_value(manifest, "schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"Step-2 schema_version expected {SCHEMA_VERSION}, got {schema_version!r}")

    latest_t = manifest_value(manifest, "latest_T")
    if not latest_t:
        errors.append("Step-2 latest_T is empty")

    duplicate_count = int(feature.duplicated(["股票代码", "日期"]).sum())
    if duplicate_count:
        errors.append(f"step2_feature_table_daily.csv duplicate 股票代码+日期 rows: {duplicate_count}")

    code_count = int(feature["股票代码"].nunique()) if "股票代码" in feature.columns else 0
    if code_count != expected_stock_count:
        errors.append(f"step2_feature_table_daily.csv stock count expected {expected_stock_count}, got {code_count}")

    if "防泄漏说明" not in metadata.columns:
        errors.append("step2_feature_metadata.csv missing 防泄漏说明")
    else:
        empty_leakage = metadata["防泄漏说明"].isna() | metadata["防泄漏说明"].astype(str).str.strip().eq("")
        if bool(empty_leakage.any()):
            errors.append("step2_feature_metadata.csv has empty 防泄漏说明")

    dates = sorted(feature["日期"].dropna().astype(str).unique())
    if len(dates) < 6:
        errors.append("Step-2 feature dates fewer than 6, cannot build 5-day labels")

    raise_if_errors(errors)

    return {
        "input_step2_experiment": step2_experiment_dir.name,
        "input_step2_output_dir": str(step2_output_dir),
        "input_step2_latest_T": latest_t,
        "input_step2_feature_rows": int(len(feature)),
        "input_step2_stock_count": code_count,
        "input_step2_date_count": int(len(dates)),
        "input_step2_feature_duplicates": duplicate_count,
        "last_labelable_T": dates[-6] if len(dates) >= 6 else "",
    }


def key_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df[keys].astype(str).sort_values(keys).reset_index(drop=True)


def validate_outputs(
    output_dir: Path,
    *,
    input_step2_latest_t: str | None = None,
    last_labelable_t: str | None = None,
    expected_stock_count: int = 300,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []

    sample = read_csv(output_dir / STEP3_OUTPUT_FILES["sample"], dtype={"股票代码": str})
    window = read_csv(output_dir / STEP3_OUTPUT_FILES["window"], dtype={"股票代码": str})
    group = read_csv(output_dir / STEP3_OUTPUT_FILES["group"])
    rank = read_csv(output_dir / STEP3_OUTPUT_FILES["rank"], dtype={"股票代码": str})
    manifest = read_csv(output_dir / STEP3_OUTPUT_FILES["manifest"])
    label_distribution = read_csv(output_dir / STEP3_OUTPUT_FILES["label_distribution"])
    quality = read_csv(output_dir / STEP3_OUTPUT_FILES["quality"])

    errors += validate_columns(sample, STEP3_SAMPLE_COLUMNS, STEP3_OUTPUT_FILES["sample"])
    errors += validate_columns(window, STEP3_WINDOW_INDEX_COLUMNS, STEP3_OUTPUT_FILES["window"])
    errors += validate_columns(group, STEP3_GROUP_COLUMNS, STEP3_OUTPUT_FILES["group"])
    errors += validate_columns(rank, STEP3_RANK_LABEL_COLUMNS, STEP3_OUTPUT_FILES["rank"])
    errors += validate_columns(manifest, ["项目", "说明"], STEP3_OUTPUT_FILES["manifest"])
    errors += validate_columns(label_distribution, STEP3_LABEL_DISTRIBUTION_COLUMNS, STEP3_OUTPUT_FILES["label_distribution"])
    errors += validate_columns(quality, STEP3_QUALITY_COLUMNS, STEP3_OUTPUT_FILES["quality"])
    errors += require_manifest_items(manifest, REQUIRED_STEP3_MANIFEST_ITEMS, STEP3_OUTPUT_FILES["manifest"])

    schema_version = manifest_value(manifest, "schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"step3_sample_manifest.csv schema_version expected {SCHEMA_VERSION}, got {schema_version!r}")

    sample_set_id = manifest_value(manifest, "sample_set_id")
    if sample_set_id != SAMPLE_SET_ID:
        errors.append(f"step3_sample_manifest.csv sample_set_id expected {SAMPLE_SET_ID}, got {sample_set_id!r}")

    if input_step2_latest_t and manifest_value(manifest, "input_step2_latest_T") != input_step2_latest_t:
        errors.append(
            f"Step-3 input_step2_latest_T must match Step-2 latest_T: "
            f"{manifest_value(manifest, 'input_step2_latest_T')} vs {input_step2_latest_t}"
        )

    sample_duplicates = int(sample.duplicated(["样本日期T", "股票代码"]).sum())
    if sample_duplicates:
        errors.append(f"step3_sample_table.csv duplicate 样本日期T+股票代码 rows: {sample_duplicates}")

    window_duplicates = int(window.duplicated(["样本日期T", "股票代码"]).sum())
    if window_duplicates:
        errors.append(f"step3_window_index.csv duplicate 样本日期T+股票代码 rows: {window_duplicates}")

    rank_duplicates = int(rank.duplicated(["样本日期T", "股票代码"]).sum())
    if rank_duplicates:
        errors.append(f"step3_rank_label_table.csv duplicate 样本日期T+股票代码 rows: {rank_duplicates}")

    group_duplicates = int(group.duplicated(["样本日期T"]).sum())
    if group_duplicates:
        errors.append(f"step3_group_info.csv duplicate 样本日期T rows: {group_duplicates}")

    sample_keys = key_frame(sample, ["样本日期T", "股票代码"])
    if not sample_keys.equals(key_frame(window, ["样本日期T", "股票代码"])):
        errors.append("window_index keys do not match sample_table keys")
    if not sample_keys.equals(key_frame(rank, ["样本日期T", "股票代码"])):
        errors.append("rank_label_table keys do not match sample_table keys")

    if not sample.empty:
        numeric_label = pd.to_numeric(sample["label_ret_5d_open_to_open"], errors="coerce")
        if numeric_label.isna().any() or np.isinf(numeric_label.to_numpy()).any():
            errors.append("sample_table has NaN or inf labels")

        for flag_col in ["label_top5_flag", "label_top10_flag", "label_top30_flag"]:
            sample[flag_col] = pd.to_numeric(sample[flag_col], errors="coerce").fillna(0).astype(int)

        top_limits = {"label_top5_flag": 5, "label_top10_flag": 10, "label_top30_flag": 30}
        for flag_col, limit in top_limits.items():
            max_count = int(sample.groupby("样本日期T")[flag_col].sum().max())
            if max_count > limit:
                errors.append(f"{flag_col} has daily count {max_count}, expected <= {limit}")

        if not sample["样本可用标记"].eq("是").all():
            errors.append("sample_table contains unavailable samples")
        if not window["窗口完整标记"].eq("是").all():
            errors.append("window_index contains incomplete windows")

        window_row_count = pd.to_numeric(window["window_row_count"], errors="coerce")
        window_length = pd.to_numeric(window["window_length"], errors="coerce")
        if not window_row_count.eq(window_length).all():
            errors.append("window_index has window_row_count != window_length")

        if not (pd.to_datetime(window["window_end"]) == pd.to_datetime(window["样本日期T"])).all():
            errors.append("window_end must equal 样本日期T")
        if not (pd.to_datetime(window["window_start"]) < pd.to_datetime(window["样本日期T"])).all():
            errors.append("window_start must be before 样本日期T")

        t_dates = pd.to_datetime(sample["样本日期T"])
        t1_dates = pd.to_datetime(sample["label_open_t1_date"])
        t5_dates = pd.to_datetime(sample["label_open_t5_date"])
        if not (t1_dates > t_dates).all():
            errors.append("label_open_t1_date must be after 样本日期T")
        if not (t5_dates > t1_dates).all():
            errors.append("label_open_t5_date must be after label_open_t1_date")
        if not pd.to_numeric(sample["label_open_t1"], errors="coerce").gt(0).all():
            errors.append("label_open_t1 must be > 0")
        if not pd.to_numeric(sample["label_open_t5"], errors="coerce").gt(0).all():
            errors.append("label_open_t5 must be > 0")

        if last_labelable_t and str(sample["样本日期T"].max()) > last_labelable_t:
            errors.append(
                f"sample_table max 样本日期T must be <= last_labelable_T={last_labelable_t}, got {sample['样本日期T'].max()}"
            )

        group_counts = sample.groupby("样本日期T").size()
        group_count_map = group.set_index("样本日期T")["group_stock_count"].astype(int)
        for sample_date, count in group_counts.items():
            if int(group_count_map.get(sample_date, -1)) != int(count):
                errors.append(f"group_info count mismatch for {sample_date}: {group_count_map.get(sample_date)} vs {count}")
                break

        if int(sample["股票代码"].nunique()) != expected_stock_count:
            errors.append(
                f"step3_sample_table.csv stock count expected {expected_stock_count}, got {sample['股票代码'].nunique()}"
            )

    leakage_note = manifest_value(manifest, "leakage_control_note")
    if not leakage_note:
        errors.append("step3_sample_manifest.csv missing leakage_control_note")

    raise_if_errors(errors)

    return {
        "output_sample_rows": int(len(sample)),
        "output_sample_dates": int(sample["样本日期T"].nunique()) if "样本日期T" in sample.columns else 0,
        "output_sample_duplicates": sample_duplicates,
        "output_window_rows": int(len(window)),
        "output_group_rows": int(len(group)),
        "output_rank_rows": int(len(rank)),
        "output_label_distribution_rows": int(len(label_distribution)),
        "output_quality_rows": int(len(quality)),
        "output_sample_date_start": "" if sample.empty else str(sample["样本日期T"].min()),
        "output_sample_date_end": "" if sample.empty else str(sample["样本日期T"].max()),
        "output_stock_count": int(sample["股票代码"].nunique()) if "股票代码" in sample.columns else 0,
    }


def validate_step3(
    step2_experiment_dir: Path,
    output_dir: Path,
    expected_stock_count: int = 300,
) -> dict[str, object]:
    metrics = validate_step2_input(step2_experiment_dir, expected_stock_count=expected_stock_count)
    output_metrics = validate_outputs(
        output_dir,
        input_step2_latest_t=str(metrics["input_step2_latest_T"]),
        last_labelable_t=str(metrics["last_labelable_T"]),
        expected_stock_count=expected_stock_count,
    )
    metrics.update(output_metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-3 inputs and sample outputs.")
    parser.add_argument("--step2-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-stock-count", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step3(
            args.step2_experiment_dir,
            args.output_dir,
            expected_stock_count=args.expected_stock_count,
        )
    except Step3ValidationError as exc:
        print(f"Step-3 validation failed: {exc}")
        return 1

    print("Step-3 validation passed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

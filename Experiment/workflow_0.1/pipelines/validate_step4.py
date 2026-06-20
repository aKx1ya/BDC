#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step3_outputs import SAMPLE_SET_ID  # noqa: E402
from build_step4_outputs import (  # noqa: E402
    DEFAULT_EVAL_DAYS,
    DEFAULT_FINAL_TEST_DAYS,
    DEFAULT_GAP_DAYS,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_WALK_FORWARD_STEP,
    SCHEMA_VERSION,
    SPLIT_SET_ID,
    STEP4_FINAL_RETRAIN_COLUMNS,
    STEP4_LEAKAGE_CHECK_COLUMNS,
    STEP4_SPLIT_DETAIL_COLUMNS,
    STEP4_SPLIT_SUMMARY_COLUMNS,
    STEP4_WALK_FORWARD_COLUMNS,
)


STEP3_OUTPUT_FILES = {
    "sample": "step3_sample_table.csv",
    "window": "step3_window_index.csv",
    "group": "step3_group_info.csv",
    "rank": "step3_rank_label_table.csv",
    "manifest": "step3_sample_manifest.csv",
    "label_distribution": "step3_label_distribution.csv",
    "quality": "step3_sample_quality_summary.csv",
}

STEP4_OUTPUT_FILES = {
    "split_detail": "step4_split_detail.csv",
    "split_summary": "step4_split_summary.csv",
    "walk_forward": "step4_walk_forward_plan.csv",
    "final_retrain": "step4_final_retrain_plan.csv",
    "manifest": "step4_split_manifest.csv",
    "leakage_check": "step4_leakage_check.csv",
}

REQUIRED_STEP4_MANIFEST_ITEMS = {
    "schema_version",
    "split_set_id",
    "input_step3_path",
    "input_step3_experiment",
    "input_step3_sample_set_id",
    "split_mode",
    "sample_date_start",
    "sample_date_end",
    "sample_date_count",
    "sample_row_count",
    "train_window",
    "gap_days",
    "eval_days",
    "walk_forward_step",
    "train_ratio",
    "final_test_days",
    "walk_forward_rounds",
    "generated_at",
    "data_window_note",
    "leakage_control_note",
}


class Step4ValidationError(Exception):
    """Step-4 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step4ValidationError("; ".join(errors))


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step4ValidationError(f"missing file: {path}")
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


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def date_index_map(dates: list[str]) -> dict[str, int]:
    return {sample_date: idx for idx, sample_date in enumerate(dates)}


def contiguous_indexes(values: list[int]) -> bool:
    return values == list(range(min(values), max(values) + 1)) if values else False


def validate_step3_input(step3_experiment_dir: Path) -> dict[str, object]:
    step3_experiment_dir = Path(step3_experiment_dir)
    step3_output_dir = step3_experiment_dir / "outputs" / "step3"
    report_path = step3_experiment_dir / "notes" / "step3_run_report.md"
    errors: list[str] = []

    if not report_is_success(report_path):
        errors.append(f"Step-3 report is not SUCCESS: {report_path}")

    for file_name in STEP3_OUTPUT_FILES.values():
        if not (step3_output_dir / file_name).exists():
            errors.append(f"missing Step-3 output: {step3_output_dir / file_name}")

    sample = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["sample"], dtype={"股票代码": str})
    group = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["group"])
    rank = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["rank"], dtype={"股票代码": str})
    manifest = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["manifest"])

    errors += validate_columns(manifest, ["项目", "说明"], STEP3_OUTPUT_FILES["manifest"])

    schema_version = manifest_value(manifest, "schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"Step-3 schema_version expected {SCHEMA_VERSION}, got {schema_version!r}")

    sample_set_id = manifest_value(manifest, "sample_set_id")
    if sample_set_id != SAMPLE_SET_ID:
        errors.append(f"Step-3 sample_set_id expected {SAMPLE_SET_ID}, got {sample_set_id!r}")

    for file_name, frame in [
        (STEP3_OUTPUT_FILES["sample"], sample),
        (STEP3_OUTPUT_FILES["group"], group),
        (STEP3_OUTPUT_FILES["rank"], rank),
    ]:
        if "样本日期T" not in frame.columns:
            errors.append(f"{file_name} missing 样本日期T")

    if "样本日期T" in group.columns:
        group_duplicates = int(group.duplicated(["样本日期T"]).sum())
        if group_duplicates:
            errors.append(f"step3_group_info.csv duplicate 样本日期T rows: {group_duplicates}")

    if {"样本日期T", "股票代码"} <= set(sample.columns):
        sample_duplicates = int(sample.duplicated(["样本日期T", "股票代码"]).sum())
        if sample_duplicates:
            errors.append(f"step3_sample_table.csv duplicate 样本日期T+股票代码 rows: {sample_duplicates}")

    if {"样本日期T", "股票代码"} <= set(rank.columns):
        rank_duplicates = int(rank.duplicated(["样本日期T", "股票代码"]).sum())
        if rank_duplicates:
            errors.append(f"step3_rank_label_table.csv duplicate 样本日期T+股票代码 rows: {rank_duplicates}")

    sample_dates = sorted(sample["样本日期T"].dropna().astype(str).unique()) if "样本日期T" in sample.columns else []
    group_dates = sorted(group["样本日期T"].dropna().astype(str).unique()) if "样本日期T" in group.columns else []
    if sample_dates != group_dates:
        errors.append("Step-3 sample_table dates do not match group_info dates")

    if not sample_dates:
        errors.append("Step-3 has no sample dates")

    raise_if_errors(errors)

    return {
        "input_step3_experiment": step3_experiment_dir.name,
        "input_step3_output_dir": str(step3_output_dir),
        "input_step3_sample_set_id": sample_set_id,
        "input_step3_schema_version": schema_version,
        "input_step3_sample_date_start": sample_dates[0],
        "input_step3_sample_date_end": sample_dates[-1],
        "input_step3_sample_date_count": int(len(sample_dates)),
        "input_step3_sample_rows": int(len(sample)),
        "input_step3_group_rows": int(len(group)),
        "input_step3_stock_count": int(sample["股票代码"].nunique()) if "股票代码" in sample.columns else 0,
    }


def validate_split_detail(
    detail: pd.DataFrame,
    *,
    train_ratio: float,
    gap_days: int,
    final_test_days: int,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(detail, STEP4_SPLIT_DETAIL_COLUMNS, STEP4_OUTPUT_FILES["split_detail"])
    if errors:
        return errors
    if detail.empty:
        return ["step4_split_detail.csv is empty"]

    duplicate_dates = int(detail.duplicated(["样本日期T"]).sum())
    if duplicate_dates:
        errors.append(f"step4_split_detail.csv duplicate 样本日期T rows: {duplicate_dates}")

    dates = detail["样本日期T"].astype(str).tolist()
    sorted_dates = pd.to_datetime(detail["样本日期T"], errors="coerce").sort_values().dt.strftime("%Y-%m-%d").tolist()
    if dates != sorted_dates:
        errors.append("step4_split_detail.csv dates must be sorted ascending")

    expected_order = list(range(len(detail)))
    actual_order = pd.to_numeric(detail["split_order"], errors="coerce").fillna(-1).astype(int).tolist()
    if actual_order != expected_order:
        errors.append("split_order must be 0-based and match sample date order")

    roles = detail["split_role"].astype(str)
    allowed_roles = {"inner_train", "gap", "validation", "final_test"}
    invalid_roles = sorted(set(roles) - allowed_roles)
    if invalid_roles:
        errors.append(f"step4_split_detail.csv has invalid split_role values: {invalid_roles}")
    if "unassigned" in set(roles):
        errors.append("step4_split_detail.csv contains unassigned dates")

    role_indexes = {role: detail.index[roles.eq(role)].tolist() for role in allowed_roles}
    for role, indexes in role_indexes.items():
        if not indexes:
            errors.append(f"step4_split_detail.csv missing role: {role}")
        elif not contiguous_indexes(indexes):
            errors.append(f"split_role {role} is not contiguous")

    if all(role_indexes.get(role) for role in ["inner_train", "gap", "validation", "final_test"]):
        if not max(role_indexes["inner_train"]) < min(role_indexes["gap"]):
            errors.append("inner_train must be before gap")
        if not max(role_indexes["gap"]) < min(role_indexes["validation"]):
            errors.append("gap must be before validation")
        if not max(role_indexes["validation"]) < min(role_indexes["final_test"]):
            errors.append("validation must be before final_test")

    role_counts = roles.value_counts().to_dict()
    if int(role_counts.get("gap", 0)) != gap_days:
        errors.append(f"gap date count expected {gap_days}, got {role_counts.get('gap', 0)}")
    if int(role_counts.get("final_test", 0)) != final_test_days:
        errors.append(f"final_test date count expected {final_test_days}, got {role_counts.get('final_test', 0)}")

    modeling_count = max(len(detail) - final_test_days, 0)
    expected_train = int(modeling_count * train_ratio)
    if int(role_counts.get("inner_train", 0)) != expected_train:
        errors.append(f"inner_train date count expected {expected_train}, got {role_counts.get('inner_train', 0)}")

    final_test_dates = set(dates[-final_test_days:])
    actual_final_dates = set(detail.loc[roles.eq("final_test"), "样本日期T"].astype(str))
    if actual_final_dates != final_test_dates:
        errors.append("final_test must be the last final_test_days sample dates")

    train_allowed = pd.to_numeric(detail["is_train_allowed"], errors="coerce").fillna(-1).astype(int)
    validation_allowed = pd.to_numeric(detail["is_validation_allowed"], errors="coerce").fillna(-1).astype(int)
    is_final_test = pd.to_numeric(detail["is_final_test"], errors="coerce").fillna(-1).astype(int)
    if not ((train_allowed.eq(1) == roles.eq("inner_train")).all()):
        errors.append("is_train_allowed must be 1 only for inner_train")
    if not ((validation_allowed.eq(1) == roles.eq("validation")).all()):
        errors.append("is_validation_allowed must be 1 only for validation")
    if not ((is_final_test.eq(1) == roles.eq("final_test")).all()):
        errors.append("is_final_test must be 1 only for final_test")
    if pd.to_numeric(detail["sample_row_count"], errors="coerce").le(0).any():
        errors.append("sample_row_count must be positive for every sample date")
    if detail["leakage_guard_note"].isna().any() or detail["leakage_guard_note"].astype(str).str.strip().eq("").any():
        errors.append("leakage_guard_note must be non-empty")
    return errors


def validate_split_summary(summary: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(summary, STEP4_SPLIT_SUMMARY_COLUMNS, STEP4_OUTPUT_FILES["split_summary"])
    if errors:
        return errors
    duplicate_roles = int(summary.duplicated(["split_role"]).sum())
    if duplicate_roles:
        errors.append(f"step4_split_summary.csv duplicate split_role rows: {duplicate_roles}")

    detail_roles = detail.groupby("split_role").agg(
        date_start=("样本日期T", "min"),
        date_end=("样本日期T", "max"),
        date_count=("样本日期T", "count"),
        sample_row_count=("sample_row_count", "sum"),
    )
    for _, row in summary.iterrows():
        role = str(row["split_role"])
        if role not in detail_roles.index:
            errors.append(f"split_summary role {role} not found in split_detail")
            continue
        expected = detail_roles.loc[role]
        if str(row["date_start"]) != str(expected["date_start"]):
            errors.append(f"split_summary date_start mismatch for {role}")
        if str(row["date_end"]) != str(expected["date_end"]):
            errors.append(f"split_summary date_end mismatch for {role}")
        if int_value(row["date_count"]) != int(expected["date_count"]):
            errors.append(f"split_summary date_count mismatch for {role}")
        if int_value(row["sample_row_count"]) != int(expected["sample_row_count"]):
            errors.append(f"split_summary sample_row_count mismatch for {role}")
    return errors


def validate_walk_forward(
    walk_forward: pd.DataFrame,
    detail: pd.DataFrame,
    *,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(walk_forward, STEP4_WALK_FORWARD_COLUMNS, STEP4_OUTPUT_FILES["walk_forward"])
    if errors:
        return errors
    if walk_forward.empty:
        return ["step4_walk_forward_plan.csv must have at least 1 round"]

    duplicate_rounds = int(walk_forward.duplicated(["wf_round"]).sum())
    if duplicate_rounds:
        errors.append(f"step4_walk_forward_plan.csv duplicate wf_round rows: {duplicate_rounds}")

    rounds = pd.to_numeric(walk_forward["wf_round"], errors="coerce").fillna(-1).astype(int).tolist()
    if rounds != list(range(1, len(walk_forward) + 1)):
        errors.append("wf_round must be 1-based and continuous")

    sample_dates = detail["样本日期T"].astype(str).tolist()
    indexes = date_index_map(sample_dates)
    train_starts: list[int] = []
    for _, row in walk_forward.iterrows():
        fields = ["train_start", "train_end", "gap_start", "gap_end", "eval_start", "eval_end"]
        if any(str(row[field]) not in indexes for field in fields):
            errors.append(f"walk_forward round {row['wf_round']} references dates outside split_detail")
            continue
        train_start_idx = indexes[str(row["train_start"])]
        train_end_idx = indexes[str(row["train_end"])]
        gap_start_idx = indexes[str(row["gap_start"])]
        gap_end_idx = indexes[str(row["gap_end"])]
        eval_start_idx = indexes[str(row["eval_start"])]
        eval_end_idx = indexes[str(row["eval_end"])]
        train_starts.append(train_start_idx)

        if not train_start_idx <= train_end_idx < gap_start_idx <= gap_end_idx < eval_start_idx <= eval_end_idx:
            errors.append(f"walk_forward round {row['wf_round']} order is invalid")
        if train_end_idx - train_start_idx + 1 != train_window:
            errors.append(f"walk_forward round {row['wf_round']} train_date_count index span mismatch")
        if gap_end_idx - gap_start_idx + 1 != gap_days:
            errors.append(f"walk_forward round {row['wf_round']} gap_date_count index span mismatch")
        if eval_end_idx - eval_start_idx + 1 != eval_days:
            errors.append(f"walk_forward round {row['wf_round']} eval_date_count index span mismatch")

        if int_value(row["train_date_count"]) != train_window:
            errors.append(f"walk_forward round {row['wf_round']} train_date_count expected {train_window}")
        if int_value(row["gap_date_count"]) != gap_days:
            errors.append(f"walk_forward round {row['wf_round']} gap_date_count expected {gap_days}")
        if int_value(row["eval_date_count"]) != eval_days:
            errors.append(f"walk_forward round {row['wf_round']} eval_date_count expected {eval_days}")
        if int_value(row["train_window"]) != train_window:
            errors.append(f"walk_forward round {row['wf_round']} train_window expected {train_window}")
        if int_value(row["gap_days"]) != gap_days:
            errors.append(f"walk_forward round {row['wf_round']} gap_days expected {gap_days}")
        if int_value(row["eval_days"]) != eval_days:
            errors.append(f"walk_forward round {row['wf_round']} eval_days expected {eval_days}")
        if int_value(row["walk_forward_step"]) != walk_forward_step:
            errors.append(f"walk_forward round {row['wf_round']} walk_forward_step expected {walk_forward_step}")

        train_rows = detail.iloc[train_start_idx : train_end_idx + 1]["sample_row_count"].sum()
        eval_rows = detail.iloc[eval_start_idx : eval_end_idx + 1]["sample_row_count"].sum()
        if int_value(row["train_sample_rows"]) != int(train_rows):
            errors.append(f"walk_forward round {row['wf_round']} train_sample_rows mismatch")
        if int_value(row["eval_sample_rows"]) != int(eval_rows):
            errors.append(f"walk_forward round {row['wf_round']} eval_sample_rows mismatch")
        if str(row["round_status"]) != "ready":
            errors.append(f"walk_forward round {row['wf_round']} round_status must be ready")

    if len(train_starts) > 1:
        diffs = [right - left for left, right in zip(train_starts, train_starts[1:])]
        if any(diff != walk_forward_step for diff in diffs):
            errors.append("walk_forward train_start must advance by walk_forward_step")
    return errors


def validate_final_retrain(final_retrain: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(final_retrain, STEP4_FINAL_RETRAIN_COLUMNS, STEP4_OUTPUT_FILES["final_retrain"])
    if errors:
        return errors
    duplicate_dates = int(final_retrain.duplicated(["样本日期T"]).sum())
    if duplicate_dates:
        errors.append(f"step4_final_retrain_plan.csv duplicate 样本日期T rows: {duplicate_dates}")
    if final_retrain["样本日期T"].astype(str).tolist() != detail["样本日期T"].astype(str).tolist():
        errors.append("final_retrain_plan dates must match split_detail dates")

    merged = final_retrain.merge(detail[["样本日期T", "split_role"]], on="样本日期T", how="left")
    allowed = pd.to_numeric(merged["final_retrain_allowed"], errors="coerce").fillna(-1).astype(int)
    roles = merged["split_role"].astype(str)
    if not ((allowed.eq(1) == roles.isin(["inner_train", "validation"])).all()):
        errors.append("final_retrain_allowed must be 1 only for inner_train and validation")
    if merged["reason"].isna().any() or merged["reason"].astype(str).str.strip().eq("").any():
        errors.append("final_retrain_plan reason must be non-empty")
    return errors


def validate_manifest(
    manifest: pd.DataFrame,
    detail: pd.DataFrame,
    walk_forward: pd.DataFrame,
    *,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
    train_ratio: float,
    final_test_days: int,
) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(manifest, ["项目", "说明"], STEP4_OUTPUT_FILES["manifest"])
    errors += require_manifest_items(manifest, REQUIRED_STEP4_MANIFEST_ITEMS, STEP4_OUTPUT_FILES["manifest"])
    if errors:
        return errors
    if manifest_value(manifest, "schema_version") != SCHEMA_VERSION:
        errors.append(f"step4_split_manifest.csv schema_version expected {SCHEMA_VERSION}")
    if manifest_value(manifest, "split_set_id") != SPLIT_SET_ID:
        errors.append(f"step4_split_manifest.csv split_set_id expected {SPLIT_SET_ID}")
    if manifest_value(manifest, "split_mode") != "time_ordered":
        errors.append("step4_split_manifest.csv split_mode must be time_ordered")
    if manifest_value(manifest, "input_step3_sample_set_id") != SAMPLE_SET_ID:
        errors.append(f"input_step3_sample_set_id expected {SAMPLE_SET_ID}")
    if manifest_value(manifest, "sample_date_start") != str(detail["样本日期T"].iloc[0]):
        errors.append("manifest sample_date_start mismatch")
    if manifest_value(manifest, "sample_date_end") != str(detail["样本日期T"].iloc[-1]):
        errors.append("manifest sample_date_end mismatch")
    if int_value(manifest_value(manifest, "sample_date_count")) != len(detail):
        errors.append("manifest sample_date_count mismatch")
    if int_value(manifest_value(manifest, "sample_row_count")) != int(detail["sample_row_count"].sum()):
        errors.append("manifest sample_row_count mismatch")
    if int_value(manifest_value(manifest, "train_window")) != train_window:
        errors.append("manifest train_window mismatch")
    if int_value(manifest_value(manifest, "gap_days")) != gap_days:
        errors.append("manifest gap_days mismatch")
    if int_value(manifest_value(manifest, "eval_days")) != eval_days:
        errors.append("manifest eval_days mismatch")
    if int_value(manifest_value(manifest, "walk_forward_step")) != walk_forward_step:
        errors.append("manifest walk_forward_step mismatch")
    if abs(float_value(manifest_value(manifest, "train_ratio")) - train_ratio) > 1e-9:
        errors.append("manifest train_ratio mismatch")
    if int_value(manifest_value(manifest, "final_test_days")) != final_test_days:
        errors.append("manifest final_test_days mismatch")
    if int_value(manifest_value(manifest, "walk_forward_rounds")) != len(walk_forward):
        errors.append("manifest walk_forward_rounds mismatch")
    if not manifest_value(manifest, "generated_at").strip():
        errors.append("manifest generated_at is empty")
    if not manifest_value(manifest, "leakage_control_note").strip():
        errors.append("manifest leakage_control_note is empty")
    return errors


def validate_leakage_check(leakage_check: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    errors += validate_columns(leakage_check, STEP4_LEAKAGE_CHECK_COLUMNS, STEP4_OUTPUT_FILES["leakage_check"])
    if errors:
        return errors
    if leakage_check.empty:
        errors.append("step4_leakage_check.csv is empty")
    bad = leakage_check[~leakage_check["状态"].astype(str).eq("PASS")]
    if not bad.empty:
        errors.append(f"step4_leakage_check.csv has non-PASS rows: {bad['检查项'].astype(str).tolist()}")
    return errors


def validate_outputs(
    output_dir: Path,
    *,
    input_step3_metrics: dict[str, object] | None = None,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    gap_days: int = DEFAULT_GAP_DAYS,
    eval_days: int = DEFAULT_EVAL_DAYS,
    walk_forward_step: int = DEFAULT_WALK_FORWARD_STEP,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    final_test_days: int = DEFAULT_FINAL_TEST_DAYS,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []

    detail = read_csv(output_dir / STEP4_OUTPUT_FILES["split_detail"])
    summary = read_csv(output_dir / STEP4_OUTPUT_FILES["split_summary"])
    walk_forward = read_csv(output_dir / STEP4_OUTPUT_FILES["walk_forward"])
    final_retrain = read_csv(output_dir / STEP4_OUTPUT_FILES["final_retrain"])
    manifest = read_csv(output_dir / STEP4_OUTPUT_FILES["manifest"])
    leakage_check = read_csv(output_dir / STEP4_OUTPUT_FILES["leakage_check"])

    errors += validate_split_detail(
        detail,
        train_ratio=train_ratio,
        gap_days=gap_days,
        final_test_days=final_test_days,
    )
    if not errors:
        errors += validate_split_summary(summary, detail)
        errors += validate_walk_forward(
            walk_forward,
            detail,
            train_window=train_window,
            gap_days=gap_days,
            eval_days=eval_days,
            walk_forward_step=walk_forward_step,
        )
        errors += validate_final_retrain(final_retrain, detail)
        errors += validate_manifest(
            manifest,
            detail,
            walk_forward,
            train_window=train_window,
            gap_days=gap_days,
            eval_days=eval_days,
            walk_forward_step=walk_forward_step,
            train_ratio=train_ratio,
            final_test_days=final_test_days,
        )
        errors += validate_leakage_check(leakage_check)

    if input_step3_metrics:
        if int(input_step3_metrics.get("input_step3_sample_date_count", -1)) != int(len(detail)):
            errors.append("Step-4 sample_date_count does not match Step-3 input")
        if int(input_step3_metrics.get("input_step3_sample_rows", -1)) != int(detail["sample_row_count"].sum()):
            errors.append("Step-4 sample_row_count does not match Step-3 input")
        if str(input_step3_metrics.get("input_step3_sample_date_start", "")) != str(detail["样本日期T"].iloc[0]):
            errors.append("Step-4 sample_date_start does not match Step-3 input")
        if str(input_step3_metrics.get("input_step3_sample_date_end", "")) != str(detail["样本日期T"].iloc[-1]):
            errors.append("Step-4 sample_date_end does not match Step-3 input")

    raise_if_errors(errors)

    role_counts = detail["split_role"].value_counts().to_dict()
    return {
        "output_split_dates": int(len(detail)),
        "output_sample_rows": int(detail["sample_row_count"].sum()),
        "output_inner_train_dates": int(role_counts.get("inner_train", 0)),
        "output_gap_dates": int(role_counts.get("gap", 0)),
        "output_validation_dates": int(role_counts.get("validation", 0)),
        "output_final_test_dates": int(role_counts.get("final_test", 0)),
        "output_walk_forward_rounds": int(len(walk_forward)),
        "output_final_retrain_allowed_dates": int(
            pd.to_numeric(final_retrain["final_retrain_allowed"], errors="coerce").fillna(0).astype(int).sum()
        ),
        "output_date_start": str(detail["样本日期T"].iloc[0]),
        "output_date_end": str(detail["样本日期T"].iloc[-1]),
        "output_final_test_start": str(detail[detail["split_role"].eq("final_test")]["样本日期T"].min()),
        "output_final_test_end": str(detail[detail["split_role"].eq("final_test")]["样本日期T"].max()),
        "output_leakage_check_rows": int(len(leakage_check)),
    }


def validate_step4(
    step3_experiment_dir: Path,
    output_dir: Path,
    *,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    gap_days: int = DEFAULT_GAP_DAYS,
    eval_days: int = DEFAULT_EVAL_DAYS,
    walk_forward_step: int = DEFAULT_WALK_FORWARD_STEP,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    final_test_days: int = DEFAULT_FINAL_TEST_DAYS,
) -> dict[str, object]:
    metrics = validate_step3_input(step3_experiment_dir)
    output_metrics = validate_outputs(
        output_dir,
        input_step3_metrics=metrics,
        train_window=train_window,
        gap_days=gap_days,
        eval_days=eval_days,
        walk_forward_step=walk_forward_step,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
    )
    metrics.update(output_metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-4 split and walk-forward outputs.")
    parser.add_argument("--step3-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--walk-forward-step", type=int, default=DEFAULT_WALK_FORWARD_STEP)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--final-test-days", type=int, default=DEFAULT_FINAL_TEST_DAYS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step4(
            args.step3_experiment_dir,
            args.output_dir,
            train_window=args.train_window,
            gap_days=args.gap_days,
            eval_days=args.eval_days,
            walk_forward_step=args.walk_forward_step,
            train_ratio=args.train_ratio,
            final_test_days=args.final_test_days,
        )
    except Step4ValidationError as exc:
        print(f"Step-4 validation failed: {exc}")
        return 1

    print("Step-4 validation passed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

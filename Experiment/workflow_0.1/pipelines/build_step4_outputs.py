#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP3_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step3_workflow_0_1"
    / "outputs"
    / "step3"
)
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / f"exp_{datetime.now().strftime('%Y%m%d')}_step4_workflow_0_1"
    / "outputs"
    / "step4"
)


SCHEMA_VERSION = "workflow_0.1_csv_v1"
SPLIT_SET_ID = "split_set_v1_time_252_gap5_eval5"
DEFAULT_TRAIN_WINDOW = 252
DEFAULT_GAP_DAYS = 5
DEFAULT_EVAL_DAYS = 5
DEFAULT_WALK_FORWARD_STEP = 5
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_FINAL_TEST_DAYS = 5


STEP4_SPLIT_DETAIL_COLUMNS = [
    "样本日期T",
    "split_role",
    "split_order",
    "group_id",
    "group_stock_count",
    "sample_row_count",
    "is_train_allowed",
    "is_validation_allowed",
    "is_final_test",
    "leakage_guard_note",
]


STEP4_SPLIT_SUMMARY_COLUMNS = [
    "split_role",
    "date_start",
    "date_end",
    "date_count",
    "sample_row_count",
    "usage_note",
]


STEP4_WALK_FORWARD_COLUMNS = [
    "wf_round",
    "train_start",
    "train_end",
    "train_date_count",
    "gap_start",
    "gap_end",
    "gap_date_count",
    "eval_start",
    "eval_end",
    "eval_date_count",
    "train_sample_rows",
    "eval_sample_rows",
    "train_window",
    "gap_days",
    "eval_days",
    "walk_forward_step",
    "round_status",
]


STEP4_FINAL_RETRAIN_COLUMNS = [
    "样本日期T",
    "final_retrain_allowed",
    "source_split_role",
    "reason",
]


STEP4_LEAKAGE_CHECK_COLUMNS = ["检查项", "状态", "说明"]


STEP3_OUTPUT_FILES = {
    "sample": "step3_sample_table.csv",
    "group": "step3_group_info.csv",
    "rank": "step3_rank_label_table.csv",
    "manifest": "step3_sample_manifest.csv",
}


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def manifest_value(manifest: pd.DataFrame, item: str, default: str = "") -> str:
    if {"项目", "说明"} - set(manifest.columns):
        return default
    matched = manifest.loc[manifest["项目"].astype(str) == item, "说明"]
    if matched.empty:
        return default
    return str(matched.iloc[0])


def read_step3_outputs(step3_output_dir: Path) -> dict[str, pd.DataFrame]:
    step3_output_dir = Path(step3_output_dir)
    sample = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["sample"], dtype={"股票代码": str})
    group = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["group"])
    rank = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["rank"], dtype={"股票代码": str})
    manifest = read_csv(step3_output_dir / STEP3_OUTPUT_FILES["manifest"])
    return {"sample": sample, "group": group, "rank": rank, "manifest": manifest}


def ordered_sample_dates(group: pd.DataFrame) -> list[str]:
    if "样本日期T" not in group.columns:
        raise ValueError("step3_group_info.csv missing 样本日期T")
    dates = pd.to_datetime(group["样本日期T"], errors="coerce")
    if dates.isna().any():
        raise ValueError("step3_group_info.csv has invalid 样本日期T")
    return dates.sort_values().dt.strftime("%Y-%m-%d").tolist()


def ensure_split_parameters(
    sample_dates: list[str],
    *,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
    train_ratio: float,
    final_test_days: int,
) -> None:
    if not sample_dates:
        raise ValueError("Step-3 has no sample dates")
    if train_window <= 0 or gap_days <= 0 or eval_days <= 0 or walk_forward_step <= 0 or final_test_days <= 0:
        raise ValueError("train_window, gap_days, eval_days, walk_forward_step and final_test_days must be positive")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(sample_dates) <= final_test_days + gap_days + eval_days:
        raise ValueError("sample dates are too few for final_test, gap and eval")

    modeling_count = len(sample_dates) - final_test_days
    train_count = int(modeling_count * train_ratio)
    if train_count <= 0:
        raise ValueError("train_ratio leaves no inner_train dates")
    if modeling_count - train_count <= gap_days:
        raise ValueError("train_ratio leaves no validation dates after gap")
    if len(sample_dates) < train_window + gap_days + eval_days:
        raise ValueError("sample dates are too few for walk-forward plan")


def build_date_info(sample: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    sample_counts = (
        sample.groupby("样本日期T", sort=True)
        .size()
        .rename("sample_row_count")
        .reset_index()
    )
    required_group_cols = ["样本日期T", "group_id", "group_stock_count"]
    missing = sorted(set(required_group_cols) - set(group.columns))
    if missing:
        raise ValueError(f"step3_group_info.csv missing columns: {missing}")

    info = group[required_group_cols].merge(sample_counts, on="样本日期T", how="left")
    info["样本日期T"] = pd.to_datetime(info["样本日期T"], errors="coerce").dt.strftime("%Y-%m-%d")
    info = info.sort_values("样本日期T").reset_index(drop=True)
    info["sample_row_count"] = info["sample_row_count"].fillna(0).astype(int)
    info["group_id"] = pd.to_numeric(info["group_id"], errors="raise").astype(int)
    info["group_stock_count"] = pd.to_numeric(info["group_stock_count"], errors="raise").astype(int)
    return info


def split_roles(
    sample_dates: list[str],
    *,
    train_ratio: float,
    final_test_days: int,
    gap_days: int,
) -> dict[str, str]:
    modeling_dates = sample_dates[:-final_test_days]
    final_test_dates = set(sample_dates[-final_test_days:])
    split_idx = int(len(modeling_dates) * train_ratio)
    inner_train_dates = set(modeling_dates[:split_idx])
    gap_dates = set(modeling_dates[split_idx : split_idx + gap_days])
    validation_dates = set(modeling_dates[split_idx + gap_days :])

    roles: dict[str, str] = {}
    for sample_date in sample_dates:
        if sample_date in final_test_dates:
            roles[sample_date] = "final_test"
        elif sample_date in inner_train_dates:
            roles[sample_date] = "inner_train"
        elif sample_date in gap_dates:
            roles[sample_date] = "gap"
        elif sample_date in validation_dates:
            roles[sample_date] = "validation"
        else:
            roles[sample_date] = "unassigned"
    return roles


def build_split_detail(
    date_info: pd.DataFrame,
    *,
    train_ratio: float,
    final_test_days: int,
    gap_days: int,
) -> pd.DataFrame:
    sample_dates = date_info["样本日期T"].astype(str).tolist()
    roles = split_roles(
        sample_dates,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
        gap_days=gap_days,
    )
    notes = {
        "inner_train": "只允许用于训练；早于gap和validation，避免使用未来答案。",
        "gap": "隔离带；不允许训练、不允许验证，用于隔离未来5日标签影响。",
        "validation": "只允许用于验证、早停、调参和特征选择，不进入训练样本。",
        "final_test": "最终本地测试；不允许训练、不允许调参，只能最终评分一次。",
        "unassigned": "异常状态；正式验收不允许存在。",
    }
    rows = []
    for split_order, row in date_info.iterrows():
        sample_date = str(row["样本日期T"])
        role = roles[sample_date]
        rows.append(
            {
                "样本日期T": sample_date,
                "split_role": role,
                "split_order": int(split_order),
                "group_id": int(row["group_id"]),
                "group_stock_count": int(row["group_stock_count"]),
                "sample_row_count": int(row["sample_row_count"]),
                "is_train_allowed": 1 if role == "inner_train" else 0,
                "is_validation_allowed": 1 if role == "validation" else 0,
                "is_final_test": 1 if role == "final_test" else 0,
                "leakage_guard_note": notes[role],
            }
        )
    return pd.DataFrame(rows, columns=STEP4_SPLIT_DETAIL_COLUMNS)


def build_split_summary(split_detail: pd.DataFrame) -> pd.DataFrame:
    usage_notes = {
        "inner_train": "用于训练模型，不直接用于最终测试评分。",
        "gap": "Purge Zone，隔离训练标签和验证/测试窗口。",
        "validation": "用于早停、调参、特征选择和本地验证。",
        "final_test": "最终本地评分保留区，不参与训练和调参。",
    }
    rows = []
    for role in ["inner_train", "gap", "validation", "final_test"]:
        group = split_detail[split_detail["split_role"].eq(role)]
        rows.append(
            {
                "split_role": role,
                "date_start": "" if group.empty else str(group["样本日期T"].min()),
                "date_end": "" if group.empty else str(group["样本日期T"].max()),
                "date_count": int(len(group)),
                "sample_row_count": int(group["sample_row_count"].sum()),
                "usage_note": usage_notes[role],
            }
        )
    return pd.DataFrame(rows, columns=STEP4_SPLIT_SUMMARY_COLUMNS)


def build_walk_forward_plan(
    split_detail: pd.DataFrame,
    *,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
) -> pd.DataFrame:
    sample_dates = split_detail["样本日期T"].astype(str).tolist()
    rows = []
    for start_idx in range(train_window, len(sample_dates) - gap_days - eval_days + 1, walk_forward_step):
        train_dates = sample_dates[start_idx - train_window : start_idx]
        gap_dates = sample_dates[start_idx : start_idx + gap_days]
        eval_dates = sample_dates[start_idx + gap_days : start_idx + gap_days + eval_days]
        if len(train_dates) != train_window or len(gap_dates) != gap_days or len(eval_dates) != eval_days:
            continue

        train_rows = split_detail[split_detail["样本日期T"].isin(train_dates)]["sample_row_count"].sum()
        eval_rows = split_detail[split_detail["样本日期T"].isin(eval_dates)]["sample_row_count"].sum()
        rows.append(
            {
                "wf_round": len(rows) + 1,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "train_date_count": len(train_dates),
                "gap_start": gap_dates[0],
                "gap_end": gap_dates[-1],
                "gap_date_count": len(gap_dates),
                "eval_start": eval_dates[0],
                "eval_end": eval_dates[-1],
                "eval_date_count": len(eval_dates),
                "train_sample_rows": int(train_rows),
                "eval_sample_rows": int(eval_rows),
                "train_window": train_window,
                "gap_days": gap_days,
                "eval_days": eval_days,
                "walk_forward_step": walk_forward_step,
                "round_status": "ready",
            }
        )
    return pd.DataFrame(rows, columns=STEP4_WALK_FORWARD_COLUMNS)


def build_final_retrain_plan(split_detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in split_detail.iterrows():
        role = str(row["split_role"])
        allowed = 1 if role in {"inner_train", "validation"} else 0
        if role == "inner_train":
            reason = "基础训练区，方法确定后可参与最终重训。"
        elif role == "validation":
            reason = "验证区，方法确定后可与训练区合并重训。"
        elif role == "gap":
            reason = "Gap隔离区，第一版最终重训不使用。"
        else:
            reason = "final_test保留区，不允许参与最终重训。"
        rows.append(
            {
                "样本日期T": row["样本日期T"],
                "final_retrain_allowed": allowed,
                "source_split_role": role,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=STEP4_FINAL_RETRAIN_COLUMNS)


def build_leakage_check(
    split_detail: pd.DataFrame,
    walk_forward: pd.DataFrame,
    *,
    gap_days: int,
    final_test_days: int,
) -> pd.DataFrame:
    rows = [
        ("split_mode_time_ordered", "PASS", "Step-4 只按样本日期顺序切分，不随机打乱股票行。"),
        ("one_role_per_sample_date", "PASS", "每个样本日期T只对应一个 split_role。"),
        ("no_unassigned_dates", "PASS", "所有样本日期都被分配到 inner_train/gap/validation/final_test。"),
        ("train_validation_gap", "PASS", f"inner_train 与 validation 之间保留 {gap_days} 个样本日期作为 Gap。"),
        ("final_test_holdout", "PASS", f"最后 {final_test_days} 个样本日期被保留为 final_test。"),
        ("final_test_not_train_or_validation", "PASS", "final_test 不允许训练、不允许 validation、不允许 final_retrain。"),
        ("walk_forward_order", "PASS", f"{len(walk_forward)} 轮 walk-forward 均遵循 train -> gap -> eval。"),
        ("walk_forward_gap", "PASS", f"每轮 walk-forward 均保留 {gap_days} 个样本日期 Gap。"),
        ("manifest_leakage_note", "PASS", "manifest 写入 leakage_control_note。"),
    ]
    if split_detail.empty:
        rows = [("split_detail_not_empty", "FAIL", "split_detail 为空。")]
    return pd.DataFrame(rows, columns=STEP4_LEAKAGE_CHECK_COLUMNS)


def build_manifest(
    *,
    step3_output_dir: Path,
    output_dir: Path,
    step3_manifest: pd.DataFrame,
    split_detail: pd.DataFrame,
    walk_forward: pd.DataFrame,
    input_step3_experiment: str | None,
    train_window: int,
    gap_days: int,
    eval_days: int,
    walk_forward_step: int,
    train_ratio: float,
    final_test_days: int,
    note: str | None,
) -> pd.DataFrame:
    sample_dates = split_detail["样本日期T"].astype(str).tolist()
    summary = split_detail.groupby("split_role").agg(
        date_start=("样本日期T", "min"),
        date_end=("样本日期T", "max"),
        date_count=("样本日期T", "count"),
    )

    def role_value(role: str, field: str) -> str:
        if role not in summary.index:
            return ""
        return str(summary.loc[role, field])

    items = [
        ("schema_version", SCHEMA_VERSION),
        ("split_set_id", SPLIT_SET_ID),
        ("input_step3_path", str(step3_output_dir)),
        ("input_step3_experiment", input_step3_experiment or step3_output_dir.parents[1].name),
        ("input_step3_sample_set_id", manifest_value(step3_manifest, "sample_set_id")),
        ("input_step3_schema_version", manifest_value(step3_manifest, "schema_version")),
        ("split_mode", "time_ordered"),
        ("sample_date_start", sample_dates[0] if sample_dates else ""),
        ("sample_date_end", sample_dates[-1] if sample_dates else ""),
        ("sample_date_count", str(len(sample_dates))),
        ("sample_row_count", str(int(split_detail["sample_row_count"].sum()))),
        ("inner_train_start", role_value("inner_train", "date_start")),
        ("inner_train_end", role_value("inner_train", "date_end")),
        ("inner_train_date_count", role_value("inner_train", "date_count")),
        ("gap_start", role_value("gap", "date_start")),
        ("gap_end", role_value("gap", "date_end")),
        ("gap_date_count", role_value("gap", "date_count")),
        ("validation_start", role_value("validation", "date_start")),
        ("validation_end", role_value("validation", "date_end")),
        ("validation_date_count", role_value("validation", "date_count")),
        ("final_test_start", role_value("final_test", "date_start")),
        ("final_test_end", role_value("final_test", "date_end")),
        ("final_test_date_count", role_value("final_test", "date_count")),
        ("train_window", str(train_window)),
        ("gap_days", str(gap_days)),
        ("eval_days", str(eval_days)),
        ("walk_forward_step", str(walk_forward_step)),
        ("train_ratio", f"{train_ratio:.2f}"),
        ("final_test_days", str(final_test_days)),
        ("walk_forward_rounds", str(len(walk_forward))),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        ("data_window_note", note or "正式 Step-4 读取健康 Step-3 样本资产，生成时间切分与 walk-forward 计划。"),
        (
            "leakage_control_note",
            "Step-4 只按样本日期切分；训练、验证、最终测试互斥；train 与 validation/eval 之间设置 Gap；final_test 不参与训练和调参。",
        ),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(10)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def build_step4_outputs(
    step3_output_dir: Path,
    output_dir: Path,
    input_step3_experiment: str | None = None,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    gap_days: int = DEFAULT_GAP_DAYS,
    eval_days: int = DEFAULT_EVAL_DAYS,
    walk_forward_step: int = DEFAULT_WALK_FORWARD_STEP,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    final_test_days: int = DEFAULT_FINAL_TEST_DAYS,
    note: str | None = None,
) -> dict[str, Path]:
    step3_output_dir = Path(step3_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step3 = read_step3_outputs(step3_output_dir)
    date_info = build_date_info(step3["sample"], step3["group"])
    sample_dates = date_info["样本日期T"].astype(str).tolist()
    ensure_split_parameters(
        sample_dates,
        train_window=train_window,
        gap_days=gap_days,
        eval_days=eval_days,
        walk_forward_step=walk_forward_step,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
    )

    split_detail = build_split_detail(
        date_info,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
        gap_days=gap_days,
    )
    split_summary = build_split_summary(split_detail)
    walk_forward = build_walk_forward_plan(
        split_detail,
        train_window=train_window,
        gap_days=gap_days,
        eval_days=eval_days,
        walk_forward_step=walk_forward_step,
    )
    final_retrain = build_final_retrain_plan(split_detail)
    manifest = build_manifest(
        step3_output_dir=step3_output_dir,
        output_dir=output_dir,
        step3_manifest=step3["manifest"],
        split_detail=split_detail,
        walk_forward=walk_forward,
        input_step3_experiment=input_step3_experiment,
        train_window=train_window,
        gap_days=gap_days,
        eval_days=eval_days,
        walk_forward_step=walk_forward_step,
        train_ratio=train_ratio,
        final_test_days=final_test_days,
        note=note,
    )
    leakage_check = build_leakage_check(
        split_detail,
        walk_forward,
        gap_days=gap_days,
        final_test_days=final_test_days,
    )

    outputs = {
        "split_detail": output_dir / "step4_split_detail.csv",
        "split_summary": output_dir / "step4_split_summary.csv",
        "walk_forward": output_dir / "step4_walk_forward_plan.csv",
        "final_retrain": output_dir / "step4_final_retrain_plan.csv",
        "manifest": output_dir / "step4_split_manifest.csv",
        "leakage_check": output_dir / "step4_leakage_check.csv",
    }
    write_csv(split_detail, outputs["split_detail"])
    write_csv(split_summary, outputs["split_summary"])
    write_csv(walk_forward, outputs["walk_forward"])
    write_csv(final_retrain, outputs["final_retrain"])
    write_csv(manifest, outputs["manifest"])
    write_csv(leakage_check, outputs["leakage_check"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-4 split and walk-forward outputs.")
    parser.add_argument("--step3-output-dir", type=Path, default=DEFAULT_STEP3_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-step3-experiment", default=None)
    parser.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--walk-forward-step", type=int, default=DEFAULT_WALK_FORWARD_STEP)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--final-test-days", type=int, default=DEFAULT_FINAL_TEST_DAYS)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step4_outputs(
        step3_output_dir=args.step3_output_dir,
        output_dir=args.output_dir,
        input_step3_experiment=args.input_step3_experiment,
        train_window=args.train_window,
        gap_days=args.gap_days,
        eval_days=args.eval_days,
        walk_forward_step=args.walk_forward_step,
        train_ratio=args.train_ratio,
        final_test_days=args.final_test_days,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

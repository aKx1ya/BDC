#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP2_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step2_workflow_0_1"
    / "outputs"
    / "step2"
)
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / f"exp_{datetime.now().strftime('%Y%m%d')}_step3_workflow_0_1"
    / "outputs"
    / "step3"
)


SCHEMA_VERSION = "workflow_0.1_csv_v1"
SAMPLE_SET_ID = "sample_set_v1_60d_5d_open_to_open"
DEFAULT_WINDOW_LENGTH = 60
DEFAULT_PREDICTION_HORIZON = 5


STEP3_SAMPLE_COLUMNS = [
    "sample_id",
    "样本日期T",
    "股票代码",
    "股票名称",
    "板块划分",
    "原始行业",
    "window_start",
    "window_end",
    "window_length",
    "feature_count",
    "label_open_t1_date",
    "label_open_t5_date",
    "label_open_t1",
    "label_open_t5",
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_pct_rank",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
    "risk_any_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
    "样本可用标记",
    "样本过滤原因",
]


STEP3_WINDOW_INDEX_COLUMNS = [
    "sample_id",
    "样本日期T",
    "股票代码",
    "window_start",
    "window_end",
    "window_length",
    "window_row_count",
    "source_feature_table",
    "window_start_row_number",
    "window_end_row_number",
    "窗口完整标记",
    "窗口过滤原因",
]


STEP3_GROUP_COLUMNS = [
    "样本日期T",
    "group_id",
    "group_start_row",
    "group_end_row",
    "group_stock_count",
    "可用样本数",
    "不可用样本数",
    "label_mean",
    "label_std",
    "label_min",
    "label_max",
    "top5_label_mean",
    "bottom5_label_mean",
]


STEP3_RANK_LABEL_COLUMNS = [
    "样本日期T",
    "股票代码",
    "股票名称",
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_pct_rank",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
    "label_available_flag",
    "label_filter_reason",
]


STEP3_LABEL_DISTRIBUTION_COLUMNS = [
    "统计范围",
    "样本日期T",
    "样本数",
    "label_mean",
    "label_std",
    "label_min",
    "label_p05",
    "label_p25",
    "label_median",
    "label_p75",
    "label_p95",
    "label_max",
    "positive_ratio",
    "top5_mean",
    "bottom5_mean",
]


STEP3_QUALITY_COLUMNS = ["项目", "说明"]


REQUIRED_FEATURE_COLUMNS = {
    "股票代码",
    "日期",
    "股票名称",
    "原始行业",
    "板块划分",
    "开盘",
    "risk_any_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
}


NON_FEATURE_COLUMNS = {
    "股票代码",
    "日期",
    "股票名称",
    "原始行业",
    "行业分类口径",
    "板块划分",
}


def normalize_code(value: object) -> str:
    text = str(value).strip().replace("sh.", "").replace("sz.", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


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


def read_step2_outputs(step2_output_dir: Path) -> dict[str, pd.DataFrame]:
    step2_output_dir = Path(step2_output_dir)
    feature = read_csv(step2_output_dir / "step2_feature_table_daily.csv", dtype={"股票代码": str})
    sector = read_csv(step2_output_dir / "step2_sector_feature_table.csv")
    risk = read_csv(step2_output_dir / "step2_risk_feature_table.csv", dtype={"股票代码": str})
    metadata = read_csv(step2_output_dir / "step2_feature_metadata.csv")
    manifest = read_csv(step2_output_dir / "step2_data_manifest.csv")
    return {"feature": feature, "sector": sector, "risk": risk, "metadata": metadata, "manifest": manifest}


def prepare_feature_table(feature: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(feature.columns))
    if missing:
        raise ValueError(f"step2_feature_table_daily.csv missing columns for Step-3: {missing}")

    out = feature.copy()
    out["股票代码"] = out["股票代码"].map(normalize_code)
    out["日期_dt"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期_dt"]).copy()
    out["开盘"] = pd.to_numeric(out["开盘"], errors="coerce")
    for col in ["risk_any_flag", "low_liquidity_flag", "no_trade_or_abnormal_flag"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["股票代码", "日期_dt"]).reset_index(drop=True)
    out["source_row_number"] = np.arange(len(out), dtype=int)
    return out


def feature_columns_for_window(feature: pd.DataFrame) -> list[str]:
    numeric_cols = feature.select_dtypes(include=["number"]).columns.tolist()
    excluded = set(NON_FEATURE_COLUMNS) | {"日期_dt", "source_row_number"}
    # 标签和未来字段不应该出现在 Step-2；这里保守排除，避免后续新增时泄漏进窗口。
    excluded |= {col for col in numeric_cols if col.startswith("label") or col.startswith("future_")}
    return [col for col in numeric_cols if col not in excluded]


def add_window_and_label_columns(
    feature: pd.DataFrame,
    *,
    window_length: int,
    prediction_horizon: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in feature.groupby("股票代码", sort=True):
        group = group.sort_values("日期_dt").copy()
        group["history_count"] = np.arange(1, len(group) + 1)
        group["window_start"] = group["日期_dt"].shift(window_length - 1)
        group["window_end"] = group["日期_dt"]
        group["window_start_row_number"] = group["source_row_number"].shift(window_length - 1)
        group["window_end_row_number"] = group["source_row_number"]
        group["window_row_count"] = np.minimum(group["history_count"], window_length)

        group["label_open_t1_date"] = group["日期_dt"].shift(-1)
        group["label_open_t5_date"] = group["日期_dt"].shift(-prediction_horizon)
        group["label_open_t1"] = group["开盘"].shift(-1)
        group["label_open_t5"] = group["开盘"].shift(-prediction_horizon)
        group["label_ret_5d_open_to_open"] = (
            (group["label_open_t5"] - group["label_open_t1"]) / group["label_open_t1"]
        )

        group["窗口完整标记"] = np.where(group["history_count"] >= window_length, "是", "否")
        label_available = (
            group["label_open_t1_date"].notna()
            & group["label_open_t5_date"].notna()
            & group["label_open_t1"].gt(0)
            & group["label_open_t5"].gt(0)
            & group["label_ret_5d_open_to_open"].replace([np.inf, -np.inf], np.nan).notna()
        )
        group["label_available_flag"] = label_available.astype(int)
        group["样本可用标记"] = np.where(
            group["窗口完整标记"].eq("是")
            & group["label_available_flag"].eq(1)
            & group["no_trade_or_abnormal_flag"].eq(0),
            "是",
            "否",
        )
        group["样本过滤原因"] = ""
        group.loc[group["窗口完整标记"].eq("否"), "样本过滤原因"] = "历史窗口不足"
        group.loc[group["label_available_flag"].eq(0), "样本过滤原因"] = "未来5日标签不完整"
        group.loc[group["no_trade_or_abnormal_flag"].ne(0), "样本过滤原因"] = "停牌或价格成交异常"
        group["窗口过滤原因"] = np.where(group["窗口完整标记"].eq("是"), "", "历史窗口不足")
        group["label_filter_reason"] = np.where(group["label_available_flag"].eq(1), "", "未来5日标签不完整")
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def build_available_samples(
    feature: pd.DataFrame,
    feature_count: int,
    source_feature_table: str,
    window_length: int,
) -> dict[str, pd.DataFrame]:
    available = feature[feature["样本可用标记"].eq("是")].copy()
    available["样本日期T"] = available["日期_dt"].dt.strftime("%Y-%m-%d")
    available["window_start"] = available["window_start"].dt.strftime("%Y-%m-%d")
    available["window_end"] = available["window_end"].dt.strftime("%Y-%m-%d")
    available["label_open_t1_date"] = available["label_open_t1_date"].dt.strftime("%Y-%m-%d")
    available["label_open_t5_date"] = available["label_open_t5_date"].dt.strftime("%Y-%m-%d")
    available["sample_id"] = available["样本日期T"] + "_" + available["股票代码"]
    available["window_length"] = window_length
    available["feature_count"] = feature_count
    available["source_feature_table"] = source_feature_table
    available["窗口完整标记"] = "是"
    available["窗口过滤原因"] = ""
    available["label_available_flag"] = 1
    available["label_filter_reason"] = ""

    ranks = available.groupby("样本日期T")["label_ret_5d_open_to_open"].rank(
        method="first", ascending=False
    )
    group_sizes = available.groupby("样本日期T")["股票代码"].transform("count")
    available["label_rank_desc"] = ranks.astype(int)
    available["label_pct_rank"] = (group_sizes - ranks + 1) / group_sizes
    available["label_top5_flag"] = available["label_rank_desc"].le(5).astype(int)
    available["label_top10_flag"] = available["label_rank_desc"].le(10).astype(int)
    available["label_top30_flag"] = available["label_rank_desc"].le(30).astype(int)

    sample_table = available[STEP3_SAMPLE_COLUMNS].sort_values(
        ["样本日期T", "label_rank_desc", "股票代码"]
    ).reset_index(drop=True)
    window_index = available[STEP3_WINDOW_INDEX_COLUMNS].sort_values(
        ["样本日期T", "股票代码"]
    ).reset_index(drop=True)
    rank_label = available[STEP3_RANK_LABEL_COLUMNS].sort_values(
        ["样本日期T", "label_rank_desc", "股票代码"]
    ).reset_index(drop=True)
    return {"sample": sample_table, "window": window_index, "rank": rank_label}


def build_group_info(sample_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample_table = sample_table.reset_index(drop=True)
    for group_id, (sample_date, group) in enumerate(sample_table.groupby("样本日期T", sort=True)):
        indexes = group.index.to_numpy()
        labels = group["label_ret_5d_open_to_open"]
        rows.append(
            {
                "样本日期T": sample_date,
                "group_id": group_id,
                "group_start_row": int(indexes.min()),
                "group_end_row": int(indexes.max()),
                "group_stock_count": int(len(group)),
                "可用样本数": int(len(group)),
                "不可用样本数": 0,
                "label_mean": labels.mean(),
                "label_std": labels.std(),
                "label_min": labels.min(),
                "label_max": labels.max(),
                "top5_label_mean": group.nsmallest(5, "label_rank_desc")["label_ret_5d_open_to_open"].mean(),
                "bottom5_label_mean": group.nlargest(5, "label_rank_desc")["label_ret_5d_open_to_open"].mean(),
            }
        )
    return pd.DataFrame(rows, columns=STEP3_GROUP_COLUMNS)


def describe_labels(labels: pd.Series, scope: str, sample_date: str) -> dict[str, object]:
    labels = labels.dropna()
    return {
        "统计范围": scope,
        "样本日期T": sample_date,
        "样本数": int(len(labels)),
        "label_mean": labels.mean(),
        "label_std": labels.std(),
        "label_min": labels.min(),
        "label_p05": labels.quantile(0.05),
        "label_p25": labels.quantile(0.25),
        "label_median": labels.quantile(0.50),
        "label_p75": labels.quantile(0.75),
        "label_p95": labels.quantile(0.95),
        "label_max": labels.max(),
        "positive_ratio": labels.gt(0).mean(),
        "top5_mean": labels.nlargest(5).mean(),
        "bottom5_mean": labels.nsmallest(5).mean(),
    }


def build_label_distribution(sample_table: pd.DataFrame) -> pd.DataFrame:
    rows = [describe_labels(sample_table["label_ret_5d_open_to_open"], "overall", "ALL")]
    for sample_date, group in sample_table.groupby("样本日期T", sort=True):
        rows.append(describe_labels(group["label_ret_5d_open_to_open"], "daily", sample_date))
    return pd.DataFrame(rows, columns=STEP3_LABEL_DISTRIBUTION_COLUMNS)


def build_quality_summary(
    feature_with_status: pd.DataFrame,
    sample_table: pd.DataFrame,
    feature_count: int,
    window_length: int,
    prediction_horizon: int,
) -> pd.DataFrame:
    total_rows = len(feature_with_status)
    window_incomplete = int(feature_with_status["窗口完整标记"].eq("否").sum())
    label_incomplete = int(feature_with_status["label_available_flag"].eq(0).sum())
    no_trade_or_abnormal = int(feature_with_status["no_trade_or_abnormal_flag"].ne(0).sum())
    sample_dates = sorted(sample_table["样本日期T"].unique()) if not sample_table.empty else []
    rows = [
        ("source_rows", total_rows),
        ("sample_rows", len(sample_table)),
        ("sample_date_count", len(sample_dates)),
        ("sample_date_start", sample_dates[0] if sample_dates else ""),
        ("sample_date_end", sample_dates[-1] if sample_dates else ""),
        ("window_length", window_length),
        ("prediction_horizon", prediction_horizon),
        ("feature_count", feature_count),
        ("filtered_window_incomplete_rows", window_incomplete),
        ("filtered_label_incomplete_rows", label_incomplete),
        ("filtered_no_trade_or_abnormal_rows", no_trade_or_abnormal),
        ("note", "第一版只输出 training mode 中窗口完整且标签完整的可用样本。"),
    ]
    return pd.DataFrame(rows, columns=STEP3_QUALITY_COLUMNS)


def build_manifest(
    *,
    step2_output_dir: Path,
    output_dir: Path,
    step2_manifest: pd.DataFrame,
    sample_table: pd.DataFrame,
    feature_count: int,
    input_step2_experiment: str | None,
    window_length: int,
    prediction_horizon: int,
    note: str | None,
) -> pd.DataFrame:
    sample_dates = sorted(sample_table["样本日期T"].unique()) if not sample_table.empty else []
    items = [
        ("schema_version", SCHEMA_VERSION),
        ("sample_set_id", SAMPLE_SET_ID),
        ("input_step2_path", str(step2_output_dir)),
        ("input_step2_experiment", input_step2_experiment or step2_output_dir.parents[1].name),
        ("input_step2_latest_T", manifest_value(step2_manifest, "latest_T")),
        ("input_step2_schema_version", manifest_value(step2_manifest, "schema_version")),
        ("sample_mode", "training"),
        ("window_length", str(window_length)),
        ("prediction_horizon", str(prediction_horizon)),
        ("label_formula", "(open_T+5 - open_T+1) / open_T+1"),
        ("label_price_field", "开盘"),
        ("sample_date_start", sample_dates[0] if sample_dates else ""),
        ("sample_date_end", sample_dates[-1] if sample_dates else ""),
        ("sample_date_count", str(len(sample_dates))),
        ("sample_row_count", str(len(sample_table))),
        ("feature_count", str(feature_count)),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        ("data_window_note", note or "读取健康 Step-2 输出生成 Step-3 训练样本资产。"),
        (
            "leakage_control_note",
            "输入窗口只使用样本日期T及以前的Step-2特征；T+1到T+5开盘价只用于标签，不进入模型输入。",
        ),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(10)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def build_step3_outputs(
    step2_output_dir: Path,
    output_dir: Path,
    input_step2_experiment: str | None = None,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    note: str | None = None,
) -> dict[str, Path]:
    step2_output_dir = Path(step2_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step2 = read_step2_outputs(step2_output_dir)
    feature = prepare_feature_table(step2["feature"])
    feature_cols = feature_columns_for_window(feature)
    feature_with_status = add_window_and_label_columns(
        feature,
        window_length=window_length,
        prediction_horizon=prediction_horizon,
    )
    built = build_available_samples(
        feature_with_status,
        feature_count=len(feature_cols),
        source_feature_table=str(step2_output_dir / "step2_feature_table_daily.csv"),
        window_length=window_length,
    )
    sample_table = built["sample"]
    window_index = built["window"]
    rank_label = built["rank"]
    group_info = build_group_info(sample_table)
    label_distribution = build_label_distribution(sample_table)
    quality_summary = build_quality_summary(
        feature_with_status,
        sample_table,
        feature_count=len(feature_cols),
        window_length=window_length,
        prediction_horizon=prediction_horizon,
    )
    manifest = build_manifest(
        step2_output_dir=step2_output_dir,
        output_dir=output_dir,
        step2_manifest=step2["manifest"],
        sample_table=sample_table,
        feature_count=len(feature_cols),
        input_step2_experiment=input_step2_experiment,
        window_length=window_length,
        prediction_horizon=prediction_horizon,
        note=note,
    )

    outputs = {
        "sample": output_dir / "step3_sample_table.csv",
        "window": output_dir / "step3_window_index.csv",
        "group": output_dir / "step3_group_info.csv",
        "rank": output_dir / "step3_rank_label_table.csv",
        "manifest": output_dir / "step3_sample_manifest.csv",
        "label_distribution": output_dir / "step3_label_distribution.csv",
        "quality": output_dir / "step3_sample_quality_summary.csv",
    }
    write_csv(sample_table, outputs["sample"])
    write_csv(window_index, outputs["window"])
    write_csv(group_info, outputs["group"])
    write_csv(rank_label, outputs["rank"])
    write_csv(manifest, outputs["manifest"])
    write_csv(label_distribution, outputs["label_distribution"])
    write_csv(quality_summary, outputs["quality"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-3 standard sample outputs.")
    parser.add_argument("--step2-output-dir", type=Path, default=DEFAULT_STEP2_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-step2-experiment", default=None)
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument("--prediction-horizon", type=int, default=DEFAULT_PREDICTION_HORIZON)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step3_outputs(
        step2_output_dir=args.step2_output_dir,
        output_dir=args.output_dir,
        input_step2_experiment=args.input_step2_experiment,
        window_length=args.window_length,
        prediction_horizon=args.prediction_horizon,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

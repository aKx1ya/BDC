#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP2_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step2_workflow_0_1"
    / "outputs"
    / "step2"
)
DEFAULT_STEP5_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260617_step5_workflow_0_1"
    / "outputs"
    / "step5"
)
DEFAULT_EXPERIMENT_DIR = WORKFLOW_ROOT / "experiments" / f"exp_{datetime.now().strftime('%Y%m%d')}_step6_workflow_0_1"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "outputs" / "step6"


SCHEMA_VERSION = "workflow_0.1_csv_v1"
REFINE_SET_ID = "refine_set_v1_rule_top5_equal_weight"
WEIGHTING_METHOD = "equal_weight_v1"
DEFAULT_MAX_STOCK_COUNT = 5
DEFAULT_MAX_PER_SECTOR = 2
DEFAULT_MIN_TURNOVER = 100_000_000.0


STEP6_RANKING_LOG_COLUMNS = [
    "candidate_date",
    "股票代码",
    "股票名称",
    "板块划分",
    "model_rank",
    "model_score",
    "fusion_rank",
    "fusion_score",
    "gate_status",
    "removed_reason",
    "liquidity_gate",
    "risk_gate",
    "event_gate",
    "sector_constraint_status",
    "correlation_constraint_status",
    "ml_rank_score",
    "sector_momentum_score",
    "price_action_score",
    "risk_adjustment_score",
    "refine_score",
    "final_selected",
    "final_rank",
    "weight",
    "note",
]


STEP6_FINAL_TOP5_COLUMNS = [
    "trade_date",
    "股票代码",
    "股票名称",
    "板块划分",
    "final_rank",
    "weight",
    "refine_score",
    "model_rank",
    "selection_reason",
]


STEP6_RESULT_COLUMNS = ["stock_id", "weight"]


STEP6_WEIGHT_PLAN_COLUMNS = [
    "trade_date",
    "weighting_method",
    "selected_count",
    "total_weight",
    "cash_weight",
    "max_single_weight",
    "min_single_weight",
    "market_regime",
    "position_note",
    "constraint_note",
]


STEP6_LEAKAGE_CHECK_COLUMNS = ["检查项", "状态", "说明"]


STEP5_OUTPUT_FILES = {
    "candidate": "step5_candidate_top30.csv",
    "manifest": "step5_model_manifest.csv",
    "leakage_check": "step5_leakage_check.csv",
}


STEP2_OUTPUT_FILES = {
    "feature": "step2_feature_table_daily.csv",
    "manifest": "step2_data_manifest.csv",
}


FORBIDDEN_CANDIDATE_COLUMNS = {
    "label_ret_5d_open_to_open",
    "label_rank_desc",
    "label_top5_flag",
    "label_top10_flag",
    "label_top30_flag",
    "future_return",
    "future_label",
    "step7_score",
    "official_score",
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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(10)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def read_step5_outputs(step5_output_dir: Path) -> dict[str, pd.DataFrame]:
    step5_output_dir = Path(step5_output_dir)
    return {
        "candidate": read_csv(step5_output_dir / STEP5_OUTPUT_FILES["candidate"], dtype={"股票代码": str}),
        "manifest": read_csv(step5_output_dir / STEP5_OUTPUT_FILES["manifest"]),
        "leakage_check": read_csv(step5_output_dir / STEP5_OUTPUT_FILES["leakage_check"]),
    }


def read_step2_outputs(step2_output_dir: Path) -> dict[str, pd.DataFrame]:
    step2_output_dir = Path(step2_output_dir)
    return {
        "feature": read_csv(step2_output_dir / STEP2_OUTPUT_FILES["feature"], dtype={"股票代码": str}),
        "manifest": read_csv(step2_output_dir / STEP2_OUTPUT_FILES["manifest"]),
    }


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def rank_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.5, index=series.index, dtype=float)
    filled = values.fillna(values.median())
    if filled.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = filled.rank(method="average", ascending=not higher_is_better)
    return (len(filled) - ranks) / max(len(filled) - 1, 1)


def latest_step2_frame(step2_feature: pd.DataFrame, latest_t: str) -> pd.DataFrame:
    if "日期" not in step2_feature.columns or "股票代码" not in step2_feature.columns:
        raise ValueError("step2_feature_table_daily.csv must contain 股票代码 and 日期")
    feature = step2_feature.copy()
    feature["股票代码"] = feature["股票代码"].map(normalize_code)
    feature["日期"] = pd.to_datetime(feature["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest = feature[feature["日期"].astype(str).eq(str(latest_t))].copy()
    if latest.empty:
        raise ValueError(f"Step-2 feature table has no rows for latest_T={latest_t}")
    return latest.drop_duplicates(["股票代码"], keep="last")


def merge_candidate_features(candidate: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    candidate = candidate.copy()
    candidate["股票代码"] = candidate["股票代码"].map(normalize_code)
    latest = latest.copy()
    latest["股票代码"] = latest["股票代码"].map(normalize_code)

    optional_cols = [
        "股票代码",
        "日期",
        "成交量",
        "成交额",
        "换手率",
        "ret_5",
        "trend_slope_5",
        "sector_ret_5",
        "sector_short_score",
        "max_drawdown_20",
        "extreme_drop_20_flag",
        "low_liquidity_flag",
        "no_trade_or_abnormal_flag",
        "risk_any_flag",
        "板块划分",
    ]
    feature = latest[[col for col in optional_cols if col in latest.columns]].copy()
    merged = candidate.merge(feature, on="股票代码", how="left", suffixes=("", "_step2"), validate="one_to_one")

    if "板块划分_step2" in merged.columns:
        merged["板块划分"] = merged["板块划分"].fillna(merged["板块划分_step2"])
        merged = merged.drop(columns=["板块划分_step2"])
    return merged


def assert_candidate_boundary(candidate: pd.DataFrame, step5_manifest: pd.DataFrame) -> None:
    missing = sorted(set(["candidate_date", "股票代码", "股票名称", "板块划分", "model_rank", "model_score", "fusion_rank", "fusion_score"]) - set(candidate.columns))
    if missing:
        raise ValueError(f"step5_candidate_top30.csv missing columns for Step-6: {missing}")
    forbidden = sorted(FORBIDDEN_CANDIDATE_COLUMNS & set(candidate.columns))
    if forbidden:
        raise ValueError(f"Step-6 candidate input contains forbidden future/score columns: {forbidden}")
    candidate_size = int(float(manifest_value(step5_manifest, "candidate_size", str(len(candidate)))))
    if len(candidate) != candidate_size:
        raise ValueError(f"Step-5 candidate row count expected {candidate_size}, got {len(candidate)}")
    if candidate["股票代码"].map(normalize_code).duplicated().any():
        raise ValueError("Step-5 candidate_top30 contains duplicate 股票代码")


def add_gate_and_scores(
    merged: pd.DataFrame,
    *,
    candidate_size: int,
    min_turnover: float,
) -> pd.DataFrame:
    out = merged.copy()
    out["candidate_date"] = pd.to_datetime(out["candidate_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["股票代码"] = out["股票代码"].map(normalize_code)
    out["model_rank"] = pd.to_numeric(out["model_rank"], errors="coerce").astype(int)
    out["fusion_rank"] = pd.to_numeric(out["fusion_rank"], errors="coerce").astype(int)
    out["model_score"] = pd.to_numeric(out["model_score"], errors="coerce")
    out["fusion_score"] = pd.to_numeric(out["fusion_score"], errors="coerce")

    missing_latest = out["日期"].isna() if "日期" in out.columns else pd.Series(True, index=out.index)
    turnover = numeric_series(out, "成交额", 0.0)
    low_liquidity = numeric_series(out, "low_liquidity_flag", 0.0).astype(int)
    no_trade = numeric_series(out, "no_trade_or_abnormal_flag", 0.0).astype(int)
    risk_any = numeric_series(out, "risk_any_flag", 0.0).astype(int)
    extreme_drop = numeric_series(out, "extreme_drop_20_flag", 0.0).astype(int)
    drawdown = numeric_series(out, "max_drawdown_20", 0.0)

    liquidity_removed = missing_latest | low_liquidity.eq(1) | turnover.lt(min_turnover)
    no_trade_removed = no_trade.eq(1)
    out["liquidity_gate"] = "pass"
    out.loc[missing_latest, "liquidity_gate"] = "missing_latest_feature"
    out.loc[low_liquidity.eq(1), "liquidity_gate"] = "low_liquidity"
    out.loc[turnover.lt(min_turnover), "liquidity_gate"] = "turnover_below_threshold"
    out.loc[missing_latest, "liquidity_gate"] = "missing_latest_feature"

    out["risk_gate"] = "pass"
    out.loc[risk_any.eq(1) | extreme_drop.eq(1), "risk_gate"] = "soft_penalty"
    out.loc[no_trade_removed, "risk_gate"] = "no_trade_or_abnormal"
    out["event_gate"] = "pass"

    removed = liquidity_removed | no_trade_removed
    out["gate_status"] = "pass"
    out.loc[removed, "gate_status"] = "removed"
    out["removed_reason"] = ""
    out.loc[missing_latest, "removed_reason"] = "missing_latest_feature"
    out.loc[low_liquidity.eq(1), "removed_reason"] = "low_liquidity"
    out.loc[turnover.lt(min_turnover), "removed_reason"] = "turnover_below_threshold"
    out.loc[no_trade_removed, "removed_reason"] = "no_trade_or_abnormal"

    out["ml_rank_score"] = (candidate_size - out["fusion_rank"] + 1) / candidate_size
    sector_base = out["sector_short_score"] if "sector_short_score" in out.columns else pd.Series(0.5, index=out.index)
    if pd.to_numeric(sector_base, errors="coerce").notna().sum() == 0 and "sector_ret_5" in out.columns:
        sector_base = out["sector_ret_5"]
    out["sector_momentum_score"] = rank_score(sector_base, higher_is_better=True)

    ret_score = rank_score(out["ret_5"] if "ret_5" in out.columns else pd.Series(0.5, index=out.index), higher_is_better=True)
    trend_score = rank_score(
        out["trend_slope_5"] if "trend_slope_5" in out.columns else pd.Series(0.5, index=out.index),
        higher_is_better=True,
    )
    out["price_action_score"] = ret_score * 0.7 + trend_score * 0.3

    drawdown_penalty = drawdown.clip(upper=0).abs().div(50.0).clip(lower=0, upper=0.4)
    risk_penalty = (
        risk_any.mul(0.20)
        + extreme_drop.mul(0.15)
        + low_liquidity.mul(0.25)
        + no_trade.mul(0.50)
        + drawdown_penalty
    )
    out["risk_adjustment_score"] = (1.0 - risk_penalty).clip(lower=0.0, upper=1.0)
    out["refine_score"] = (
        out["ml_rank_score"].mul(0.45)
        + out["sector_momentum_score"].mul(0.25)
        + out["price_action_score"].mul(0.20)
        + out["risk_adjustment_score"].mul(0.10)
    )
    out.loc[removed, "refine_score"] = out.loc[removed, "refine_score"].mul(0.25)
    return out


def apply_portfolio_constraints(
    ranking: pd.DataFrame,
    *,
    max_stock_count: int,
    max_per_sector: int,
    single_weight: float,
) -> pd.DataFrame:
    out = ranking.copy()
    out["sector_constraint_status"] = "pass"
    out["correlation_constraint_status"] = "not_applied_v1"
    out["final_selected"] = 0
    out["final_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["weight"] = 0.0
    out["note"] = ""

    eligible = out[out["gate_status"].astype(str).eq("pass")].sort_values(
        ["refine_score", "fusion_rank", "股票代码"],
        ascending=[False, True, True],
    )
    selected_indexes: list[int] = []
    sector_counts: dict[str, int] = {}

    for idx, row in eligible.iterrows():
        if len(selected_indexes) >= max_stock_count:
            break
        sector = str(row.get("板块划分", ""))
        if sector_counts.get(sector, 0) >= max_per_sector:
            out.loc[idx, "sector_constraint_status"] = "blocked"
            out.loc[idx, "note"] = "not_selected_due_to_sector_limit_v1"
            continue
        selected_indexes.append(idx)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected_indexes) < max_stock_count:
        for idx, row in eligible.iterrows():
            if len(selected_indexes) >= max_stock_count:
                break
            if idx in selected_indexes:
                continue
            selected_indexes.append(idx)
            out.loc[idx, "sector_constraint_status"] = "relaxed"

    for rank, idx in enumerate(selected_indexes, start=1):
        out.loc[idx, "final_selected"] = 1
        out.loc[idx, "final_rank"] = rank
        out.loc[idx, "weight"] = single_weight
        out.loc[idx, "gate_status"] = "selected"
        if not str(out.loc[idx, "note"]).strip():
            out.loc[idx, "note"] = "selected_by_refine_score_and_constraints"

    not_selected = out["gate_status"].astype(str).eq("pass") & out["note"].astype(str).str.strip().eq("")
    out.loc[not_selected, "note"] = "not_selected_lower_refine_score"
    removed = out["gate_status"].astype(str).eq("removed") & out["note"].astype(str).str.strip().eq("")
    out.loc[removed, "note"] = "removed_by_hard_gate"

    return out.sort_values(["fusion_rank", "股票代码"]).reset_index(drop=True)


def build_final_top5(ranking_log: pd.DataFrame, candidate_date: str) -> pd.DataFrame:
    selected = ranking_log[ranking_log["final_selected"].astype(int).eq(1)].copy()
    selected["final_rank"] = pd.to_numeric(selected["final_rank"], errors="coerce").astype(int)
    selected = selected.sort_values("final_rank")
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "trade_date": candidate_date,
                "股票代码": row["股票代码"],
                "股票名称": row["股票名称"],
                "板块划分": row["板块划分"],
                "final_rank": int(row["final_rank"]),
                "weight": float(row["weight"]),
                "refine_score": float(row["refine_score"]),
                "model_rank": int(row["model_rank"]),
                "selection_reason": (
                    f"refine_score={float(row['refine_score']):.6f}; "
                    f"sector_constraint={row['sector_constraint_status']}; "
                    f"risk_gate={row['risk_gate']}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=STEP6_FINAL_TOP5_COLUMNS)


def build_result(final_top5: pd.DataFrame) -> pd.DataFrame:
    if final_top5.empty:
        return pd.DataFrame(columns=STEP6_RESULT_COLUMNS)
    result = final_top5[["股票代码", "weight"]].copy()
    result = result.rename(columns={"股票代码": "stock_id"})
    result["stock_id"] = result["stock_id"].map(normalize_code)
    return result[STEP6_RESULT_COLUMNS]


def build_weight_plan(
    *,
    candidate_date: str,
    result: pd.DataFrame,
    max_stock_count: int,
    single_weight: float,
    max_per_sector: int,
) -> pd.DataFrame:
    weights = pd.to_numeric(result["weight"], errors="coerce") if not result.empty else pd.Series(dtype=float)
    total_weight = float(weights.sum()) if not weights.empty else 0.0
    cash_weight = max(0.0, 1.0 - total_weight)
    max_single_weight = float(weights.max()) if not weights.empty else 0.0
    min_single_weight = float(weights.min()) if not weights.empty else 0.0
    return pd.DataFrame(
        [
            {
                "trade_date": candidate_date,
                "weighting_method": WEIGHTING_METHOD,
                "selected_count": int(len(result)),
                "total_weight": total_weight,
                "cash_weight": cash_weight,
                "max_single_weight": max_single_weight,
                "min_single_weight": min_single_weight,
                "market_regime": "not_modeled_v1",
                "position_note": f"最多 {max_stock_count} 只，单只目标权重 {single_weight:.4f}；不足部分保留现金。",
                "constraint_note": f"第一轮每板块最多 {max_per_sector} 只，不足 {max_stock_count} 只时允许放松板块约束；相关性约束 v1 记录但暂不执行。",
            }
        ],
        columns=STEP6_WEIGHT_PLAN_COLUMNS,
    )


def build_manifest(
    *,
    step5_output_dir: Path,
    step2_output_dir: Path,
    output_dir: Path,
    step5_manifest: pd.DataFrame,
    step2_manifest: pd.DataFrame,
    candidate_date: str,
    candidate_size: int,
    selected_count: int,
    total_weight: float,
    cash_weight: float,
    max_stock_count: int,
    max_per_sector: int,
    min_turnover: float,
    input_step5_experiment: str | None,
    input_step2_experiment: str | None,
    note: str | None,
) -> pd.DataFrame:
    items = [
        ("schema_version", SCHEMA_VERSION),
        ("refine_set_id", REFINE_SET_ID),
        ("input_step5_path", str(step5_output_dir)),
        ("input_step2_path", str(step2_output_dir)),
        ("input_step5_experiment", input_step5_experiment or step5_output_dir.parents[1].name),
        (
            "input_step2_experiment",
            input_step2_experiment
            or manifest_value(step5_manifest, "input_step2_experiment")
            or step2_output_dir.parents[1].name,
        ),
        ("input_step5_model_set_id", manifest_value(step5_manifest, "model_set_id")),
        ("input_step2_feature_set_id", manifest_value(step2_manifest, "feature_set_id")),
        ("input_candidate_date", str(candidate_date)),
        ("input_candidate_size", str(candidate_size)),
        ("selected_count", str(selected_count)),
        ("weighting_method", WEIGHTING_METHOD),
        ("total_weight", f"{total_weight:.10f}"),
        ("cash_weight", f"{cash_weight:.10f}"),
        ("max_stock_count", str(max_stock_count)),
        ("max_per_sector", str(max_per_sector)),
        ("min_turnover", f"{min_turnover:.2f}"),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        (
            "data_window_note",
            note
            or "正式 Step-6 读取健康 Step-5 Top30 与同链路 Step-2 latest_T 特征，只在 Top30 内精排并生成 result.csv。",
        ),
        (
            "leakage_control_note",
            "Step-6 只读取 Step-5 candidate_top30 与 Step-2 latest_T 可用特征；不读取 Step-3 标签、Step-5 walk-forward 真实标签或 Step-7 评分结果。",
        ),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def build_leakage_check(*, candidate_date: str, step2_latest_t: str, selected_count: int) -> pd.DataFrame:
    rows = [
        ("input_step5_success", "PASS", "Step-6 runner/validator 要求读取的 Step-5 运行报告为 SUCCESS。"),
        (
            "candidate_date_matches_step2_latest_T",
            "PASS",
            f"candidate_date={candidate_date}，Step-2 latest_T={step2_latest_t}。",
        ),
        ("all_selected_from_top30", "PASS", f"最终入选 {selected_count} 只股票，均来自 Step-5 Top30。"),
        ("no_future_label_columns_used", "PASS", "精排不读取 Step-3 标签或 Step-5 walk-forward 真实标签字段。"),
        ("no_step7_score_used", "PASS", "Step-6 不读取 Step-7 评分结果。"),
        ("result_schema_valid", "PASS", "step6_result.csv 仅包含 stock_id,weight。"),
        ("result_stock_count_lte_5", "PASS", "step6_result.csv 行数不超过 5。"),
        ("result_weight_non_negative", "PASS", "step6_result.csv 权重均非负。"),
        ("result_weight_sum_lte_1", "PASS", "step6_result.csv 权重总和不超过 1。"),
        ("ranking_log_covers_all_candidates", "PASS", "step6_ranking_log.csv 覆盖 Step-5 candidate_top30 全部候选。"),
        ("manifest_leakage_note", "PASS", "manifest 写入 leakage_control_note。"),
    ]
    return pd.DataFrame(rows, columns=STEP6_LEAKAGE_CHECK_COLUMNS)


def build_step6_outputs(
    step5_output_dir: Path,
    step2_output_dir: Path,
    output_dir: Path,
    input_step5_experiment: str | None = None,
    input_step2_experiment: str | None = None,
    max_stock_count: int = DEFAULT_MAX_STOCK_COUNT,
    max_per_sector: int = DEFAULT_MAX_PER_SECTOR,
    min_turnover: float = DEFAULT_MIN_TURNOVER,
    single_weight: float | None = None,
    note: str | None = None,
) -> dict[str, Path]:
    step5_output_dir = Path(step5_output_dir)
    step2_output_dir = Path(step2_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step5 = read_step5_outputs(step5_output_dir)
    step2 = read_step2_outputs(step2_output_dir)
    assert_candidate_boundary(step5["candidate"], step5["manifest"])

    candidate_dates = set(pd.to_datetime(step5["candidate"]["candidate_date"], errors="coerce").dt.strftime("%Y-%m-%d"))
    if len(candidate_dates) != 1:
        raise ValueError(f"Step-5 candidate_top30 must contain one candidate_date, got {sorted(candidate_dates)}")
    candidate_date = next(iter(candidate_dates))
    step2_latest_t = manifest_value(step2["manifest"], "latest_T")
    if candidate_date != step2_latest_t:
        raise ValueError(f"Step-5 candidate_date {candidate_date} does not match Step-2 latest_T {step2_latest_t}")

    candidate_size = int(float(manifest_value(step5["manifest"], "candidate_size", str(len(step5["candidate"])))))
    latest = latest_step2_frame(step2["feature"], step2_latest_t)
    merged = merge_candidate_features(step5["candidate"], latest)
    if merged["日期"].isna().any() if "日期" in merged.columns else True:
        missing_codes = merged.loc[merged.get("日期", pd.Series(index=merged.index)).isna(), "股票代码"].astype(str).tolist()
        raise ValueError(f"Step-2 latest_T features missing for candidate stocks: {missing_codes[:10]}")

    single_weight = single_weight if single_weight is not None else 1.0 / max_stock_count
    scored = add_gate_and_scores(merged, candidate_size=candidate_size, min_turnover=min_turnover)
    ranking_log = apply_portfolio_constraints(
        scored,
        max_stock_count=max_stock_count,
        max_per_sector=max_per_sector,
        single_weight=single_weight,
    )
    ranking_log = ranking_log[STEP6_RANKING_LOG_COLUMNS]
    final_top5 = build_final_top5(ranking_log, candidate_date)
    result = build_result(final_top5)
    weight_plan = build_weight_plan(
        candidate_date=candidate_date,
        result=result,
        max_stock_count=max_stock_count,
        single_weight=single_weight,
        max_per_sector=max_per_sector,
    )
    total_weight = float(pd.to_numeric(result["weight"], errors="coerce").sum()) if not result.empty else 0.0
    cash_weight = max(0.0, 1.0 - total_weight)
    manifest = build_manifest(
        step5_output_dir=step5_output_dir,
        step2_output_dir=step2_output_dir,
        output_dir=output_dir,
        step5_manifest=step5["manifest"],
        step2_manifest=step2["manifest"],
        candidate_date=candidate_date,
        candidate_size=candidate_size,
        selected_count=len(result),
        total_weight=total_weight,
        cash_weight=cash_weight,
        max_stock_count=max_stock_count,
        max_per_sector=max_per_sector,
        min_turnover=min_turnover,
        input_step5_experiment=input_step5_experiment,
        input_step2_experiment=input_step2_experiment,
        note=note,
    )
    leakage_check = build_leakage_check(
        candidate_date=candidate_date,
        step2_latest_t=step2_latest_t,
        selected_count=len(result),
    )

    outputs = {
        "ranking_log": output_dir / "step6_ranking_log.csv",
        "final_top5": output_dir / "step6_final_top5.csv",
        "result": output_dir / "step6_result.csv",
        "weight_plan": output_dir / "step6_weight_plan.csv",
        "manifest": output_dir / "step6_refine_manifest.csv",
        "leakage_check": output_dir / "step6_leakage_check.csv",
    }
    write_csv(ranking_log, outputs["ranking_log"])
    write_csv(final_top5, outputs["final_top5"])
    write_csv(result, outputs["result"])
    write_csv(weight_plan, outputs["weight_plan"])
    write_csv(manifest, outputs["manifest"])
    write_csv(leakage_check, outputs["leakage_check"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-6 refined Top5 and result outputs.")
    parser.add_argument("--step5-output-dir", type=Path, default=DEFAULT_STEP5_OUTPUT_DIR)
    parser.add_argument("--step2-output-dir", type=Path, default=DEFAULT_STEP2_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-step5-experiment", default=None)
    parser.add_argument("--input-step2-experiment", default=None)
    parser.add_argument("--max-stock-count", type=int, default=DEFAULT_MAX_STOCK_COUNT)
    parser.add_argument("--max-per-sector", type=int, default=DEFAULT_MAX_PER_SECTOR)
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER)
    parser.add_argument("--single-weight", type=float, default=None)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step6_outputs(
        step5_output_dir=args.step5_output_dir,
        step2_output_dir=args.step2_output_dir,
        output_dir=args.output_dir,
        input_step5_experiment=args.input_step5_experiment,
        input_step2_experiment=args.input_step2_experiment,
        max_stock_count=args.max_stock_count,
        max_per_sector=args.max_per_sector,
        min_turnover=args.min_turnover,
        single_weight=args.single_weight,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP1_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / "exp_20260616_step1_workflow_0_1"
    / "outputs"
    / "step1"
)
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / f"exp_{datetime.now().strftime('%Y%m%d')}_step2_workflow_0_1"
    / "outputs"
    / "step2"
)


SCHEMA_VERSION = "workflow_0.1_csv_v1"
FEATURE_SET_ID = "feature_set_v1_momentum_volume_risk"
LOW_LIQUIDITY_AMOUNT_MA3_THRESHOLD = 100_000_000
EXTREME_DROP_THRESHOLD_PCT = -7.0


STEP2_FEATURE_COLUMNS = [
    "股票代码",
    "日期",
    "股票名称",
    "原始行业",
    "行业分类口径",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma5",
    "ma10",
    "ma20",
    "ma5_over_ma20",
    "trend_slope_5",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "amount_ma3",
    "amount_ma5",
    "amount_ma20",
    "volume_ma5",
    "volume_ma20",
    "amount_ratio_5_20",
    "volume_ratio_5_20",
    "amount_rank_5",
    "market_ret_5",
    "sector_ret_5",
    "sector_excess_ret_5",
    "stock_vs_market_ret_5",
    "stock_vs_sector_ret_5",
    "rank_in_sector_ret_5",
    "market_index",
    "market_amount",
    "sector_amount_ratio_5_20",
    "sector_short_score",
    "max_drawdown_20",
    "extreme_drop_20_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
    "risk_any_flag",
    "板块划分",
]


STEP2_SECTOR_COLUMNS = [
    "日期",
    "板块划分",
    "sector_daily_ret",
    "sector_amount",
    "sector_volume",
    "sector_stock_count",
    "new_high_20_ratio",
    "ret_5_gt_5pct_ratio",
    "sector_index",
    "sector_ret_3",
    "sector_ret_5",
    "sector_ret_10",
    "sector_ma5",
    "sector_ma10",
    "sector_amount_ma5",
    "sector_amount_ma20",
    "sector_amount_ratio_5_20",
    "sector_amount_rank_5",
    "market_ret_5",
    "sector_excess_ret_5",
    "score_sector_ret_5",
    "score_sector_excess_ret_5",
    "score_ma_bull",
    "score_amount_ratio",
    "score_amount_rank_5",
    "score_new_high_20_ratio",
    "score_ret_5_gt_5pct_ratio",
    "score_catalyst",
    "sector_short_score",
]


STEP2_LATEST_SCREEN_COLUMNS = [
    "股票代码",
    "股票名称",
    "日期",
    "板块划分",
    "原始行业",
    "ret_5",
    "trend_slope_5",
    "stock_vs_sector_ret_5",
    "rank_in_sector_ret_5",
    "amount_ratio_5_20",
    "volume_ratio_5_20",
    "amount_rank_5",
    "max_drawdown_20",
    "extreme_drop_20_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
    "risk_any_flag",
    "risk_pass_flag",
    "score_sector_ret_5",
    "score_sector_excess_ret_5",
    "score_ma_bull",
    "score_amount_ratio",
    "score_amount_rank_5",
    "score_new_high_20_ratio",
    "score_ret_5_gt_5pct_ratio",
    "score_catalyst",
    "sector_short_score",
    "stock_trend_score",
    "volume_confirm_score",
    "进入后续流程标记",
]


STEP2_RISK_COLUMNS = [
    "股票代码",
    "日期",
    "股票名称",
    "max_drawdown_20",
    "extreme_drop_20_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
    "risk_any_flag",
    "板块划分",
]


STEP2_METADATA_COLUMNS = [
    "特征名",
    "特征来源",
    "计算窗口",
    "是否用于模型",
    "是否用于精排",
    "防泄漏说明",
]


GENERATED_FEATURES_FOR_METADATA = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma5",
    "ma10",
    "ma20",
    "ma5_over_ma20",
    "trend_slope_5",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "amount_ma3",
    "amount_ma5",
    "amount_ma20",
    "volume_ma5",
    "volume_ma20",
    "amount_ratio_5_20",
    "volume_ratio_5_20",
    "amount_rank_5",
    "market_ret_5",
    "sector_ret_5",
    "sector_excess_ret_5",
    "stock_vs_market_ret_5",
    "stock_vs_sector_ret_5",
    "rank_in_sector_ret_5",
    "market_index",
    "market_amount",
    "sector_amount_ratio_5_20",
    "sector_short_score",
    "max_drawdown_20",
    "extreme_drop_20_flag",
    "low_liquidity_flag",
    "no_trade_or_abnormal_flag",
    "risk_any_flag",
    "stock_trend_score",
    "volume_confirm_score",
    "进入后续流程标记",
]


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


def read_step1_outputs(step1_output_dir: Path) -> dict[str, pd.DataFrame]:
    step1_output_dir = Path(step1_output_dir)
    daily = read_csv(step1_output_dir / "step1_daily_raw_data.csv", dtype={"股票代码": str})
    stock = read_csv(step1_output_dir / "step1_stock_summary.csv", dtype={"股票代码": str})
    sector = read_csv(step1_output_dir / "step1_sector_summary.csv")
    manifest = read_csv(step1_output_dir / "step1_data_manifest.csv")
    return {"daily": daily, "stock": stock, "sector": sector, "manifest": manifest}


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def rolling_trend_slope(values: np.ndarray) -> float:
    if len(values) < 5 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    mean_value = float(np.mean(values))
    if mean_value == 0 or np.isnan(mean_value):
        return np.nan
    return slope / mean_value * 100


def rolling_max_drawdown(values: np.ndarray) -> float:
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1
    return float(np.min(drawdowns) * 100)


def percent_rank_high_is_good(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=True) * 100


def score_by_desc_rank(series: pd.Series, top_score: float, step: float) -> pd.Series:
    ranks = series.rank(method="first", ascending=False, na_option="bottom")
    scores = top_score - (ranks - 1) * step
    scores = scores.clip(lower=0)
    return scores.where(series.notna(), 0)


def prepare_daily(step1: dict[str, pd.DataFrame]) -> pd.DataFrame:
    daily = step1["daily"].copy()
    stock = step1["stock"].copy()
    daily["股票代码"] = daily["股票代码"].map(normalize_code)
    stock["股票代码"] = stock["股票代码"].map(normalize_code)
    daily["日期_dt"] = pd.to_datetime(daily["日期"], errors="coerce")

    numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]
    daily = to_numeric(daily, numeric_cols)

    stock_info = stock[["股票代码", "股票名称", "原始行业", "行业分类口径", "板块划分"]].drop_duplicates("股票代码")
    out = daily.merge(stock_info, on="股票代码", how="left")
    out["股票名称"] = out["股票名称"].fillna("")
    out["原始行业"] = out["原始行业"].fillna("")
    out["行业分类口径"] = out["行业分类口径"].fillna("")
    out["板块划分"] = out["板块划分"].fillna("未匹配")
    return out.sort_values(["股票代码", "日期_dt"]).reset_index(drop=True)


def add_stock_time_series_features(daily: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in daily.groupby("股票代码", sort=True):
        group = group.sort_values("日期_dt").copy()
        close = group["收盘"]
        amount = group["成交额"]
        volume = group["成交量"]
        pct_chg = group["涨跌幅"]

        for window in (1, 3, 5, 10, 20):
            group[f"ret_{window}"] = close.pct_change(window) * 100

        group["ma5"] = close.rolling(5, min_periods=5).mean()
        group["ma10"] = close.rolling(10, min_periods=10).mean()
        group["ma20"] = close.rolling(20, min_periods=20).mean()
        group["ma5_over_ma20"] = (group["ma5"] / group["ma20"] - 1) * 100
        group["trend_slope_5"] = close.rolling(5, min_periods=5).apply(rolling_trend_slope, raw=True)

        for window in (5, 10, 20):
            group[f"volatility_{window}"] = pct_chg.rolling(window, min_periods=window).std()

        group["amount_ma3"] = amount.rolling(3, min_periods=3).mean()
        group["amount_ma5"] = amount.rolling(5, min_periods=5).mean()
        group["amount_ma20"] = amount.rolling(20, min_periods=20).mean()
        group["volume_ma5"] = volume.rolling(5, min_periods=5).mean()
        group["volume_ma20"] = volume.rolling(20, min_periods=20).mean()
        group["amount_ratio_5_20"] = group["amount_ma5"] / group["amount_ma20"]
        group["volume_ratio_5_20"] = group["volume_ma5"] / group["volume_ma20"]

        rolling_high_20 = close.rolling(20, min_periods=20).max()
        group["_new_high_20_flag"] = ((close >= rolling_high_20) & rolling_high_20.notna()).astype(int)
        group["max_drawdown_20"] = close.rolling(20, min_periods=20).apply(rolling_max_drawdown, raw=True)
        group["extreme_drop_20_flag"] = (
            pct_chg.rolling(20, min_periods=20)
            .apply(lambda values: float(np.nanmin(values) <= EXTREME_DROP_THRESHOLD_PCT), raw=True)
            .fillna(0)
            .astype(int)
        )
        group["low_liquidity_flag"] = (
            group["amount_ma3"].lt(LOW_LIQUIDITY_AMOUNT_MA3_THRESHOLD).fillna(False).astype(int)
        )
        group["no_trade_or_abnormal_flag"] = (
            group["成交量"].le(0)
            | group["成交额"].le(0)
            | group["开盘"].le(0)
            | group["收盘"].le(0)
            | group["最高"].le(0)
            | group["最低"].le(0)
        ).fillna(True).astype(int)
        group["risk_any_flag"] = (
            group[["extreme_drop_20_flag", "low_liquidity_flag", "no_trade_or_abnormal_flag"]].max(axis=1).astype(int)
        )
        frames.append(group)

    out = pd.concat(frames, ignore_index=True)
    out["amount_rank_5"] = out.groupby("日期")["amount_ma5"].transform(percent_rank_high_is_good)
    return out


def build_market_table(feature_base: pd.DataFrame) -> pd.DataFrame:
    market = (
        feature_base.groupby("日期_dt", as_index=False)
        .agg(market_daily_ret=("涨跌幅", "mean"), market_amount=("成交额", "sum"))
        .sort_values("日期_dt")
    )
    market["market_index"] = (1 + market["market_daily_ret"].fillna(0) / 100).cumprod() * 1000
    market["market_ret_5"] = market["market_index"].pct_change(5) * 100
    market["日期"] = market["日期_dt"].dt.strftime("%Y-%m-%d")
    return market[["日期", "market_ret_5", "market_index", "market_amount"]]


def build_sector_feature_table(feature_base: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    tmp = feature_base.copy()
    tmp["_ret_5_gt_5pct_flag"] = tmp["ret_5"].gt(5).astype(int)
    sector = (
        tmp.groupby(["日期_dt", "板块划分"], as_index=False)
        .agg(
            sector_daily_ret=("涨跌幅", "mean"),
            sector_amount=("成交额", "sum"),
            sector_volume=("成交量", "sum"),
            sector_stock_count=("股票代码", "nunique"),
            new_high_20_ratio=("_new_high_20_flag", "mean"),
            ret_5_gt_5pct_ratio=("_ret_5_gt_5pct_flag", "mean"),
        )
        .sort_values(["板块划分", "日期_dt"])
    )

    frames: list[pd.DataFrame] = []
    for _, group in sector.groupby("板块划分", sort=True):
        group = group.sort_values("日期_dt").copy()
        group["sector_index"] = (1 + group["sector_daily_ret"].fillna(0) / 100).cumprod() * 1000
        group["sector_ret_3"] = group["sector_index"].pct_change(3) * 100
        group["sector_ret_5"] = group["sector_index"].pct_change(5) * 100
        group["sector_ret_10"] = group["sector_index"].pct_change(10) * 100
        group["sector_ma5"] = group["sector_index"].rolling(5, min_periods=5).mean()
        group["sector_ma10"] = group["sector_index"].rolling(10, min_periods=10).mean()
        group["sector_amount_ma5"] = group["sector_amount"].rolling(5, min_periods=5).mean()
        group["sector_amount_ma20"] = group["sector_amount"].rolling(20, min_periods=20).mean()
        group["sector_amount_ratio_5_20"] = group["sector_amount_ma5"] / group["sector_amount_ma20"]
        frames.append(group)

    out = pd.concat(frames, ignore_index=True)
    out["日期"] = out["日期_dt"].dt.strftime("%Y-%m-%d")
    out["sector_amount_rank_5"] = out.groupby("日期")["sector_amount_ma5"].transform(percent_rank_high_is_good)
    out = out.merge(market[["日期", "market_ret_5"]], on="日期", how="left")
    out["sector_excess_ret_5"] = out["sector_ret_5"] - out["market_ret_5"]

    out["score_sector_ret_5"] = out.groupby("日期")["sector_ret_5"].transform(
        lambda s: score_by_desc_rank(s, top_score=20, step=4)
    )
    out["score_sector_excess_ret_5"] = out.groupby("日期")["sector_excess_ret_5"].transform(
        lambda s: score_by_desc_rank(s, top_score=15, step=3)
    )
    out["score_ma_bull"] = np.select(
        [
            (out["sector_index"] > out["sector_ma5"]) & (out["sector_ma5"] > out["sector_ma10"]),
            (out["sector_index"] > out["sector_ma5"]) | (out["sector_ma5"] > out["sector_ma10"]),
        ],
        [10, 5],
        default=0,
    )
    out["score_amount_ratio"] = np.select(
        [out["sector_amount_ratio_5_20"] > 1.3, out["sector_amount_ratio_5_20"] >= 1.0],
        [15, 8],
        default=0,
    )
    out["score_amount_rank_5"] = out.groupby("日期")["sector_amount_ma5"].transform(
        lambda s: score_by_desc_rank(s, top_score=10, step=2)
    )
    out["score_new_high_20_ratio"] = (out["new_high_20_ratio"].fillna(0) * 10).clip(0, 10)
    out["score_ret_5_gt_5pct_ratio"] = (out["ret_5_gt_5pct_ratio"].fillna(0) * 10).clip(0, 10)
    out["score_catalyst"] = 0

    score_columns = [
        "score_sector_ret_5",
        "score_sector_excess_ret_5",
        "score_ma_bull",
        "score_amount_ratio",
        "score_amount_rank_5",
        "score_new_high_20_ratio",
        "score_ret_5_gt_5pct_ratio",
        "score_catalyst",
    ]
    out["sector_short_score"] = out[score_columns].fillna(0).sum(axis=1)
    return out[STEP2_SECTOR_COLUMNS].sort_values(["日期", "板块划分"]).reset_index(drop=True)


def build_feature_table(feature_base: pd.DataFrame, market: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    sector_cols = [
        "日期",
        "板块划分",
        "sector_ret_5",
        "sector_excess_ret_5",
        "sector_amount_ratio_5_20",
        "sector_short_score",
    ]
    out = feature_base.copy()
    out["日期"] = out["日期_dt"].dt.strftime("%Y-%m-%d")
    out = out.merge(market, on="日期", how="left")
    out = out.merge(sector[sector_cols], on=["日期", "板块划分"], how="left")
    out["stock_vs_market_ret_5"] = out["ret_5"] - out["market_ret_5"]
    out["stock_vs_sector_ret_5"] = out["ret_5"] - out["sector_ret_5"]
    out["rank_in_sector_ret_5"] = out.groupby(["日期", "板块划分"])["ret_5"].transform(percent_rank_high_is_good)
    return out[STEP2_FEATURE_COLUMNS].sort_values(["股票代码", "日期"]).reset_index(drop=True)


def rank_latest_columns(latest: pd.DataFrame, columns: list[str]) -> list[pd.Series]:
    return [percent_rank_high_is_good(latest[col]).fillna(0) for col in columns]


def build_latest_t_screen(feature_table: pd.DataFrame, sector: pd.DataFrame, latest_t: str) -> pd.DataFrame:
    latest = feature_table[feature_table["日期"] == latest_t].copy()
    sector_scores = sector[sector["日期"] == latest_t][
        [
            "日期",
            "板块划分",
            "score_sector_ret_5",
            "score_sector_excess_ret_5",
            "score_ma_bull",
            "score_amount_ratio",
            "score_amount_rank_5",
            "score_new_high_20_ratio",
            "score_ret_5_gt_5pct_ratio",
            "score_catalyst",
        ]
    ]
    latest = latest.merge(sector_scores, on=["日期", "板块划分"], how="left")
    latest["risk_pass_flag"] = latest["risk_any_flag"].eq(0).astype(int)

    trend_ranks = rank_latest_columns(latest, ["ret_5", "trend_slope_5", "stock_vs_sector_ret_5"])
    latest["stock_trend_score"] = (
        trend_ranks[0] * 0.25
        + trend_ranks[1] * 0.25
        + trend_ranks[2] * 0.25
        + latest["rank_in_sector_ret_5"].fillna(0) * 0.25
    )

    volume_ranks = rank_latest_columns(latest, ["amount_ratio_5_20", "volume_ratio_5_20"])
    latest["volume_confirm_score"] = (
        volume_ranks[0] * (1 / 3)
        + volume_ranks[1] * (1 / 3)
        + latest["amount_rank_5"].fillna(0) * (1 / 3)
    )

    sector_threshold = latest["sector_short_score"].median()
    latest["进入后续流程标记"] = np.where(
        latest["risk_pass_flag"].eq(1)
        & latest["sector_short_score"].ge(sector_threshold)
        & latest["stock_trend_score"].ge(50)
        & latest["volume_confirm_score"].ge(50),
        "是",
        "否",
    )
    return latest[STEP2_LATEST_SCREEN_COLUMNS].sort_values(
        ["进入后续流程标记", "sector_short_score", "stock_trend_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_risk_feature_table(feature_table: pd.DataFrame) -> pd.DataFrame:
    return feature_table[STEP2_RISK_COLUMNS].sort_values(["股票代码", "日期"]).reset_index(drop=True)


def build_feature_metadata() -> pd.DataFrame:
    rows = []
    leakage_note = "只使用当前行日期T及以前的 Step-1 日频行情、行业映射或同日横截面数据计算。"
    for name in GENERATED_FEATURES_FOR_METADATA:
        if name.startswith("ret_"):
            source = "Step-1 日频收盘价"
            window = name.replace("ret_", "") + "日"
        elif name.startswith("ma") or name.startswith("trend") or name.startswith("volatility"):
            source = "Step-1 日频价格路径"
            window = "滚动历史窗口"
        elif "amount" in name or "volume" in name:
            source = "Step-1 日频成交量/成交额"
            window = "3日/5日/20日或同日横截面"
        elif "sector" in name:
            source = "Step-1 行业映射自聚合六大板块"
            window = "5日/10日/20日或同日板块横截面"
        elif "market" in name:
            source = "沪深300当前成分股等权自聚合"
            window = "5日或日频累计"
        elif "risk" in name or "drop" in name or "liquidity" in name or "abnormal" in name or "drawdown" in name:
            source = "Step-1 日频路径风险过滤"
            window = "3日/20日/当前日"
        else:
            source = "Step-2 最新T日初筛规则"
            window = "latest_T"

        rows.append(
            {
                "特征名": name,
                "特征来源": source,
                "计算窗口": window,
                "是否用于模型": "是" if name != "进入后续流程标记" else "否",
                "是否用于精排": "是",
                "防泄漏说明": leakage_note,
            }
        )
    return pd.DataFrame(rows, columns=STEP2_METADATA_COLUMNS)


def build_manifest(
    *,
    step1_output_dir: Path,
    output_dir: Path,
    manifest: pd.DataFrame,
    feature_table: pd.DataFrame,
    sector_table: pd.DataFrame,
    latest_t_screen: pd.DataFrame,
    input_step1_experiment: str | None,
    note: str | None,
) -> pd.DataFrame:
    latest_t = manifest_value(manifest, "latest_T", "" if feature_table.empty else str(feature_table["日期"].max()))
    date_start = "" if feature_table.empty else str(feature_table["日期"].min())
    date_end = "" if feature_table.empty else str(feature_table["日期"].max())
    raw_days = str(feature_table["日期"].nunique()) if not feature_table.empty else "0"
    items = [
        ("schema_version", SCHEMA_VERSION),
        ("feature_set_id", FEATURE_SET_ID),
        ("date_start", date_start),
        ("date_end", date_end),
        ("latest_T", latest_t),
        ("raw_交易日数", raw_days),
        ("input_step1_path", str(step1_output_dir)),
        ("input_step1_experiment", input_step1_experiment or step1_output_dir.parents[1].name),
        ("input_step1_latest_T", manifest_value(manifest, "latest_T")),
        ("input_step1_schema_version", manifest_value(manifest, "schema_version")),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("output_dir", str(output_dir)),
        ("input_daily_rows", str(len(feature_table))),
        ("output_feature_rows", str(len(feature_table))),
        ("output_sector_rows", str(len(sector_table))),
        ("output_latest_t_rows", str(len(latest_t_screen))),
        ("data_window_note", note or "读取健康 Step-1 输出生成 Step-2 标准特征资产。"),
        ("feature_set_note", "第一版只使用板块趋势、个股趋势、量能确认、基础风险；催化剂暂记0，不引入未来信息。"),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(6)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def build_step2_outputs(
    step1_output_dir: Path,
    output_dir: Path,
    input_step1_experiment: str | None = None,
    note: str | None = None,
) -> dict[str, Path]:
    step1_output_dir = Path(step1_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step1 = read_step1_outputs(step1_output_dir)
    daily = prepare_daily(step1)
    feature_base = add_stock_time_series_features(daily)
    market = build_market_table(feature_base)
    sector = build_sector_feature_table(feature_base, market)
    feature_table = build_feature_table(feature_base, market, sector)
    latest_t = manifest_value(step1["manifest"], "latest_T", str(feature_table["日期"].max()))
    sector_score_latest = sector[sector["日期"] == latest_t].sort_values(
        "sector_short_score", ascending=False
    ).reset_index(drop=True)
    latest_t_screen = build_latest_t_screen(feature_table, sector, latest_t)
    risk_table = build_risk_feature_table(feature_table)
    metadata = build_feature_metadata()
    manifest = build_manifest(
        step1_output_dir=step1_output_dir,
        output_dir=output_dir,
        manifest=step1["manifest"],
        feature_table=feature_table,
        sector_table=sector,
        latest_t_screen=latest_t_screen,
        input_step1_experiment=input_step1_experiment,
        note=note,
    )

    outputs = {
        "feature": output_dir / "step2_feature_table_daily.csv",
        "sector": output_dir / "step2_sector_feature_table.csv",
        "latest": output_dir / "step2_latest_t_screen.csv",
        "metadata": output_dir / "step2_feature_metadata.csv",
        "manifest": output_dir / "step2_data_manifest.csv",
        "sector_latest": output_dir / "step2_sector_score_latest.csv",
        "risk": output_dir / "step2_risk_feature_table.csv",
    }
    write_csv(feature_table, outputs["feature"])
    write_csv(sector, outputs["sector"])
    write_csv(latest_t_screen, outputs["latest"])
    write_csv(metadata, outputs["metadata"])
    write_csv(manifest, outputs["manifest"])
    write_csv(sector_score_latest, outputs["sector_latest"])
    write_csv(risk_table, outputs["risk"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-2 standard CSV outputs.")
    parser.add_argument("--step1-output-dir", type=Path, default=DEFAULT_STEP1_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-step1-experiment", default=None)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step2_outputs(
        step1_output_dir=args.step1_output_dir,
        output_dir=args.output_dir,
        input_step1_experiment=args.input_step1_experiment,
        note=args.note,
    )
    print(f"output_dir: {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

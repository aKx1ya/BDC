from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "股票代码": "stock_id",
    "证券代码": "stock_id",
    "代码": "stock_id",
    "日期": "date",
    "交易日期": "date",
    "开盘": "open",
    "开盘价": "open",
    "收盘": "close",
    "收盘价": "close",
    "最高": "high",
    "最高价": "high",
    "最低": "low",
    "最低价": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_chg",
    "行业": "sector",
    "申万一级行业": "sector",
}

REQUIRED_COLUMNS = ["stock_id", "date", "open", "close", "high", "low"]
OPTIONAL_NUMERIC_COLUMNS = ["volume", "amount", "turnover", "pct_chg"]
IDENTIFIER_COLUMNS = {"stock_id", "date", "sector"}
LEAKAGE_COLUMNS = {"label", "future_open_t1", "future_open_t5", "history_count"}


def read_market_data(path: str | Path) -> pd.DataFrame:
    market = pd.read_csv(path)
    return normalize_columns(market)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={col: COLUMN_MAP.get(col, col) for col in df.columns}).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required market data columns: {missing}")

    normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for col in ["open", "close", "high", "low", *OPTIONAL_NUMERIC_COLUMNS]:
        if col not in normalized.columns:
            normalized[col] = 0.0
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "sector" not in normalized.columns:
        normalized["sector"] = "UNKNOWN"
    normalized["sector"] = normalized["sector"].fillna("UNKNOWN").astype(str)

    normalized = normalized.dropna(subset=["stock_id", "date", "open", "close", "high", "low"])
    normalized = normalized.sort_values(["stock_id", "date"]).drop_duplicates(["stock_id", "date"], keep="last")
    return normalized.reset_index(drop=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = normalize_columns(df) if not {"stock_id", "date", "open", "close", "high", "low"}.issubset(df.columns) else df.copy()
    features = features.sort_values(["stock_id", "date"]).reset_index(drop=True)
    grouped = features.groupby("stock_id", group_keys=False)

    features["history_count"] = grouped.cumcount() + 1
    features["ret_1"] = grouped["close"].pct_change(1)
    for window in [3, 5, 10, 20, 40]:
        features[f"ret_{window}"] = grouped["close"].pct_change(window)
        features[f"ma_{window}"] = grouped["close"].transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        features[f"close_to_ma_{window}"] = _safe_divide(features["close"], features[f"ma_{window}"]) - 1

    for window in [5, 10, 20]:
        features[f"volatility_{window}"] = grouped["ret_1"].transform(lambda s, w=window: s.rolling(w, min_periods=2).std())

    for window in [3, 5, 20]:
        features[f"amount_ma_{window}"] = grouped["amount"].transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        features[f"volume_ma_{window}"] = grouped["volume"].transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())

    features["amount_ratio_3"] = _safe_divide(features["amount"], features["amount_ma_3"])
    features["volume_ratio_5"] = _safe_divide(features["volume"], features["volume_ma_5"])
    features["avg_amount_3"] = features["amount_ma_3"]

    price_range = (features["high"] - features["low"]).replace(0, np.nan)
    features["clv"] = ((features["close"] - features["low"]) - (features["high"] - features["close"])) / price_range
    features["lower_shadow_ratio"] = (features[["open", "close"]].min(axis=1) - features["low"]) / price_range
    features["upper_shadow_ratio"] = (features["high"] - features[["open", "close"]].max(axis=1)) / price_range
    features["body_ratio"] = (features["close"] - features["open"]).abs() / price_range
    features["range_ratio"] = _safe_divide(features["high"] - features["low"], features["close"])
    features["volume_close_strength"] = features["amount_ratio_3"] * features["clv"]

    rolling_max = grouped["close"].transform(lambda s: s.rolling(20, min_periods=1).max())
    features["drawdown_20"] = _safe_divide(features["close"], rolling_max) - 1
    features["max_drop_20"] = grouped["ret_1"].transform(lambda s: s.rolling(20, min_periods=1).min())

    for col in ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "amount", "volatility_20"]:
        rank_col = f"{col}_rank_pct"
        features[rank_col] = features.groupby("date")[col].rank(pct=True, method="average")

    sector_daily = (
        features.groupby(["date", "sector"], as_index=False)["ret_1"]
        .mean()
        .rename(columns={"ret_1": "sector_ret_1"})
        .sort_values(["sector", "date"])
    )
    sector_daily["sector_momentum_3"] = sector_daily.groupby("sector")["sector_ret_1"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    sector_daily["sector_momentum_rank_pct"] = sector_daily.groupby("date")["sector_momentum_3"].rank(
        pct=True, method="average"
    )
    features = features.merge(sector_daily, on=["date", "sector"], how="left")

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.sort_values(["date", "stock_id"]).reset_index(drop=True)


def build_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    labeled = df.sort_values(["stock_id", "date"]).copy()
    grouped = labeled.groupby("stock_id", group_keys=False)
    labeled["future_open_t1"] = grouped["open"].shift(-1)
    labeled["future_open_t5"] = grouped["open"].shift(-horizon)
    labeled["label"] = (labeled["future_open_t5"] - labeled["future_open_t1"]) / labeled["future_open_t1"]
    labeled["label"] = labeled["label"].replace([np.inf, -np.inf], np.nan)
    return labeled.sort_values(["date", "stock_id"]).reset_index(drop=True)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    excluded = IDENTIFIER_COLUMNS | LEAKAGE_COLUMNS
    return [col for col in numeric_cols if col not in excluded]


def prepare_training_frame(raw: pd.DataFrame, horizon: int, sequence_length: int) -> tuple[pd.DataFrame, List[str]]:
    features = engineer_features(raw)
    labeled = build_labels(features, horizon=horizon)
    labeled = labeled[labeled["history_count"] >= sequence_length].copy()
    labeled = labeled.dropna(subset=["label"]).copy()
    feature_cols = select_feature_columns(labeled)
    if not feature_cols:
        raise ValueError("No numeric feature columns are available for training.")
    return labeled, feature_cols


def latest_prediction_frame(raw: pd.DataFrame) -> pd.DataFrame:
    features = engineer_features(raw)
    latest_date = features["date"].max()
    return features[features["date"] == latest_date].sort_values("stock_id").reset_index(drop=True)

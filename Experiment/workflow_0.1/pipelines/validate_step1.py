#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_step1_outputs import (  # noqa: E402
    STEP1_DAILY_COLUMNS,
    STEP1_SECTOR_COLUMNS,
    STEP1_STOCK_COLUMNS,
    normalize_code,
)


STEP1_OUTPUT_FILES = {
    "daily": "step1_daily_raw_data.csv",
    "stock": "step1_stock_summary.csv",
    "sector": "step1_sector_summary.csv",
    "manifest": "step1_data_manifest.csv",
}

REQUIRED_MANIFEST_ITEMS = {
    "schema_version",
    "date_start",
    "date_end",
    "latest_T",
    "raw_交易日数",
    "data_source",
    "generated_at",
}


class Step1ValidationError(Exception):
    """Step-1 正式验收失败。"""


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise Step1ValidationError("; ".join(errors))


def require_columns(df: pd.DataFrame, required: set[str], file_name: str) -> list[str]:
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"{file_name} missing columns: {missing}"]
    return []


def read_csv(path: Path, dtype: dict[str, object] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise Step1ValidationError(f"missing file: {path}")
    return pd.read_csv(path, dtype=dtype)


def validate_raw_data(raw_dir: Path, expected_stock_count: int = 300) -> dict[str, object]:
    raw_dir = Path(raw_dir)
    errors: list[str] = []

    hs300 = read_csv(raw_dir / "hs300_stocks.csv", dtype={"code": str})
    daily = read_csv(raw_dir / "daily_price_volume.csv", dtype={"code": str})

    errors += require_columns(hs300, {"updateDate", "code", "code_name"}, "hs300_stocks.csv")
    errors += require_columns(daily, {"date", "code"}, "daily_price_volume.csv")
    raise_if_errors(errors)

    hs300 = hs300.copy()
    daily = daily.copy()
    hs300["股票代码"] = hs300["code"].map(normalize_code)
    daily["股票代码"] = daily["code"].map(normalize_code)
    daily["日期"] = pd.to_datetime(daily["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    current_codes = set(hs300["股票代码"].dropna())
    hs300_count = len(current_codes)
    if hs300_count != expected_stock_count:
        errors.append(f"hs300 stock count expected {expected_stock_count}, got {hs300_count}")

    duplicate_count = int(daily.duplicated(subset=["股票代码", "日期"]).sum())
    if duplicate_count:
        errors.append(f"daily_price_volume.csv duplicate 股票代码+日期 rows: {duplicate_count}")

    current_daily = daily[daily["股票代码"].isin(current_codes)].copy()
    daily_codes = set(current_daily["股票代码"].dropna())
    missing_codes = sorted(current_codes - daily_codes)
    if missing_codes:
        errors.append(f"missing daily data for current stocks: {missing_codes[:10]}")

    latest_by_code = current_daily.groupby("股票代码")["日期"].max()
    unique_latest_dates = sorted(set(latest_by_code.dropna()))
    if len(unique_latest_dates) > 1:
        errors.append(f"current stocks latest dates are not aligned: {unique_latest_dates}")

    raise_if_errors(errors)

    latest_t = unique_latest_dates[0] if unique_latest_dates else ""
    return {
        "hs300_count": hs300_count,
        "daily_rows": int(len(current_daily)),
        "daily_current_code_count": len(daily_codes),
        "daily_date_start": "" if current_daily.empty else str(current_daily["日期"].min()),
        "daily_latest_T": latest_t,
        "daily_unique_dates": int(current_daily["日期"].nunique()),
        "daily_duplicates": duplicate_count,
    }


def validate_columns(df: pd.DataFrame, expected: list[str], file_name: str) -> list[str]:
    actual = list(df.columns)
    if actual != expected:
        return [f"{file_name} columns mismatch: expected {expected}, got {actual}"]
    return []


def validate_outputs(output_dir: Path, expected_stock_count: int = 300) -> dict[str, object]:
    output_dir = Path(output_dir)
    errors: list[str] = []

    daily = read_csv(output_dir / STEP1_OUTPUT_FILES["daily"], dtype={"股票代码": str})
    stock = read_csv(output_dir / STEP1_OUTPUT_FILES["stock"], dtype={"股票代码": str})
    sector = read_csv(output_dir / STEP1_OUTPUT_FILES["sector"])
    manifest = read_csv(output_dir / STEP1_OUTPUT_FILES["manifest"])

    errors += validate_columns(daily, STEP1_DAILY_COLUMNS, STEP1_OUTPUT_FILES["daily"])
    errors += validate_columns(stock, STEP1_STOCK_COLUMNS, STEP1_OUTPUT_FILES["stock"])
    errors += validate_columns(sector, STEP1_SECTOR_COLUMNS, STEP1_OUTPUT_FILES["sector"])
    errors += validate_columns(manifest, ["项目", "说明"], STEP1_OUTPUT_FILES["manifest"])

    output_stock_count = int(len(stock))
    if output_stock_count != expected_stock_count:
        errors.append(f"step1_stock_summary.csv stock count expected {expected_stock_count}, got {output_stock_count}")

    duplicate_count = int(daily.duplicated(subset=["股票代码", "日期"]).sum()) if {"股票代码", "日期"} <= set(daily.columns) else -1
    if duplicate_count:
        errors.append(f"step1_daily_raw_data.csv duplicate 股票代码+日期 rows: {duplicate_count}")

    latest_dates = sorted(set(stock["最新日期"].dropna())) if "最新日期" in stock.columns else []
    if len(latest_dates) > 1:
        errors.append(f"step1_stock_summary.csv latest dates are not aligned: {latest_dates}")

    unmatched_count = int(stock["板块划分"].eq("未匹配").sum()) if "板块划分" in stock.columns else -1
    if unmatched_count:
        errors.append(f"unmatched sector count must be 0, got {unmatched_count}")

    manifest_items = set(manifest["项目"].astype(str)) if "项目" in manifest.columns else set()
    missing_manifest_items = sorted(REQUIRED_MANIFEST_ITEMS - manifest_items)
    if missing_manifest_items:
        errors.append(f"step1_data_manifest.csv missing items: {missing_manifest_items}")

    raise_if_errors(errors)

    latest_t = latest_dates[0] if latest_dates else ""
    return {
        "output_daily_rows": int(len(daily)),
        "output_daily_code_count": int(daily["股票代码"].nunique()) if "股票代码" in daily.columns else 0,
        "output_stock_count": output_stock_count,
        "output_latest_T": latest_t,
        "output_unmatched_sector_count": unmatched_count,
        "output_daily_duplicates": duplicate_count,
        "output_sector_count": int(len(sector)),
    }


def validate_step1(raw_dir: Path, output_dir: Path, expected_stock_count: int = 300) -> dict[str, object]:
    metrics = {}
    metrics.update(validate_raw_data(raw_dir, expected_stock_count=expected_stock_count))
    metrics.update(validate_outputs(output_dir, expected_stock_count=expected_stock_count))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate workflow_0.1 Step-1 raw data and standard outputs.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-stock-count", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = validate_step1(args.raw_dir, args.output_dir, expected_stock_count=args.expected_stock_count)
    except Step1ValidationError as exc:
        print(f"Step-1 validation failed: {exc}")
        return 1

    print("Step-1 validation passed")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

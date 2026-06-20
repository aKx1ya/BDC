#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "bigdata_challenge" / "data" / "raw"
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "experiments"
    / f"exp_{datetime.now().strftime('%Y%m%d')}_step1_workflow_0_1"
    / "outputs"
    / "step1"
)


STEP1_DAILY_COLUMNS = [
    "股票代码",
    "日期",
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
]

STEP1_STOCK_COLUMNS = [
    "股票代码",
    "股票名称",
    "成分股更新日期",
    "raw_起始日期",
    "raw_结束日期",
    "raw_交易日数",
    "最新日期",
    "最新开盘",
    "最新收盘",
    "最新最高",
    "最新最低",
    "最新成交量",
    "最新成交额",
    "最新振幅",
    "最新涨跌额",
    "最新换手率",
    "最新涨跌幅",
    "近5日收益率",
    "近5日成交量均值",
    "近5日成交额均值",
    "近5日换手率均值",
    "近5日涨跌幅波动率",
    "近20日收益率",
    "近20日成交量均值",
    "近20日成交额均值",
    "近20日换手率均值",
    "近20日涨跌幅波动率",
    "近60日收益率",
    "近60日成交量均值",
    "近60日成交额均值",
    "近60日换手率均值",
    "近60日涨跌幅波动率",
    "行业来源日期",
    "原始行业",
    "行业分类口径",
    "板块划分",
]

STEP1_SECTOR_COLUMNS = [
    "板块划分",
    "股票数量",
    "最新成交额合计",
    "最新换手率均值",
    "近5日收益率均值",
    "近20日收益率均值",
    "近60日收益率均值",
]


SECTOR_BY_INDUSTRY_PREFIX = {
    "A03": "消费",
    "C13": "消费",
    "C14": "消费",
    "C15": "消费",
    "C19": "消费",
    "C27": "消费",
    "F52": "消费",
    "L72": "消费",
    "M73": "消费",
    "Q84": "消费",
    "C39": "科技（TMT）",
    "I63": "科技（TMT）",
    "I64": "科技（TMT）",
    "I65": "科技（TMT）",
    "R87": "科技（TMT）",
    "J66": "金融地产",
    "J67": "金融地产",
    "J68": "金融地产",
    "J69": "金融地产",
    "K70": "金融地产",
    "B06": "周期",
    "B07": "周期",
    "B09": "周期",
    "B11": "周期",
    "C26": "周期",
    "C29": "周期",
    "C30": "周期",
    "C31": "周期",
    "C32": "周期",
    "C33": "制造",
    "C34": "制造",
    "C35": "制造",
    "C36": "制造",
    "C37": "制造",
    "C38": "制造",
    "D44": "基建与公用",
    "D45": "基建与公用",
    "E48": "基建与公用",
    "G53": "基建与公用",
    "G54": "基建与公用",
    "G55": "基建与公用",
    "G56": "基建与公用",
    "G60": "基建与公用",
}

SECTOR_BY_STOCK_CODE = {
    "002027": "科技（TMT）",
}


def normalize_code(value: object) -> str:
    text = str(value).strip()
    text = text.replace("sh.", "").replace("sz.", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def classify_sector(stock_code: str, industry: object) -> str:
    if stock_code in SECTOR_BY_STOCK_CODE:
        return SECTOR_BY_STOCK_CODE[stock_code]
    prefix = str(industry or "").strip()[:3]
    return SECTOR_BY_INDUSTRY_PREFIX.get(prefix, "未匹配")


def read_hs300(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "hs300_stocks.csv", dtype=str)
    required = {"updateDate", "code", "code_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"hs300_stocks.csv missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "股票代码": df["code"].map(normalize_code),
            "股票名称": df["code_name"].str.strip(),
            "成分股更新日期": df["updateDate"].str.strip(),
        }
    ).drop_duplicates("股票代码", keep="last")


def read_industry(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "stock_industry.csv", dtype=str)
    required = {"updateDate", "code", "industry", "industryClassification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stock_industry.csv missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "股票代码": df["code"].map(normalize_code),
            "行业来源日期": df["updateDate"].fillna("").str.strip(),
            "原始行业": df["industry"].fillna("").str.strip(),
            "行业分类口径": df["industryClassification"].fillna("").str.strip(),
        }
    )
    return out.drop_duplicates("股票代码", keep="last")


def read_daily(raw_dir: Path, hs300_codes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "daily_price_volume.csv", dtype={"code": str})
    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "pctChg",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"daily_price_volume.csv missing columns: {sorted(missing)}")

    df["股票代码"] = df["code"].map(normalize_code)
    df = df[df["股票代码"].isin(hs300_codes)].copy()
    df["日期_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["日期_dt"]).copy()

    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["日期"] = df["日期_dt"].dt.strftime("%Y-%m-%d")
    df["振幅"] = ((df["high"] - df["low"]) / df["preclose"].replace(0, pd.NA)) * 100
    df["涨跌额"] = df["close"] - df["preclose"]

    out = pd.DataFrame(
        {
            "股票代码": df["股票代码"],
            "日期": df["日期"],
            "开盘": df["open"],
            "收盘": df["close"],
            "最高": df["high"],
            "最低": df["low"],
            "成交量": df["volume"],
            "成交额": df["amount"],
            "振幅": df["振幅"],
            "涨跌额": df["涨跌额"],
            "换手率": df["turn"],
            "涨跌幅": df["pctChg"],
            "日期_dt": df["日期_dt"],
        }
    )
    out = out.drop_duplicates(["股票代码", "日期"], keep="last")
    return out.sort_values(["股票代码", "日期_dt"]).reset_index(drop=True)


def window_return_pct(group: pd.DataFrame, window: int) -> float | None:
    window_df = group.tail(window)
    if len(window_df) < 2:
        return None
    first_close = window_df["收盘"].iloc[0]
    last_close = window_df["收盘"].iloc[-1]
    if pd.isna(first_close) or pd.isna(last_close) or first_close == 0:
        return None
    return (last_close / first_close - 1) * 100


def summarize_stock(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("日期_dt")
    latest = group.iloc[-1]
    row: dict[str, object] = {
        "raw_起始日期": group["日期"].min(),
        "raw_结束日期": group["日期"].max(),
        "raw_交易日数": int(len(group)),
        "最新日期": latest["日期"],
        "最新开盘": latest["开盘"],
        "最新收盘": latest["收盘"],
        "最新最高": latest["最高"],
        "最新最低": latest["最低"],
        "最新成交量": latest["成交量"],
        "最新成交额": latest["成交额"],
        "最新振幅": latest["振幅"],
        "最新涨跌额": latest["涨跌额"],
        "最新换手率": latest["换手率"],
        "最新涨跌幅": latest["涨跌幅"],
    }
    for window in (5, 20, 60):
        window_df = group.tail(window)
        row[f"近{window}日收益率"] = window_return_pct(group, window)
        row[f"近{window}日成交量均值"] = window_df["成交量"].mean()
        row[f"近{window}日成交额均值"] = window_df["成交额"].mean()
        row[f"近{window}日换手率均值"] = window_df["换手率"].mean()
        row[f"近{window}日涨跌幅波动率"] = window_df["涨跌幅"].std()
    return row


def build_stock_summary(daily: pd.DataFrame, hs300: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stock_code, group in daily.groupby("股票代码", sort=True):
        row = {"股票代码": stock_code}
        row.update(summarize_stock(group))
        rows.append(row)

    summary = pd.DataFrame(rows)
    out = hs300.merge(summary, on="股票代码", how="left")
    out = out.merge(industry, on="股票代码", how="left")
    out["原始行业"] = out["原始行业"].fillna("")
    out["行业分类口径"] = out["行业分类口径"].fillna("")
    out["行业来源日期"] = out["行业来源日期"].fillna("")
    out["板块划分"] = [
        classify_sector(stock_code, industry_name)
        for stock_code, industry_name in zip(out["股票代码"], out["原始行业"], strict=False)
    ]
    return out[STEP1_STOCK_COLUMNS]


def build_sector_summary(stock_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = stock_summary.groupby("板块划分", dropna=False)
    out = grouped.agg(
        股票数量=("股票代码", "count"),
        最新成交额合计=("最新成交额", "sum"),
        最新换手率均值=("最新换手率", "mean"),
        近5日收益率均值=("近5日收益率", "mean"),
        近20日收益率均值=("近20日收益率", "mean"),
        近60日收益率均值=("近60日收益率", "mean"),
    ).reset_index()
    return out[STEP1_SECTOR_COLUMNS].sort_values("板块划分").reset_index(drop=True)


def build_manifest(raw_dir: Path, daily: pd.DataFrame, stock_summary: pd.DataFrame, note: str | None) -> pd.DataFrame:
    latest_t = "" if daily.empty else str(daily["日期"].max())
    date_start = "" if daily.empty else str(daily["日期"].min())
    unique_dates = 0 if daily.empty else int(daily["日期"].nunique())
    unmatched_count = int(stock_summary["板块划分"].eq("未匹配").sum())
    items = [
        ("schema_version", "workflow_0.1_csv_v1"),
        ("date_start", date_start),
        ("date_end", latest_t),
        ("latest_T", latest_t),
        ("raw_交易日数", str(unique_dates)),
        ("data_source", "baostock: hs300_stocks.csv, daily_price_volume.csv; baostock stock_industry.csv historical raw"),
        ("raw_dir", str(raw_dir)),
        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("stock_count", str(len(stock_summary))),
        ("unmatched_sector_count", str(unmatched_count)),
        ("data_window_note", note or "使用当前 raw 数据窗口生成 Step-1 标准输出。"),
    ]
    return pd.DataFrame(items, columns=["项目", "说明"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].round(6)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_step1_outputs(raw_dir: Path, output_dir: Path, note: str | None = None) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hs300 = read_hs300(raw_dir)
    industry = read_industry(raw_dir)
    daily = read_daily(raw_dir, set(hs300["股票代码"]))
    daily_output = daily[STEP1_DAILY_COLUMNS]
    stock_summary = build_stock_summary(daily, hs300, industry)
    sector_summary = build_sector_summary(stock_summary)
    manifest = build_manifest(raw_dir, daily, stock_summary, note)

    outputs = {
        "daily": output_dir / "step1_daily_raw_data.csv",
        "stock": output_dir / "step1_stock_summary.csv",
        "sector": output_dir / "step1_sector_summary.csv",
        "manifest": output_dir / "step1_data_manifest.csv",
    }
    write_csv(daily_output, outputs["daily"])
    write_csv(stock_summary, outputs["stock"])
    write_csv(sector_summary, outputs["sector"])
    manifest.to_csv(outputs["manifest"], index=False, encoding="utf-8-sig")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build workflow_0.1 Step-1 standard CSV outputs.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_step1_outputs(raw_dir=args.raw_dir, output_dir=args.output_dir, note=args.note)
    print(f"output_dir: {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

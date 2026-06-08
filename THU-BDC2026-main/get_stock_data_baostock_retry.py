#!/usr/bin/env python3
"""
Retry runner for the official Baostock stock-data workflow.

It keeps the same data source, fields, date window, and output format as
get_stock_data.py, but adds retries around Baostock network calls so a
temporary connection reset does not stop the whole Step-1 data refresh.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd


START_DATE = "2026-01-01"
END_DATE = "2026-06-07"
SAVE_DIR = Path("./data")
OUTPUT_PATH = SAVE_DIR / "stock_data.csv"
HS300_LIST_PATH = SAVE_DIR / "hs300_stock_list.csv"
FAILED_PATH = SAVE_DIR / "failed_stocks.csv"
MAX_LOGIN_ATTEMPTS = 5
MAX_QUERY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 8


def retry_call(label, func, attempts=MAX_QUERY_ATTEMPTS, sleep_seconds=RETRY_SLEEP_SECONDS, recover=None):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            print(f"{label} 第 {attempt}/{attempts} 次失败: {exc}", flush=True)
            if recover is not None and attempt < attempts:
                recover(exc)
            if attempt < attempts:
                time.sleep(sleep_seconds)
    raise last_error


def login():
    def do_login():
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"登录失败: {lg.error_msg}")
        return lg

    lg = retry_call(
        "baostock登录",
        do_login,
        attempts=MAX_LOGIN_ATTEMPTS,
        sleep_seconds=RETRY_SLEEP_SECONDS,
    )
    print("baostock登录成功", flush=True)
    return lg


def logout():
    bs.logout()
    print("baostock已登出", flush=True)


def normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("sh.", "", regex=False).str.replace("sz.", "", regex=False).str.zfill(6)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def format_official_date(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y/%m/%d").str.replace(r"/0([1-9])", r"/\1", regex=True)


def recover_baostock_session(exc: Exception):
    message = str(exc)
    if "用户未登录" not in message and "网络接收错误" not in message:
        return
    print("  检测到 Baostock 会话异常，重新登录后重试当前查询", flush=True)
    try:
        bs.logout()
    except Exception:
        pass
    try:
        login()
    except Exception as login_exc:
        print(f"  重新登录暂时失败: {login_exc}", flush=True)


def get_hs300_stocks():
    def do_query():
        print("正在获取沪深300成分股列表...", flush=True)
        rs = bs.query_hs300_stocks()
        if rs.error_code != "0":
            raise RuntimeError(f"获取成分股失败: {rs.error_msg}")
        stocks = []
        while (rs.error_code == "0") & rs.next():
            stocks.append(rs.get_row_data())
        df = pd.DataFrame(stocks, columns=rs.fields)
        if df.empty:
            raise RuntimeError("获取成分股失败: 返回为空")
        return df

    df = retry_call("沪深300成分股查询", do_query)
    print(f"获取到 {len(df)} 只沪深300成分股", flush=True)
    return df


def get_stock_history(bs_code, start_date, end_date):
    def do_query():
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"查询失败: {rs.error_msg}")

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            return None
        return pd.DataFrame(data_list, columns=rs.fields)

    df = retry_call(f"{bs_code} 历史行情查询", do_query, recover=recover_baostock_session)
    if df is None:
        return None

    numeric_cols = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["振幅"] = ((df["high"] - df["low"]) / df["preclose"] * 100).round(2)
    df["涨跌额"] = (df["close"] - df["preclose"]).round(2)
    df["date"] = format_official_date(pd.to_datetime(df["date"]))
    df["code"] = normalize_code(df["code"])
    df = df.rename(
        columns={
            "code": "股票代码",
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
            "turn": "换手率",
            "pctChg": "涨跌幅",
        }
    )

    return df[["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅"]]


def read_existing_data():
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig", dtype={"股票代码": str})
    if df.empty:
        return df
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期_dt"] = parse_dates(df["日期"])
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    df = df.dropna(subset=["日期_dt"])
    df = df[(df["日期_dt"] >= start_dt) & (df["日期_dt"] <= end_dt)].copy()
    df = df.drop(columns=["日期_dt"])
    return df


def get_stock_date_range(existing_df, stock_code):
    if existing_df.empty:
        return None, None
    stock_df = existing_df[existing_df["股票代码"].astype(str).str.zfill(6) == stock_code].copy()
    if stock_df.empty:
        return None, None
    stock_df["日期_dt"] = parse_dates(stock_df["日期"])
    stock_df = stock_df.dropna(subset=["日期_dt"])
    if stock_df.empty:
        return None, None
    return stock_df["日期_dt"].min().strftime("%Y-%m-%d"), stock_df["日期_dt"].max().strftime("%Y-%m-%d")


def fetch_ranges_for_stock(existing_min_date, existing_max_date):
    if existing_min_date is None or existing_max_date is None:
        return [(START_DATE, END_DATE, "全量")]

    ranges = []
    if existing_min_date > START_DATE:
        fetch_end = (datetime.strptime(existing_min_date, "%Y-%m-%d") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ranges.append((START_DATE, fetch_end, "早期"))
    if existing_max_date < END_DATE:
        late_start = datetime.strptime(existing_max_date, "%Y-%m-%d") + pd.Timedelta(days=1)
        fetch_start = max(pd.to_datetime(START_DATE), pd.to_datetime(late_start)).strftime("%Y-%m-%d")
        ranges.append((fetch_start, END_DATE, "近期"))
    return ranges


def merge_stock_data(existing_df, new_df, stock_code):
    if new_df is None or new_df.empty:
        return existing_df
    if existing_df.empty:
        return new_df

    existing = existing_df.copy()
    existing["股票代码_str"] = existing["股票代码"].astype(str).str.zfill(6)
    other_df = existing[existing["股票代码_str"] != stock_code].drop(columns=["股票代码_str"])
    stock_existing = existing[existing["股票代码_str"] == stock_code].drop(columns=["股票代码_str"])

    if stock_existing.empty:
        combined = new_df
    else:
        stock_existing = stock_existing.copy()
        new_df = new_df.copy()
        stock_existing["日期_dt"] = parse_dates(stock_existing["日期"])
        new_df["日期_dt"] = parse_dates(new_df["日期"])
        combined = pd.concat([stock_existing, new_df], ignore_index=True)
        combined = combined.dropna(subset=["日期_dt"])
        combined = combined.drop_duplicates(subset=["日期_dt"], keep="last")
        combined = combined.sort_values("日期_dt")
        combined = combined.drop(columns=["日期_dt"])

    return pd.concat([other_df, combined], ignore_index=True)


def save_stock_data(df):
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期_dt"] = parse_dates(df["日期"])
    df = df.dropna(subset=["日期_dt"])
    df = df.sort_values(["股票代码", "日期_dt"])
    df["日期"] = format_official_date(df["日期_dt"])
    df = df.drop(columns=["日期_dt"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    return df


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"目标数据时间范围: {START_DATE} 至 {END_DATE}", flush=True)
    print(f"输出文件: {OUTPUT_PATH}", flush=True)
    print("=" * 60, flush=True)

    existing_df = read_existing_data()
    if not existing_df.empty:
        print(f"已加载目标区间内现有数据: {len(existing_df)} 条，{existing_df['股票代码'].nunique()} 只股票", flush=True)

    login()
    failed_stocks = []
    total_new_records = 0
    updated_count = 0

    try:
        hs300_df = get_hs300_stocks()
        hs300_df.to_csv(HS300_LIST_PATH, index=False, encoding="utf-8-sig")
        hs300_df["纯代码"] = normalize_code(hs300_df["code"])

        total = len(hs300_df)
        for idx, row in hs300_df.iterrows():
            bs_code = row.get("code", "")
            stock_name = row.get("code_name", "")
            pure_code = row.get("纯代码", "")
            existing_min_date, existing_max_date = get_stock_date_range(existing_df, pure_code)
            fetch_ranges = fetch_ranges_for_stock(existing_min_date, existing_max_date)

            if not fetch_ranges:
                print(f"[{idx + 1}/{total}] {bs_code} {stock_name} - 数据已完整 ({existing_min_date} 至 {existing_max_date})，跳过", flush=True)
                continue

            print(f"[{idx + 1}/{total}] {bs_code} {stock_name} - 待补区间: {fetch_ranges}", flush=True)
            try:
                fetched = []
                for fetch_start, fetch_end, period_name in fetch_ranges:
                    print(f"  获取{period_name}数据: {fetch_start} 至 {fetch_end}", flush=True)
                    stock_data = get_stock_history(bs_code, fetch_start, fetch_end)
                    if stock_data is not None and not stock_data.empty:
                        fetched.append(stock_data)

                if fetched:
                    new_data = pd.concat(fetched, ignore_index=True)
                    existing_df = merge_stock_data(existing_df, new_data, pure_code)
                    existing_df = save_stock_data(existing_df)
                    total_new_records += len(new_data)
                    updated_count += 1
                    print(f"  获取成功，新增 {len(new_data)} 条记录", flush=True)
                else:
                    print("  无新数据", flush=True)
            except Exception as exc:
                print(f"  失败: {exc}", flush=True)
                failed_stocks.append((bs_code, stock_name, str(exc)))

            if updated_count > 0 and updated_count % 10 == 0:
                time.sleep(2)
    finally:
        logout()

    if failed_stocks:
        pd.DataFrame(failed_stocks, columns=["股票代码", "股票名称", "错误"]).to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")
        print(f"失败股票列表已保存至: {FAILED_PATH}", flush=True)

    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig", dtype={"股票代码": str})
        dt = parse_dates(df["日期"])
        print("=" * 60, flush=True)
        print("运行完成", flush=True)
        print(f"新增记录: {total_new_records}", flush=True)
        print(f"总行数: {len(df)}", flush=True)
        print(f"股票数量: {df['股票代码'].nunique()}", flush=True)
        print(f"时间范围: {dt.min().strftime('%Y-%m-%d')} 至 {dt.max().strftime('%Y-%m-%d')}", flush=True)


if __name__ == "__main__":
    main()

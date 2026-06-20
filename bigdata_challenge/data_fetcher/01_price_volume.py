"""
01_price_volume.py
====================
优先级：P0（最核心）
数据源：baostock（完全免费，无需注册token）
获取内容：沪深300成分股的日K线数据（开高低收、成交量、成交额、换手率）

为什么这是最重要的：
- 量价数据是所有技术因子的基础
- 短期动量、波动率、换手率异动等P0因子全部从这里衍生
- baostock数据质量高，覆盖完整，且是比赛基准代码推荐的数据源
"""

import baostock as bs
import pandas as pd
import time
import os
import signal
from config import START_DATE, END_DATE, RAW_DIR, HS300_CODE


QUERY_TIMEOUT_SECONDS = 45


class QueryTimeout(TimeoutError):
    """baostock 单次请求超时。"""


def run_with_timeout(func, timeout_seconds=QUERY_TIMEOUT_SECONDS):
    """在主线程中给阻塞式 baostock socket 调用加超时保护。"""
    if not hasattr(signal, "SIGALRM"):
        return func()

    def _handle_timeout(signum, frame):
        raise QueryTimeout(f"baostock query timed out after {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def get_stock_last_dates(csv_path, code_col='code', date_col='date'):
    """读取每只股票在本地CSV中的最大日期，文件不存在返回空字典。"""
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, usecols=[code_col, date_col], dtype={code_col: str})
        if df.empty:
            return {}
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        if df.empty:
            return {}
        last_dates = df.groupby(code_col)[date_col].max()
        return {code: date.strftime('%Y-%m-%d') for code, date in last_dates.items()}
    except Exception:
        return {}


def get_update_start_for_code(stock_code, last_dates, default_start):
    """返回单只股票的增量起始日；新成分股从默认起始日开始补历史。"""
    last_date = last_dates.get(stock_code)
    if not last_date:
        return default_start
    return (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def should_skip_local_latest(stock_code, last_dates):
    """当本地进度混合时，跳过已经到达本地最新交易日的股票。"""
    if stock_code not in last_dates:
        return False

    unique_dates = {date for date in last_dates.values() if date}
    if len(unique_dates) <= 1:
        return False

    return last_dates[stock_code] == max(unique_dates)


def reset_baostock_connection():
    """baostock socket 超时后容易进入坏连接状态，失败后重连再继续下一只股票。"""
    try:
        bs.logout()
    except Exception:
        pass
    try:
        bs.login()
    except Exception as exc:
        print(f"baostock 重连失败: {exc}")


def get_hs300_stocks():
    """获取沪深300成分股列表"""
    lg = bs.login()
    rs = bs.query_hs300_stocks(date=END_DATE)

    stocks = []
    while rs.error_code == '0' and rs.next():
        stocks.append(rs.get_row_data())

    df = pd.DataFrame(stocks, columns=rs.fields)
    bs.logout()

    print(f"获取到 {len(df)} 只沪深300成分股")
    df.to_csv(os.path.join(RAW_DIR, "hs300_stocks.csv"), index=False)
    return df


def get_stock_daily(stock_code, start_date, end_date):
    """
    获取单只股票的日K线数据

    返回字段：
    - date: 日期
    - open/high/low/close: 开高低收（前复权）
    - volume: 成交量（股）
    - amount: 成交额（元）
    - turn: 换手率(%)
    - peTTM: 滚动市盈率
    - pbMRQ: 市净率
    - psTTM: 滚动市销率
    - isST: 是否ST
    """
    rs = run_with_timeout(
        lambda: bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,"
            "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",  # 日线
            adjustflag="2"  # 前复权（重要！保证价格连续性）
        )
    )

    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=rs.fields)

    # 类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                    'volume', 'amount', 'turn', 'pctChg',
                    'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'])

    # 过滤停牌日（tradestatus=0表示停牌）
    df = df[df['tradestatus'] == '1'].copy()

    return df


def fetch_all_stocks():
    """批量获取所有沪深300成分股的日K线数据（支持增量更新）"""
    from utils import append_csv

    stocks_df = get_hs300_stocks()
    stock_codes = stocks_df['code'].tolist()

    output_path = os.path.join(RAW_DIR, "daily_price_volume.csv")
    last_dates = get_stock_last_dates(output_path, code_col='code', date_col='date')
    if last_dates:
        global_last_date = max(last_dates.values())
        print(f"按单只股票增量更新，本地全局最新日期: {global_last_date}")
    else:
        print(f"全量下载: {START_DATE} ~ {END_DATE}")

    lg = bs.login()
    failed = []
    total_added = 0

    try:
        for i, code in enumerate(stock_codes):
            if should_skip_local_latest(code, last_dates):
                print(f"[{i+1}/{len(stock_codes)}] {code}: 已至本地最新交易日 {last_dates[code]}")
                continue

            start = get_update_start_for_code(code, last_dates, START_DATE)
            if start > END_DATE:
                print(f"[{i+1}/{len(stock_codes)}] {code}: 已最新")
                continue

            try:
                df = get_stock_daily(code, start, END_DATE)
                if not df.empty:
                    append_csv(output_path, df)
                    total_added += len(df)
                    last_dates[code] = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                    print(f"[{i+1}/{len(stock_codes)}] {code}: {len(df)} 条 ({start} ~ {END_DATE})")
                else:
                    print(f"[{i+1}/{len(stock_codes)}] {code}: 无新增数据 ({start} ~ {END_DATE})")
            except Exception as e:
                failed.append(code)
                print(f"[{i+1}/{len(stock_codes)}] {code} 失败: {e}")
                reset_baostock_connection()

            if i % 50 == 0 and i > 0:
                time.sleep(1)
    finally:
        bs.logout()

    if total_added:
        print(f"\n增量追加 {total_added} 条记录")

    if failed:
        print(f"失败股票 ({len(failed)}): {failed[:10]}...")

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    if not total_added:
        print("无新增数据")
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取沪深300成分股日K线数据")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)
    fetch_all_stocks()

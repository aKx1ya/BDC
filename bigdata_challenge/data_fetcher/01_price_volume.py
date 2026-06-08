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
from config import START_DATE, END_DATE, RAW_DIR, HS300_CODE


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
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,"
        "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",  # 日线
        adjustflag="2"  # 前复权（重要！保证价格连续性）
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
    from utils import get_last_date, append_csv, next_day

    stocks_df = get_hs300_stocks()
    stock_codes = stocks_df['code'].tolist()

    output_path = os.path.join(RAW_DIR, "daily_price_volume.csv")
    last_date = get_last_date(output_path, date_col='date')

    if last_date:
        start = next_day(last_date)
        if start > END_DATE:
            print(f"数据已是最新（至{last_date}），无需更新")
            return pd.read_csv(output_path)
        print(f"增量更新: {start} ~ {END_DATE}")
    else:
        start = START_DATE
        print(f"全量下载: {start} ~ {END_DATE}")

    lg = bs.login()
    all_data = []
    failed = []

    for i, code in enumerate(stock_codes):
        try:
            df = get_stock_daily(code, start, END_DATE)
            if not df.empty:
                all_data.append(df)
                print(f"[{i+1}/{len(stock_codes)}] {code}: {len(df)} 条")
            else:
                failed.append(code)
        except Exception as e:
            failed.append(code)
            print(f"[{i+1}/{len(stock_codes)}] {code} 失败: {e}")

        if i % 50 == 0 and i > 0:
            time.sleep(1)

    bs.logout()

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        if last_date:
            append_csv(output_path, result)
            print(f"\n增量追加 {len(result)} 条记录")
        else:
            result.to_csv(output_path, index=False)
            print(f"\n全量保存 {len(result)} 条记录")

        if failed:
            print(f"失败股票 ({len(failed)}): {failed[:10]}...")
        return result
    else:
        print("无新增数据")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取沪深300成分股日K线数据")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)
    fetch_all_stocks()

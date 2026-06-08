"""
03_margin_trading.py
====================
优先级：P1（重要指标）
数据源：akshare（免费）
获取内容：融资融券数据 - 融资余额、融券余额、融资买入额

为什么重要（排在P1）：
- 融资余额代表散户加杠杆的意愿，是最直接的市场情绪温度计
- 融资余额急速上升 → 市场过热，短期回调风险增大
- 融资余额持续下降 → 市场悲观，可能接近底部
- 个股融资余额变动率 vs 市场整体融资变动率 → 个股相对情绪强度
- 融券余额增加 → 做空力量增强，但A股融券量小，参考价值有限

关键衍生特征：
- 融资余额5日变动率（个股）
- 融资买入额/成交额占比（杠杆活跃度）
- 融资余额偏离度（当前vs20日均值）
"""

import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime
from config import RAW_DIR


def get_market_margin_summary():
    """获取全市场融资融券汇总数据（增量更新）"""
    from utils import retry_request, get_last_date

    print("获取全市场融资融券汇总...")

    @retry_request
    def _fetch_sh(start):
        return ak.stock_margin_sse(start_date=start)

    @retry_request
    def _fetch_sz(start):
        return ak.stock_margin_szse(start_date=start)

    for fetch_fn, market, fname in [
        (_fetch_sh, 'SH', "margin_summary_sh.csv"),
        (_fetch_sz, 'SZ', "margin_summary_sz.csv"),
    ]:
        output_path = os.path.join(RAW_DIR, fname)
        last = get_last_date(output_path, date_col='信用交易日期')
        start = last.replace("-", "") if last else "20230101"

        df = fetch_fn(start)
        if df is not None and not df.empty:
            df['market'] = market
            if last:
                df.to_csv(output_path, mode='a', header=False, index=False)
                print(f"  {market}: 增量追加 {len(df)} 条")
            else:
                df.to_csv(output_path, index=False)
                print(f"  {market}: 全量 {len(df)} 条")
        else:
            print(f"  {market}: 无新数据或获取失败")


def get_stock_margin_detail(date_str=None):
    """获取个股融资融券明细（某一天）"""
    from utils import retry_request

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    @retry_request
    def _fetch_sh():
        return ak.stock_margin_detail_sse(date=date_str)

    @retry_request
    def _fetch_sz():
        return ak.stock_margin_detail_szse(date=date_str)

    frames = []
    df_sh = _fetch_sh()
    if df_sh is not None and not df_sh.empty:
        df_sh['market'] = 'SH'
        frames.append(df_sh)
    df_sz = _fetch_sz()
    if df_sz is not None and not df_sz.empty:
        df_sz['market'] = 'SZ'
        frames.append(df_sz)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_margin_history(days=60):
    """获取最近N个交易日个股融资融券数据（增量更新）"""
    from utils import get_last_date, append_csv

    output_path = os.path.join(RAW_DIR, "margin_stock_detail.csv")
    last_date = get_last_date(output_path, date_col='date')

    print(f"获取个股融资融券数据...")

    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        trade_cal['trade_date'] = pd.to_datetime(trade_cal['trade_date'])
        today = pd.Timestamp(datetime.now().date())
        dates = trade_cal[trade_cal['trade_date'] <= today].tail(days)['trade_date'].tolist()
    except Exception:
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B').tolist()

    if last_date:
        cutoff = pd.Timestamp(last_date)
        dates = [d for d in dates if d > cutoff]
        if not dates:
            print(f"  数据已是最新（至{last_date}）")
            return pd.DataFrame()
        print(f"  增量更新: {len(dates)} 天")

    all_data = []
    for i, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")
        df = get_stock_margin_detail(date_str)
        if not df.empty:
            df['date'] = date.strftime('%Y-%m-%d')
            all_data.append(df)
            print(f"  [{i+1}/{len(dates)}] {date_str}: {len(df)} 只")
        time.sleep(0.3)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        append_csv(output_path, result)
        print(f"\n个股融资融券: 追加 {len(result)} 条")
        return result

    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取融资融券数据")
    print("=" * 60)

    # 1. 全市场汇总
    get_market_margin_summary()
    print()

    # 2. 个股明细（首次建议先获取30天）
    get_margin_history(days=30)

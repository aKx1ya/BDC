"""
02_northbound_flow.py
====================
优先级：P0（核心指标）
数据源：akshare（免费，无需token）
获取内容：沪深港通北向资金 - 个股级别的每日净买入

为什么重要（排在P0）：
- 北向资金被称为"聪明钱"，其净买入方向对未来5日收益有显著正向预测力
- 学术研究表明北向资金的信息含量远高于融资融券
- 北向资金是每日更新的，频率足够支撑5日预测
- 在A股量化因子体系中，北向因子是近年来最有效的alpha因子之一

获取方式：
- akshare提供每日北向资金个股持股数据
- 可以计算：净买入金额、持股比例变化、连续流入天数等
"""

import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime
from config import RAW_DIR


def get_northbound_daily_summary():
    """获取北向资金每日总流入（沪股通+深股通）"""
    from utils import retry_request
    print("获取北向资金每日汇总...")

    @retry_request
    def _fetch(symbol):
        return ak.stock_hsgt_hist_em(symbol=symbol)

    for symbol, fname in [("沪股通", "northbound_sh_hist.csv"),
                          ("深股通", "northbound_sz_hist.csv")]:
        df = _fetch(symbol)
        if df is not None and not df.empty:
            output_path = os.path.join(RAW_DIR, fname)
            df.to_csv(output_path, index=False)
            print(f"  {symbol}: {len(df)} 条")
        else:
            print(f"  {symbol}: 获取失败")


def get_northbound_stock_holding(date_str=None):
    """获取北向资金个股持股明细（某一天）"""
    from utils import retry_request

    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    @retry_request
    def _fetch(market):
        return ak.stock_hsgt_hold_stock_em(market=market, date=date_str)

    try:
        df_sh = _fetch("沪股通")
        df_sz = _fetch("深股通")
        frames = []
        if df_sh is not None and not df_sh.empty:
            df_sh['market'] = '沪股通'
            frames.append(df_sh)
        if df_sz is not None and not df_sz.empty:
            df_sz['market'] = '深股通'
            frames.append(df_sz)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as e:
        print(f"获取 {date_str} 北向持股明细失败: {e}")
        return pd.DataFrame()


def get_northbound_stock_history(days=120):
    """获取最近N个交易日的北向资金个股持股变化（增量更新）"""
    from utils import get_last_date, append_csv

    output_path = os.path.join(RAW_DIR, "northbound_stock_holding.csv")
    last_date = get_last_date(output_path, date_col='date')

    print(f"获取北向个股持股数据...")
    trade_dates = get_recent_trade_dates(days)

    if last_date:
        cutoff = pd.Timestamp(last_date)
        trade_dates = [d for d in trade_dates if d > cutoff]
        if not trade_dates:
            print(f"  数据已是最新（至{last_date}）")
            return pd.DataFrame()
        print(f"  增量更新: 从 {trade_dates[0].strftime('%Y-%m-%d')} 起，共{len(trade_dates)}天")

    all_data = []
    for i, date in enumerate(trade_dates):
        date_str = date.strftime("%Y%m%d")
        df = get_northbound_stock_holding(date_str)

        if not df.empty:
            df['date'] = date.strftime('%Y-%m-%d')
            all_data.append(df)
            print(f"  [{i+1}/{len(trade_dates)}] {date_str}: {len(df)} 只")

        time.sleep(0.5)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        append_csv(output_path, result)
        print(f"\n北向个股持股: 追加 {len(result)} 条")
        return result

    return pd.DataFrame()


def get_recent_trade_dates(days=120):
    """获取最近N个交易日列表"""
    try:
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        today = pd.Timestamp(datetime.now().date())
        past_dates = df[df['trade_date'] <= today].tail(days)
        return past_dates['trade_date'].tolist()
    except Exception as e:
        print(f"获取交易日历失败: {e}")
        # 回退方案：简单生成日期列表（包含非交易日，后续会自动过滤）
        dates = pd.date_range(
            end=datetime.now(),
            periods=days * 2,
            freq='B'  # 工作日
        )
        return dates.tolist()[-days:]


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取北向资金数据")
    print("=" * 60)

    # 1. 获取每日汇总（用于市场情绪）
    get_northbound_daily_summary()

    print()

    # 2. 获取个股持股明细（用于个股因子）
    # 注意：历史数据量大，首次运行建议先获取最近30天
    get_northbound_stock_history(days=30)

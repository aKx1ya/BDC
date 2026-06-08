"""
06_fundamental.py
====================
优先级：P3（辅助指标，低频更新但事件驱动时影响大）
数据源：baostock（免费）+ akshare（免费）
获取内容：基本面指标 - PE/PB分位数、ROE变化、营收增速、业绩预告

为什么优先级相对较低（P3）：
- 基本面数据更新频率低（季报才更新），对5日收益的直接预测力有限
- 但在特定时间窗口（财报发布前后），基本面因素会主导股价
- PE/PB的极端分位数对短期反转有预测力（过度低估→反弹）
- 业绩预告超预期是强事件信号，发布后5日内有显著超额收益

使用策略：
- PE/PB分位数：作为价值因子的代理，取当前值在过去252个交易日的百分位
- ROE环比变化：最近一期ROE vs 上一期ROE，反映盈利改善方向
- 营收增速：TTM同比增速，高增长+动量共振时效果更强
- 业绩预告：作为事件因子，在预告发布后5个交易日内生效

关键衍生特征：
- PE_percentile_252d: 市盈率在过去一年的百分位（0-1）
- PB_percentile_252d: 市净率在过去一年的百分位（0-1）
- ROE_change: ROE环比变化值
- revenue_growth_ttm: TTM营收同比增速
"""

import baostock as bs
import akshare as ak
import pandas as pd
import os
import time
from config import RAW_DIR


def get_fundamental_quarterly():
    """获取沪深300成分股的季度基本面数据"""
    from utils import retry_request

    print("获取季度基本面数据...")

    stocks_path = os.path.join(RAW_DIR, "hs300_stocks.csv")
    if not os.path.exists(stocks_path):
        print("  请先运行 01_price_volume.py 获取成分股列表")
        return pd.DataFrame()

    stocks = pd.read_csv(stocks_path)['code'].tolist()
    lg = bs.login()

    all_data = []
    years = [2023, 2024, 2025, 2026]
    quarters = [1, 2, 3, 4]

    for year in years:
        for quarter in quarters:
            for code in stocks:
                try:
                    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    data = []
                    while rs.error_code == '0' and rs.next():
                        data.append(rs.get_row_data())
                    if data:
                        df = pd.DataFrame(data, columns=rs.fields)
                        df['year'] = year
                        df['quarter'] = quarter
                        all_data.append(df)
                except Exception:
                    pass
            print(f"  {year}Q{quarter} 完成")
            time.sleep(0.5)

    bs.logout()

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(RAW_DIR, "fundamental_quarterly.csv")
        result.to_csv(output_path, index=False)
        print(f"\n季度基本面: {len(result)} 条")
        return result

    return pd.DataFrame()


def get_growth_data():
    """获取成长能力指标"""
    print("获取成长能力数据...")

    stocks_path = os.path.join(RAW_DIR, "hs300_stocks.csv")
    if not os.path.exists(stocks_path):
        print("  请先运行 01_price_volume.py")
        return pd.DataFrame()

    stocks = pd.read_csv(stocks_path)['code'].tolist()
    lg = bs.login()

    all_data = []
    years = [2023, 2024, 2025, 2026]
    quarters = [1, 2, 3, 4]

    for year in years:
        for quarter in quarters:
            for code in stocks:
                try:
                    rs = bs.query_growth_data(code=code, year=year, quarter=quarter)
                    data = []
                    while rs.error_code == '0' and rs.next():
                        data.append(rs.get_row_data())
                    if data:
                        df = pd.DataFrame(data, columns=rs.fields)
                        df['year'] = year
                        df['quarter'] = quarter
                        all_data.append(df)
                except Exception:
                    pass
            print(f"  成长性 {year}Q{quarter} 完成")
            time.sleep(0.5)

    bs.logout()

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(RAW_DIR, "fundamental_growth.csv")
        result.to_csv(output_path, index=False)
        print(f"\n成长能力: {len(result)} 条")
        return result

    return pd.DataFrame()


def get_earnings_forecast():
    """获取业绩预告数据"""
    from utils import retry_request

    print("获取业绩预告数据...")

    @retry_request
    def _fetch(date):
        return ak.stock_yjyg_em(date=date)

    for date in ["20260331", "20251231", "20250930", "20250630"]:
        df = _fetch(date)
        if df is not None and not df.empty:
            output_path = os.path.join(RAW_DIR, "earnings_forecast.csv")
            df.to_csv(output_path, index=False)
            print(f"  业绩预告({date}): {len(df)} 条")
            return df

    print("  所有日期均获取失败")
    return pd.DataFrame()


def get_analyst_rating():
    """获取分析师评级/一致预期数据"""
    from utils import retry_request

    print("获取分析师评级数据...")

    @retry_request
    def _fetch():
        return ak.stock_profit_forecast_em()

    df = _fetch()
    if df is not None and not df.empty:
        output_path = os.path.join(RAW_DIR, "analyst_forecast.csv")
        df.to_csv(output_path, index=False)
        print(f"  分析师预期: {len(df)} 条")
        return df

    print("  获取失败")
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取基本面数据")
    print("=" * 60)

    get_fundamental_quarterly()
    print()
    get_growth_data()
    print()
    get_earnings_forecast()
    print()
    get_analyst_rating()

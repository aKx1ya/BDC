"""
04_money_flow.py
====================
优先级：P1（重要指标）
数据源：akshare（免费，来源东方财富）
获取内容：个股资金流向 - 主力/大单/中单/小单净流入

为什么重要（排在P1）：
- 大单净流入直接反映主力资金（机构）的操作方向
- 主力资金连续3日以上净流入的股票，未来5日延续上涨的概率约60%+
- 大单流入占比（大单净流入/总成交额）是衡量主力介入深度的关键指标
- 与散户资金流向（小单）对比，可以判断多空分歧程度

关键衍生特征：
- 大单净流入占比（当日）
- 大单净流入5日累计
- 主力资金连续流入天数
- 大单净流入排名（在沪深300中的相对排名，消除市场整体影响）

注意事项：
- 资金流向数据是根据成交价和盘口推算的，不是真实的买卖方向
- 大单定义：东方财富用单笔>50万为大单，>100万为超大单
- 作为短期趋势延续的辅助信号使用，不应作为唯一依据
"""

import akshare as ak
import pandas as pd
import os
import time
from config import RAW_DIR


def get_money_flow_rank_periods():
    """获取多周期资金流向排名（带重试）"""
    from utils import retry_request

    print("获取多周期资金流向排名...")

    @retry_request
    def _fetch(period):
        return ak.stock_individual_fund_flow_rank(indicator=period)

    for period in ["今日", "3日", "5日", "10日"]:
        df = _fetch(period)
        if df is not None and not df.empty:
            output_path = os.path.join(RAW_DIR, f"money_flow_rank_{period}.csv")
            df.to_csv(output_path, index=False)
            print(f"  {period}排名: {len(df)} 只")
        else:
            print(f"  {period}: 获取失败")
        time.sleep(1)


def get_sector_money_flow():
    """获取板块资金流向（带重试）"""
    from utils import retry_request

    print("获取板块资金流向...")

    @retry_request
    def _fetch(sector_type):
        return ak.stock_sector_fund_flow_rank(indicator="今日", sector_type=sector_type)

    for stype, fname in [("行业资金流", "sector_money_flow_industry.csv"),
                         ("概念资金流", "sector_money_flow_concept.csv")]:
        df = _fetch(stype)
        if df is not None and not df.empty:
            df.to_csv(os.path.join(RAW_DIR, fname), index=False)
            print(f"  {stype}: {len(df)} 个板块")
        else:
            print(f"  {stype}: 获取失败")


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取资金流向数据")
    print("=" * 60)

    get_money_flow_rank_periods()
    print()
    get_sector_money_flow()

"""
05_sector_momentum.py
====================
优先级：P2（辅助指标）
数据源：akshare（免费）
获取内容：行业/板块涨跌幅、板块轮动信号

为什么重要（排在P2）：
- A股板块轮动效应极强，热门板块内个股有明显联动性
- "买龙头"逻辑：资金涌入某板块时，龙头股涨幅最大
- 板块动量因子：近5日涨幅靠前的板块，未来5日大概率继续强势（短期动量）
- 但板块级别的因子粒度较粗，对个股的区分度不如资金流向

关键衍生特征：
- 所属行业近5日涨幅（板块动量）
- 所属行业在全部行业中的排名分位
- 板块内个股涨跌比（板块内部强弱）
- 个股涨幅 vs 所属板块涨幅（相对强度）

数据获取策略：
- 获取申万一级行业分类（最标准的行业分类）
- 获取每个行业指数的日K线
- 获取个股-行业映射关系
"""

import akshare as ak
import pandas as pd
import os
import time
from config import RAW_DIR, START_DATE


def get_sw_industry_classification():
    """获取行业分类映射"""
    from utils import retry_request

    print("获取行业分类...")

    @retry_request
    def _fetch():
        return ak.index_stock_cons_weight_csindex(symbol="000300")

    df = _fetch()
    if df is not None and not df.empty:
        output_path = os.path.join(RAW_DIR, "hs300_industry_mapping.csv")
        df.to_csv(output_path, index=False)
        print(f"  行业映射: {len(df)} 条")
        return df

    # 备选方案
    return get_industry_from_baostock()


def get_industry_from_baostock():
    """备选方案：从baostock获取行业分类"""
    try:
        import baostock as bs
        bs.login()
        rs = bs.query_stock_industry()
        data = []
        while rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        bs.logout()

        output_path = os.path.join(RAW_DIR, "stock_industry_baostock.csv")
        df.to_csv(output_path, index=False)
        print(f"  baostock行业分类: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"  baostock也失败了: {e}")
        return pd.DataFrame()


def get_sector_daily_performance():
    """获取行业板块每日K线（增量更新）"""
    from utils import retry_request, get_last_date, append_csv

    print("获取行业板块每日K线...")

    output_path = os.path.join(RAW_DIR, "sector_daily_kline.csv")
    last_date = get_last_date(output_path, date_col='日期')

    @retry_request
    def _fetch_list():
        return ak.stock_board_industry_name_em()

    df = _fetch_list()
    if df is None or df.empty:
        print("  获取板块列表失败")
        return pd.DataFrame()

    sector_names = df['板块名称'].tolist() if '板块名称' in df.columns else []
    print(f"  {len(sector_names)} 个板块")

    start = last_date.replace("-", "") if last_date else START_DATE.replace("-", "")

    @retry_request
    def _fetch_hist(name):
        return ak.stock_board_industry_hist_em(
            symbol=name, start_date=start, end_date="20261231",
            period="日k", adjust=""
        )

    all_data = []
    for i, name in enumerate(sector_names[:35]):
        hist = _fetch_hist(name)
        if hist is not None and not hist.empty:
            hist['sector_name'] = name
            all_data.append(hist)
            print(f"    [{i+1}] {name}: {len(hist)} 条")
        time.sleep(0.3)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        if last_date:
            append_csv(output_path, result)
            print(f"\n  板块K线: 增量追加 {len(result)} 条")
        else:
            result.to_csv(output_path, index=False)
            print(f"\n  板块K线: 全量 {len(result)} 条")
        return result

    return pd.DataFrame()


def get_sector_realtime_rank():
    """
    获取板块实时涨跌排名

    用途：作为最新一天的板块强弱快照
    """
    print("获取板块实时涨跌排名...")

    try:
        df = ak.stock_board_industry_name_em()
        output_path = os.path.join(RAW_DIR, "sector_realtime_rank.csv")
        df.to_csv(output_path, index=False)
        print(f"  {len(df)} 个板块")
        return df
    except Exception as e:
        print(f"  获取失败: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("开始获取板块动量数据")
    print("=" * 60)

    # 1. 获取行业分类映射
    get_sw_industry_classification()
    print()

    # 2. 获取板块历史K线（用于计算板块动量）
    get_sector_daily_performance()
    print()

    # 3. 实时板块排名
    get_sector_realtime_rank()

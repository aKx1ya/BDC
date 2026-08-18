# code/src/features.py
# 增强特征工程模块：行业特征 + 市场状态特征 + 横截面特征 + 统一入口
# 参考获奖队伍经验：多尺度量价 + 行业动能 + 市场状态 + 横截面排名

import pandas as pd
import numpy as np

# ============================================================
# 1. 行业特征
# ============================================================

# 沪深300成分股 → 申万一级行业映射（内嵌 300 只，按2026年初口径）
# 实际使用时可从外部文件加载覆盖
_STOCK_INDUSTRY_RAW = """
600000,银行         ;600015,银行         ;600016,银行         ;600030,非银金融
600036,银行         ;600048,房地产       ;600050,通信         ;600085,医药生物
600104,汽车         ;600111,有色金属     ;600115,交通运输     ;600150,国防军工
600196,医药生物     ;600276,医药生物     ;600309,基础化工     ;600406,电力设备
600436,医药生物     ;600438,电力设备     ;600519,食品饮料     ;600570,计算机
600585,建筑材料     ;600588,计算机       ;600690,家用电器     ;600703,电子
600745,电子         ;600809,食品饮料     ;600837,非银金融     ;600887,食品饮料
600893,国防军工     ;600900,公用事业     ;600905,公用事业     ;600919,银行
600941,通信         ;600999,非银金融     ;601006,交通运输     ;601009,银行
601012,电力设备     ;601021,交通运输     ;601066,非银金融     ;601088,煤炭
601100,国防军工     ;601111,交通运输     ;601127,汽车         ;601138,电子
601166,银行         ;601169,银行         ;601186,建筑装饰     ;601211,非银金融
601225,煤炭         ;601229,银行         ;601236,非银金融     ;601238,汽车
601288,银行         ;601318,非银金融     ;601319,非银金融     ;601328,银行
601336,非银金融     ;601360,计算机       ;601390,建筑装饰     ;601398,银行
601456,非银金融     ;601567,电力设备     ;601600,有色金属     ;601601,非银金融
601607,医药生物     ;601615,电力设备     ;601618,建筑装饰     ;601628,非银金融
601633,汽车         ;601658,银行         ;601668,建筑装饰     ;601669,建筑装饰
601677,有色金属     ;601688,非银金融     ;601689,汽车         ;601698,国防军工
601699,煤炭         ;601727,电力设备     ;601728,通信         ;601766,交通运输
601788,非银金融     ;601800,建筑装饰     ;601808,石油石化     ;601816,交通运输
601818,银行         ;601838,银行         ;601857,石油石化     ;601868,公用事业
601872,交通运输     ;601877,电力设备     ;601878,非银金融     ;601881,非银金融
601888,商贸零售     ;601899,有色金属     ;601901,非银金融     ;601919,交通运输
601939,银行         ;601985,公用事业     ;601988,银行         ;601989,国防军工
601995,非银金融     ;601998,银行         ;603019,计算机       ;603160,电子
603259,医药生物     ;603260,基础化工     ;603288,食品饮料     ;603290,电力设备
603296,电子         ;603369,食品饮料     ;603392,医药生物     ;603501,电子
603659,电力设备     ;603799,有色金属     ;603806,电力设备     ;603833,轻工制造
603899,轻工制造     ;603986,电子         ;603993,有色金属     ;605117,电力设备
605499,食品饮料     ;688009,交通运输     ;688012,电子         ;688036,电子
688041,电子         ;688047,计算机       ;688082,电子         ;688111,计算机
688114,医药生物     ;688120,电子         ;688122,有色金属     ;688126,电子
688153,电子         ;688169,家用电器     ;688187,交通运输     ;688223,电力设备
688235,医药生物     ;688256,电子         ;688271,医药生物     ;688303,电力设备
688347,电子         ;688349,电力设备     ;688390,电力设备     ;688396,电子
688472,电力设备     ;688484,电子         ;688506,医药生物     ;688525,电子
688536,电子         ;688538,电子         ;688548,电子         ;688561,计算机
688567,电力设备     ;688599,电力设备     ;688608,电子         ;688630,电子
688652,电子         ;688728,电子         ;688777,计算机       ;688778,电力设备
688779,基础化工     ;688819,电力设备     ;688981,电子         ;689009,轻工制造
000001,银行         ;000002,房地产       ;000063,通信         ;000069,房地产
000100,电子         ;000157,机械设备     ;000166,非银金融     ;000301,基础化工
000333,家用电器     ;000338,机械设备     ;000425,机械设备     ;000538,医药生物
000568,食品饮料     ;000596,食品饮料     ;000617,非银金融     ;000625,汽车
000630,有色金属     ;000651,家用电器     ;000661,医药生物     ;000708,钢铁
000725,电子         ;000733,国防军工     ;000768,国防军工     ;000776,非银金融
000786,建筑材料     ;000792,基础化工     ;000800,汽车         ;000807,有色金属
000858,食品饮料     ;000876,农林牧渔     ;000895,食品饮料     ;000938,计算机
000963,医药生物     ;000975,有色金属     ;000977,计算机       ;000983,煤炭
001289,公用事业     ;001965,交通运输     ;001979,房地产       ;002001,基础化工
002007,医药生物     ;002027,传媒         ;002049,电子         ;002050,家用电器
002074,电力设备     ;002129,电力设备     ;002142,银行         ;002179,国防军工
002180,电子         ;002202,电力设备     ;002230,计算机       ;002236,计算机
002241,电子         ;002252,医药生物     ;002271,建筑材料     ;002304,食品饮料
002311,农林牧渔     ;002352,交通运输     ;002371,电子         ;002384,电子
002410,计算机       ;002414,国防军工     ;002415,电子         ;002459,电力设备
002460,有色金属     ;002463,电子         ;002466,有色金属     ;002475,电子
002493,石油石化     ;002555,传媒         ;002594,汽车         ;002600,电子
002601,基础化工     ;002603,医药生物     ;002625,国防军工     ;002648,基础化工
002714,农林牧渔     ;002736,非银金融     ;002812,电力设备     ;002821,医药生物
002841,电子         ;002916,电子         ;002920,汽车         ;002938,电子
300003,医药生物     ;300014,电力设备     ;300015,医药生物     ;300033,计算机
300059,非银金融     ;300122,医药生物     ;300124,电力设备     ;300142,医药生物
300223,电子         ;300274,电力设备     ;300308,通信         ;300316,电力设备
300339,计算机       ;300347,医药生物     ;300394,通信         ;300408,电子
300413,电子         ;300418,计算机       ;300433,电子         ;300442,机械设备
300450,电力设备     ;300454,计算机       ;300496,计算机       ;300498,农林牧渔
300502,通信         ;300529,医药生物     ;300552,计算机       ;300595,医药生物
300628,通信         ;300661,电子         ;300750,电力设备     ;300751,电力设备
300759,医药生物     ;300760,医药生物     ;300782,电子         ;300832,医药生物
300857,通信         ;300866,电子         ;300896,医药生物     ;300919,电力设备
300957,医药生物     ;300979,轻工制造     ;301004,轻工制造     ;301236,计算机
301269,电子         ;301358,电力设备     ;301551,传媒
"""


def _load_industry_map():
    """解析内嵌的行业映射表"""
    mapping = {}
    for item in _STOCK_INDUSTRY_RAW.replace('\n', ';').split(';'):
        item = item.strip()
        if not item:
            continue
        parts = item.split(',')
        if len(parts) == 2:
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping


SW_INDUSTRY_MAP = _load_industry_map()


def add_industry_features(df, industry_map=None):
    """
    添加行业分类特征。

    产出的特征列：
    - sector_daily_return: 行业日内等权平均收益
    - sector_return_3d: 行业近3日累计收益
    - sector_return_5d: 行业近5日累计收益
    - excess_vs_sector: 个股相对行业超额收益
    - sector_amount_change_5d: 行业成交额5日变化率
    - sector_momentum_rank: 行业动能排名（同一天内的行业排名0~1）
    - is_strong_sector: 是否为强势行业（前30%）
    - ind_*: 行业one-hot编码
    """
    if industry_map is None:
        industry_map = SW_INDUSTRY_MAP

    df = df.copy()
    df['industry'] = df['股票代码'].map(industry_map).fillna('未知')

    # 行业日内等权平均收益
    daily_sector_return = df.groupby(['日期', 'industry'])['涨跌幅'].transform('mean')
    df['sector_daily_return'] = daily_sector_return

    # 行业近3日、5日累计收益（按行业+股票分组滚动）
    df['sector_return_3d'] = df.groupby(['industry', '股票代码'])['sector_daily_return'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['sector_return_5d'] = df.groupby(['industry', '股票代码'])['sector_daily_return'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # 个股相对行业超额收益
    df['excess_vs_sector'] = df['涨跌幅'] - df['sector_daily_return']

    # 行业成交额变化
    daily_sector_amount = df.groupby(['日期', 'industry'])['成交额'].transform('sum')
    df['sector_amount'] = daily_sector_amount
    df['sector_amount_change_5d'] = df.groupby(['industry', '股票代码'])['sector_amount'].transform(
        lambda x: x.pct_change(5)
    )

    # 行业动能排名（同一天内各行业5日收益率排名）
    df['sector_momentum_rank'] = df.groupby('日期')['sector_return_5d'].rank(pct=True)

    # 是否为强势行业（前30%）
    df['is_strong_sector'] = (df['sector_momentum_rank'] > 0.7).astype(int)

    # 行业 one-hot 编码（保留原始 industry 列供精排阶段使用）
    industry_dummies = pd.get_dummies(df['industry'], prefix='ind')
    df = pd.concat([df, industry_dummies], axis=1)

    return df


# ============================================================
# 2. 市场状态特征
# ============================================================

def add_market_features(df):
    """
    添加沪深300市场状态特征（从同一天所有股票的等权平均近似）。

    产出的特征列：
    - market_return: 市场日内等权平均收益
    - market_return_5d: 市场近5日累计收益
    - market_return_20d: 市场近20日累计收益
    - market_volatility_20d: 市场近20日波动率
    - market_above_ma20: 市场是否位于20日均线上方
    - advance_decline_ratio: 涨跌比（上涨家数/下跌家数）
    - market_amount_change_5d: 全市场成交额5日变化率
    """
    df = df.copy()

    # 市场日内等权平均收益
    market_return = df.groupby('日期')['涨跌幅'].mean().rename('market_return')
    df = df.join(market_return, on='日期')

    # 市场近5日、20日累计收益
    df['market_return_5d'] = df.groupby('股票代码')['market_return'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['market_return_20d'] = df.groupby('股票代码')['market_return'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )

    # 市场波动率
    df['market_volatility_20d'] = df.groupby('股票代码')['market_return'].transform(
        lambda x: x.rolling(20, min_periods=1).std()
    )

    # 市场是否位于20日均线上方
    df['market_above_ma20'] = (df['market_return_20d'] > 0).astype(int)

    # 涨跌比
    daily_up = df.groupby('日期')['涨跌幅'].apply(lambda x: (x > 0).sum())
    daily_down = df.groupby('日期')['涨跌幅'].apply(lambda x: (x < 0).sum())
    adr = (daily_up / (daily_down + 1)).rename('advance_decline_ratio')
    df = df.join(adr, on='日期')

    # 全市场成交额变化
    daily_total = df.groupby('日期')['成交额'].sum().rename('market_total_amount')
    df = df.join(daily_total, on='日期')
    df['market_amount_change_5d'] = df.groupby('股票代码')['market_total_amount'].transform(
        lambda x: x.pct_change(5)
    )

    # 清理临时列
    df.drop(columns=['market_total_amount'], inplace=True, errors='ignore')

    return df


# ============================================================
# 3. 统一特征工程入口
# ============================================================

def engineer_all_features(
    df,
    enable_industry=True,
    enable_market=True,
    enable_cross_sectional=True,
    industry_map=None,
):
    """
    统一增强特征工程入口。

    按顺序执行：
    1. 基础量价特征（158+39 = 197维）——调用 utils.py 中已有函数
    2. 横截面特征（排名/Z-score）——调用 utils.py 中已有函数
    3. 额外时序因子（VWAP偏离等）——调用 utils.py 中已有函数
    4. 行业特征——本模块
    5. 市场状态特征——本模块
    6. 统一清理

    返回:
    - df: 增强后的完整特征表
    - 所有原始列 + 新增特征列
    """
    from utils import (
        engineer_features_158plus39,
        add_extra_factor_features,
        add_cross_sectional_features,
    )

    df = df.copy()

    # Step 1: 基础量价特征
    df = engineer_features_158plus39(df)

    # Step 2: 横截面特征
    if enable_cross_sectional:
        df = add_cross_sectional_features(df)

    # Step 3: 额外时序因子
    df = add_extra_factor_features(df)

    # Step 4: 行业特征
    if enable_industry:
        df = add_industry_features(df, industry_map=industry_map)

    # Step 5: 市场状态特征
    if enable_market:
        df = add_market_features(df)

    # Step 6: 统一清理
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

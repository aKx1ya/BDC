# code/src/label.py
# 标签构造模块：支持绝对收益、超额收益、排序标签、方向标签等多种策略
# 参考获奖队伍经验：超额收益标签比绝对收益稳定得多

import pandas as pd
import numpy as np


def build_absolute_return_label(df, buy_offset=1, sell_offset=6):
    """
    绝对收益标签（baseline 原有方式）。

    label = (open_{T+sell_offset} - open_{T+buy_offset}) / open_{T+buy_offset}

    这是原始 baseline 的默认标签。
    """
    df = df.copy()
    df['open_t_buy'] = df.groupby('股票代码')['开盘'].shift(-buy_offset)
    df['open_t_sell'] = df.groupby('股票代码')['开盘'].shift(-sell_offset)
    df['label'] = (df['open_t_sell'] - df['open_t_buy']) / (df['open_t_buy'] + 1e-12)
    df.drop(columns=['open_t_buy', 'open_t_sell'], inplace=True)
    return df


def build_excess_return_label(df, buy_offset=1, sell_offset=6):
    """
    超额收益标签（推荐，获奖队伍 O_O 的核心经验）。

    label = 个股5日收益 - 同日所有股票等权平均收益

    这消除了大盘情绪干扰，让模型聚焦"相对跑赢能力"，
    稳定性比绝对收益标签有质的提升。
    """
    df = df.copy()
    df['open_t_buy'] = df.groupby('股票代码')['开盘'].shift(-buy_offset)
    df['open_t_sell'] = df.groupby('股票代码')['开盘'].shift(-sell_offset)
    df['stock_return'] = (df['open_t_sell'] - df['open_t_buy']) / (df['open_t_buy'] + 1e-12)

    # 同一天所有股票的等权平均收益作为基准
    daily_mean = df.groupby('日期')['stock_return'].transform('mean')
    df['label'] = df['stock_return'] - daily_mean

    df.drop(columns=['open_t_buy', 'open_t_sell', 'stock_return'], inplace=True)
    return df


def build_rank_label(df, buy_offset=1, sell_offset=6):
    """
    排序标签：同一天内收益率排名百分位（0~1，越高越好）。

    适合 XGBRanker / LGBMRanker 直接以 rank_pct 作为 label。
    获奖队伍 7355608 的经验：让模型学习"相对排序关系"而非绝对收益。
    """
    df = df.copy()
    df['open_t_buy'] = df.groupby('股票代码')['开盘'].shift(-buy_offset)
    df['open_t_sell'] = df.groupby('股票代码')['开盘'].shift(-sell_offset)
    df['stock_return'] = (df['open_t_sell'] - df['open_t_buy']) / (df['open_t_buy'] + 1e-12)

    # 同一天内按收益率排名，转为 0~1 之间的分数
    df['label'] = df.groupby('日期')['stock_return'].rank(pct=True)

    df.drop(columns=['open_t_buy', 'open_t_sell', 'stock_return'], inplace=True)
    return df


def build_direction_label(df, buy_offset=1, sell_offset=6):
    """
    方向标签（辅助任务用）：未来5日是否上涨（0/1 二分类）。

    获奖队伍 7355608 指出辅助任务能提供更多监督信号，
    增强表征稳定性。
    """
    df = df.copy()
    df['open_t_buy'] = df.groupby('股票代码')['开盘'].shift(-buy_offset)
    df['open_t_sell'] = df.groupby('股票代码')['开盘'].shift(-sell_offset)
    df['stock_return'] = (df['open_t_sell'] - df['open_t_buy']) / (df['open_t_buy'] + 1e-12)
    df['label'] = (df['stock_return'] > 0).astype(int)
    df.drop(columns=['open_t_buy', 'open_t_sell', 'stock_return'], inplace=True)
    return df


def build_volatility_label(df, buy_offset=1, sell_offset=6):
    """
    波动率标签（辅助任务用）：未来5日内的日收益标准差（年化）。

    辅助波动率刻画任务使用。
    """
    df = df.copy()
    df['return_1d'] = df.groupby('股票代码')['收盘'].pct_change(1)

    def _future_vol(group):
        """计算未来 sell_offset 个交易日的收益波动率"""
        group = group.sort_values('日期')
        rets = group['return_1d'].shift(-buy_offset)
        future_rets = []
        for i in range(len(group)):
            window = rets.iloc[i:i + sell_offset - buy_offset]
            future_rets.append(window.std() if len(window) >= 2 else np.nan)
        return pd.Series(future_rets, index=group.index)

    df['label'] = df.groupby('股票代码', group_keys=False).apply(
        _future_vol, include_groups=False
    ).reset_index(level=0, drop=True)
    df.drop(columns=['return_1d'], inplace=True)
    return df


# 标签构造器注册表
LABEL_BUILDERS = {
    'absolute_return': build_absolute_return_label,
    'excess_return': build_excess_return_label,
    'rank': build_rank_label,
    'direction': build_direction_label,
    'volatility': build_volatility_label,
}


def get_label_builder(label_type):
    """根据配置返回对应的标签构造函数"""
    if label_type not in LABEL_BUILDERS:
        raise ValueError(f"不支持的标签类型 '{label_type}'，可选: {list(LABEL_BUILDERS.keys())}")
    return LABEL_BUILDERS[label_type]


def build_labels(df, label_type='excess_return', buy_offset=1, sell_offset=6,
                 drop_small_open=True):
    """
    统一的标签构建入口。

    参数:
    - df: 特征工程后的 DataFrame
    - label_type: 'absolute_return' | 'excess_return' | 'rank' | 'direction' | 'volatility'
    - buy_offset: 买入偏移（默认1，即T+1开盘买入）
    - sell_offset: 卖出偏移（默认6，即T+6开盘卖出）
    - drop_small_open: 是否过滤开盘价过低的样本

    返回:
    - 带 'label' 列的 DataFrame
    """
    builder = get_label_builder(label_type)
    df = builder(df, buy_offset=buy_offset, sell_offset=sell_offset)

    # 过滤无效开盘价
    if drop_small_open:
        df = df[df['开盘'] > 1e-4]

    # 删除 label 为 NaN 的行
    df = df.dropna(subset=['label'])

    return df


def build_aux_labels(df, aux_tasks=None, buy_offset=1, sell_offset=6):
    """
    为辅助任务构造额外标签。

    参数:
    - df: 特征工程后的 DataFrame
    - aux_tasks: list of str, 如 ['direction', 'volatility']

    返回:
    - df: 带 'label' 列（主标签）+ 'aux_direction' / 'aux_volatility' 列
    """
    if aux_tasks is None:
        return df

    df = df.copy()
    for task in aux_tasks:
        if task == 'direction':
            tmp = build_direction_label(df[[c for c in df.columns if c != 'label']].copy(),
                                        buy_offset=buy_offset, sell_offset=sell_offset)
            df['aux_direction'] = tmp['label']
        elif task == 'volatility':
            tmp = build_volatility_label(df[[c for c in df.columns if c != 'label']].copy(),
                                         buy_offset=buy_offset, sell_offset=sell_offset)
            df['aux_volatility'] = tmp['label']
    return df

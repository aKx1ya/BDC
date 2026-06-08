"""
07_feature_engine.py
====================
特征工程核心模块

从原始数据计算所有衍生特征，输出统一的特征矩阵。
每一行代表一个 (stock, date) 组合，列为各特征值。

特征分类：
- momentum_*: 动量/反转因子
- vol_*: 波动率因子
- turnover_*: 换手率因子
- flow_*: 资金流向因子
- north_*: 北向资金因子
- margin_*: 融资融券因子
- sector_*: 板块因子
- fundamental_*: 基本面因子
- target: 未来5日收益率（训练用标签）
"""

import pandas as pd
import numpy as np
import os
from config import (
    RAW_DIR, FEATURE_DIR,
    MOMENTUM_WINDOWS, VOLATILITY_WINDOWS,
    TURNOVER_MA_WINDOW, AMIHUD_WINDOW
)


def load_price_data():
    """加载日K线数据"""
    path = os.path.join(RAW_DIR, "daily_price_volume.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"请先运行 01_price_volume.py 获取数据: {path}")

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    return df


# ================================================================
# P0 特征：量价动量 + 波动率 + 换手率
# ================================================================

def calc_momentum_features(df):
    """
    计算动量因子

    原理：
    - 短期动量(5日)：A股短期趋势延续性强，强势股短期继续强势
    - 中期反转(20日)：过去一个月涨太多的，未来一周回调概率高
    - 不同窗口的动量组合比单一窗口更稳定
    """
    print("  计算动量因子...")
    for window in MOMENTUM_WINDOWS:
        # 收益率动量：过去N日累计收益
        df[f'momentum_{window}d'] = df.groupby('code')['close'].transform(
            lambda x: x.pct_change(window)
        )

    # 反转因子：20日收益率取负
    df['reversal_20d'] = -df['momentum_20d']

    # 价格相对位置：当前价格在过去20日的百分位
    df['price_position_20d'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20).apply(
            lambda w: (w.iloc[-1] - w.min()) / (w.max() - w.min() + 1e-10)
            if len(w) == 20 else np.nan
        )
    )

    return df


def calc_volatility_features(df):
    """
    计算波动率因子

    原理：
    - 已实现波动率：用日收益率的标准差衡量
    - 波动率聚集效应：高波动后可能继续高波动
    - 低波动→高波动的转换往往伴随趋势启动
    - 上行波动vs下行波动（偏度）反映方向性预期
    """
    print("  计算波动率因子...")

    # 日收益率
    df['daily_return'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change()
    )

    for window in VOLATILITY_WINDOWS:
        # 已实现波动率
        df[f'volatility_{window}d'] = df.groupby('code')['daily_return'].transform(
            lambda x: x.rolling(window).std()
        )

    # 波动率变化率（5日波动率 / 20日波动率）
    # >1 表示近期波动加剧，<1 表示近期波动收敛
    df['vol_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)

    # 上行波动率 vs 下行波动率（波动率偏度）
    df['upside_vol_10d'] = df.groupby('code')['daily_return'].transform(
        lambda x: x.rolling(10).apply(
            lambda w: w[w > 0].std() if (w > 0).sum() > 1 else 0
        )
    )
    df['downside_vol_10d'] = df.groupby('code')['daily_return'].transform(
        lambda x: x.rolling(10).apply(
            lambda w: w[w < 0].std() if (w < 0).sum() > 1 else 0
        )
    )
    df['vol_skew_10d'] = df['upside_vol_10d'] / (df['downside_vol_10d'] + 1e-10)

    return df


def calc_turnover_features(df):
    """
    计算换手率因子

    原理：
    - 换手率异动 = 当日换手率 / 过去20日平均换手率
    - 异常放量（>2倍）往往是趋势加速或反转信号
    - 缩量（<0.5倍）表示市场对该股关注度下降
    - 换手率的变异系数反映资金进出的不稳定性
    """
    print("  计算换手率因子...")

    # 换手率移动平均
    df['turnover_ma20'] = df.groupby('code')['turn'].transform(
        lambda x: x.rolling(TURNOVER_MA_WINDOW).mean()
    )

    # 换手率异动比（核心！）
    df['turnover_anomaly'] = df['turn'] / (df['turnover_ma20'] + 1e-10)

    # 换手率5日均值
    df['turnover_ma5'] = df.groupby('code')['turn'].transform(
        lambda x: x.rolling(5).mean()
    )

    # 换手率变异系数（过去10日）：衡量交易活跃度的稳定性
    df['turnover_cv_10d'] = df.groupby('code')['turn'].transform(
        lambda x: x.rolling(10).std() / (x.rolling(10).mean() + 1e-10)
    )

    return df


def calc_amihud_illiquidity(df):
    """
    计算Amihud非流动性指标

    原理：
    - Amihud = |日收益率| / 日成交额
    - 衡量单位资金对价格的冲击程度
    - 流动性差的股票：风险溢价高，预期收益高（补偿）
    - 但流动性突然恶化也可能意味着即将出现大幅波动
    """
    print("  计算Amihud非流动性...")

    # 单日Amihud
    df['amihud_daily'] = df['daily_return'].abs() / (df['amount'] + 1e-10)

    # 20日平均Amihud
    df['amihud_20d'] = df.groupby('code')['amihud_daily'].transform(
        lambda x: x.rolling(AMIHUD_WINDOW).mean()
    )

    # 对数化（分布更正态）
    df['log_amihud_20d'] = np.log1p(df['amihud_20d'] * 1e8)

    return df


def calc_market_cap_factor(df):
    """
    市值因子

    原理：
    - 小市值效应在A股显著存在
    - 对数市值作为控制变量（确保其他因子不被市值效应污染）
    - 也可以用于后续的行业/市值中性化处理
    """
    print("  计算市值因子...")

    # 用 close * volume 近似流通市值（粗略，但足够做因子）
    # 更准确的做法是用baostock的总市值数据
    # 这里暂用PE * 净利润推算，或直接从原始数据取

    # 简化方案：用成交额/换手率推算流通市值
    # 流通市值 ≈ 成交额 / 换手率 * 100（换手率是百分比）
    df['market_cap_approx'] = df['amount'] / (df['turn'] / 100 + 1e-10)
    df['log_market_cap'] = np.log(df['market_cap_approx'] + 1)

    return df


# ================================================================
# 目标变量：未来5日收益率
# ================================================================

def calc_target(df):
    """
    计算预测目标：T+1开盘买入 → T+5开盘卖出的收益率

    注意：
    - 使用open价格而非close，与比赛评估标准一致
    - shift(-1)获取下一个交易日的open
    - shift(-5)获取第5个交易日后的open（即T+5的open）
    """
    print("  计算目标变量(未来5日收益率)...")

    # T+1开盘价
    df['open_t1'] = df.groupby('code')['open'].shift(-1)
    # T+5开盘价（即T+1开盘后持有4个交易日）
    df['open_t5'] = df.groupby('code')['open'].shift(-5)

    # 目标收益率
    df['target_return_5d'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-10)

    # 清理临时列
    df.drop(columns=['open_t1', 'open_t5'], inplace=True, errors='ignore')

    return df


# ================================================================
# 主流程
# ================================================================

def build_features():
    """构建完整特征矩阵"""
    print("=" * 60)
    print("开始特征工程")
    print("=" * 60)

    # 加载原始数据
    df = load_price_data()
    print(f"原始数据: {df['code'].nunique()} 只股票, {len(df)} 条记录")

    # 计算各类特征
    df = calc_momentum_features(df)
    df = calc_volatility_features(df)
    df = calc_turnover_features(df)
    df = calc_amihud_illiquidity(df)
    df = calc_market_cap_factor(df)
    df = calc_target(df)

    # 选取特征列
    feature_cols = [col for col in df.columns if any(
        col.startswith(prefix) for prefix in [
            'momentum_', 'reversal_', 'price_position_',
            'volatility_', 'vol_ratio_', 'vol_skew_',
            'upside_vol_', 'downside_vol_',
            'turnover_', 'turn',
            'amihud_', 'log_amihud_',
            'log_market_cap',
            'peTTM', 'pbMRQ',
        ]
    )]

    # 保存特征矩阵
    output_cols = ['date', 'code'] + feature_cols + ['target_return_5d']
    result = df[output_cols].copy()

    # 去掉特征不完整的行（前N天没有足够数据计算滚动指标）
    result = result.dropna(subset=feature_cols, how='all')

    output_path = os.path.join(FEATURE_DIR, "feature_matrix.csv")
    result.to_csv(output_path, index=False)

    print(f"\n特征矩阵保存至: {output_path}")
    print(f"形状: {result.shape}")
    print(f"特征数: {len(feature_cols)}")
    print(f"\n特征列表:")
    for col in feature_cols:
        print(f"  - {col}")

    return result


if __name__ == "__main__":
    build_features()

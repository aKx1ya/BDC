# code/src/postprocess.py
# 精排模块：Top30候选池 → Top5最终提交
# 包括：硬门控、行业约束、相关性约束、均值方差权重优化

import pandas as pd
import numpy as np
from scipy.optimize import minimize


# ============================================================
# 1. 硬门控过滤
# ============================================================

def apply_hard_gates(candidates_df, price_df=None, config=None):
    """
    硬门控过滤。过滤条件：
    - 涨跌停（涨跌幅 ±9.5%+）
    - 近20日最大回撤 > 阈值 → 剔除
    - 近5日曾有单日大跌 > 阈值 → 剔除
    - 异常价格
    """
    if config is None:
        config = {}
    df = candidates_df.copy()
    initial = len(df)

    # 涨跌停过滤
    if '涨跌幅' in df.columns:
        df = df[df['涨跌幅'].abs() < 9.5]

    # 开盘价异常
    if '开盘' in df.columns:
        df = df[df['开盘'] > 1e-4]

    # === 最大回撤 & 暴跌过滤（需要 price_df） ===
    if price_df is not None and '股票代码' in price_df.columns and '日期' in price_df.columns:
        max_dd_threshold = float(config.get('max_drawdown_threshold', 0.15))
        crash_threshold = float(config.get('crash_threshold', 0.07))

        id_col = 'stock_id' if 'stock_id' in df.columns else '股票代码'
        stock_ids = df[id_col].astype(str).str.zfill(6).tolist()

        valid_stocks = set()
        for sid in stock_ids:
            sdata = price_df[price_df['股票代码'].astype(str).str.zfill(6) == sid].sort_values('日期')
            if len(sdata) < 20:
                valid_stocks.add(sid)
                continue

            close_col = '收盘' if '收盘' in sdata.columns else None
            if close_col is None:
                valid_stocks.add(sid)
                continue

            closes = sdata[close_col].astype(float).tail(20).values
            peak = np.maximum.accumulate(closes)
            dd = (closes - peak) / (peak + 1e-12)
            if float(np.min(dd)) < -max_dd_threshold:
                continue  # 回撤过大

            recent = np.diff(closes[-6:]) / (closes[-6:-1] + 1e-12)
            if np.any(recent < -crash_threshold):
                continue  # 近期暴跌

            valid_stocks.add(sid)

        df = df[df[id_col].astype(str).str.zfill(6).isin(valid_stocks)]

    filtered = initial - len(df)
    if filtered > 0:
        print(f"[HardGate] 过滤 {filtered}/{initial} 只（回撤/暴跌/涨跌停）")
    return df


# ============================================================
# 2. 行业集中度约束
# ============================================================

def apply_sector_constraint(ranked_df, max_per_sector=2, industry_col='industry'):
    """同一行业最多 max_per_sector 只股票"""
    if industry_col not in ranked_df.columns:
        return ranked_df

    selected = []
    sector_counts = {}
    for _, row in ranked_df.iterrows():
        sector = row.get(industry_col, '未知')
        if sector_counts.get(sector, 0) < max_per_sector:
            selected.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return pd.DataFrame(selected)


# ============================================================
# 3. 相关性约束
# ============================================================

def apply_correlation_constraint(ranked_df, returns_df, max_corr=0.3, top_k=5):
    """
    与已入选股票的历史收益相关系数 > max_corr 则跳过。
    returns_df: 行=日期，列=股票代码
    """
    selected = []
    selected_rets = []

    for _, row in ranked_df.iterrows():
        sid = str(row.get('stock_id', row.get('股票代码', '')))
        # Clean stock ID
        sid = _clean_stock_id(sid)
        if sid not in returns_df.columns:
            selected.append(row)
            continue

        stock_ret = returns_df[sid].dropna().values
        too_correlated = False
        for sel_ret in selected_rets:
            min_len = min(len(stock_ret), len(sel_ret))
            if min_len < 10:
                continue
            corr = np.corrcoef(stock_ret[-min_len:], sel_ret[-min_len:])[0, 1]
            if abs(corr) > max_corr:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(row)
            selected_rets.append(stock_ret)
        if len(selected) >= top_k:
            break

    return pd.DataFrame(selected)


# ============================================================
# 4. 均值-方差权重优化
# ============================================================

def optimize_weights_mean_variance(
    expected_returns,
    cov_matrix,
    max_weight=0.45,
    risk_aversion=1.0,
):
    """
    均值-方差优化: max(w·μ - λ·w·Σ·w)
    约束: Σw ≤ 1, 0 ≤ w_i ≤ max_weight
    """
    n = len(expected_returns)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([min(1.0, max_weight)])

    def objective(w):
        return -(np.dot(w, expected_returns) - risk_aversion * np.dot(w, np.dot(cov_matrix, w)))

    constraints = [{'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.ones(n) / n

    try:
        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12},
        )
        weights = result.x
    except Exception:
        weights = x0

    weights = np.clip(weights, 0, max_weight)
    total = weights.sum()
    if total > 0:
        weights = weights / total * min(1.0, total)
    return weights


# ============================================================
# 5. 精排主流程
# ============================================================

def fine_ranking(
    candidate_df,
    price_df=None,
    returns_df=None,
    sector_map=None,
    config=None,
):
    """
    Top30 → Top5 精排主流程。

    步骤:
    1. Hard Gates
    2. 多维打分（ML分数 + 行业动能）
    3. 行业分散度约束
    4. 相关性约束
    5. 均值-方差权重优化

    返回: DataFrame [stock_id, weight]
    """
    if config is None:
        config = {}

    df = candidate_df.copy()

    # Step 1: Hard Gates
    df = apply_hard_gates(df, price_df=price_df, config=config)

    # Step 2: 多维打分
    if 'ensemble_rank_pct' in df.columns:
        df['refine_score'] = df['ensemble_rank_pct']
    elif 'ensemble_score' in df.columns:
        df['refine_score'] = df['ensemble_score'].rank(pct=True)
    else:
        score_cols = [c for c in df.columns if c.endswith('_score')]
        if score_cols:
            df['refine_score'] = df[score_cols].mean(axis=1).rank(pct=True)
        else:
            df['refine_score'] = 1.0

    df = df.sort_values('refine_score', ascending=False)

    # Step 3: 行业约束
    max_per_sector = config.get('max_per_sector', 2)
    if 'industry' in df.columns:
        df = apply_sector_constraint(df, max_per_sector=max_per_sector)

    # Step 4: 相关性约束
    max_corr = config.get('max_correlation', 0.3)
    if returns_df is not None:
        df = apply_correlation_constraint(df, returns_df, max_corr=max_corr, top_k=5)

    # Step 5: 选择 Top5
    top5 = df.head(5)

    # Step 6: 权重优化
    if len(top5) >= 2 and returns_df is not None:
        sid_col = 'stock_id' if 'stock_id' in top5.columns else '股票代码'
        top_sids = top5[sid_col].astype(str).tolist()
        avail_sids = [s for s in top_sids if s in returns_df.columns]

        if len(avail_sids) >= 2:
            ret_data = returns_df[avail_sids].dropna()
            if len(ret_data) > 20:
                # 模型分数映射为预期收益
                if 'refine_score' in top5.columns:
                    er = top5.set_index(sid_col).loc[avail_sids, 'refine_score'].values
                else:
                    er = np.ones(len(avail_sids))
                er = (er - er.min()) / (er.max() - er.min() + 1e-8)
                cov = ret_data.cov().values
                risk_aversion = config.get('risk_aversion', 1.0)
                max_w = config.get('max_single_weight', 0.45)
                weights = optimize_weights_mean_variance(er, cov, max_weight=max_w, risk_aversion=risk_aversion)
            else:
                weights = np.ones(len(avail_sids)) / len(avail_sids)
        else:
            weights = np.ones(len(top5)) / len(top5)
    else:
        weights = np.ones(len(top5)) / len(top5)

    # 构建输出
    result = pd.DataFrame({
        'stock_id': [_clean_stock_id(top5.iloc[i].get('stock_id', top5.iloc[i].get('股票代码', '')))
                     for i in range(len(top5))],
        'weight': weights[:len(top5)],
    })
    # 确保 weight 为 float
    result['weight'] = result['weight'].astype(float)
    total_w = result['weight'].sum()
    if total_w > 0:
        result['weight'] = result['weight'] / total_w  # 确保和为1（或更小）

    return result


def _clean_stock_id(sid):
    """清理股票代码：去除 .0 后缀，补零到6位。"""
    s = str(sid).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.zfill(6)

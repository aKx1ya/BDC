# code/src/validation.py
# Walk-forward 滚动验证框架
# 参考获奖队伍经验：跨周期稳定性评估是核心，不能只看单次切分

import pandas as pd
import numpy as np


class WalkForwardValidator:
    """
    Walk-forward 滚动验证框架。

    每一步：
    1. 用过去 train_window 个交易日训练
    2. 留 purge_gap 个交易日净化带
    3. 预测接下来 test_window 个交易日
    4. 记录评分
    5. 整体窗口向前滚动 step_size

    获奖队伍「柚子」强调：不追求单次高分，重点打磨跨周期稳定性。
    """

    def __init__(
        self,
        train_window=252,       # 训练窗口（约1年交易日）
        test_window=5,          # 测试窗口（比赛口径：5个交易日）
        purge_gap=5,            # 净化带
        step_size=5,            # 滚动步长
        min_train_size=120,     # 最小训练样本数
        n_splits=None,          # 最多几轮（None=自动）
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.purge_gap = purge_gap
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.n_splits = n_splits

    def split(self, dates):
        """
        生成 Walk-forward 的 (train_dates, test_dates) 对。

        返回: list of dict
        """
        dates = sorted(pd.to_datetime(dates).unique())
        splits = []
        i = 0

        while True:
            test_end_idx = len(dates) - i * self.step_size - 1
            test_start_idx = test_end_idx - self.test_window + 1

            if test_start_idx < self.min_train_size + self.train_window:
                break

            train_end_idx = test_start_idx - self.purge_gap - 1
            train_start_idx = max(0, train_end_idx - self.train_window + 1)

            if train_end_idx - train_start_idx < self.min_train_size:
                break

            splits.append({
                'fold': len(splits),
                'train_dates': dates[train_start_idx:train_end_idx + 1],
                'test_dates': dates[test_start_idx:test_end_idx + 1],
            })

            i += 1
            if self.n_splits and len(splits) >= self.n_splits:
                break

        return splits

    def run(self, train_fn, predict_fn, score_fn, data_df, verbose=True):
        """
        执行完整的 Walk-forward 验证。

        参数:
        - train_fn(train_df, fold_info) -> model
        - predict_fn(model, test_df, latest_train_date) -> result_df (stock_id, weight)
        - score_fn(result_df, test_df) -> float (Final Score)
        - data_df: 完整 DataFrame

        返回: dict summary
        """
        dates = sorted(data_df['日期'].unique())
        splits = self.split(dates)
        fold_results = []

        for fold_info in splits:
            train_mask = data_df['日期'].isin(fold_info['train_dates'])
            test_mask = data_df['日期'].isin(fold_info['test_dates'])
            train_df = data_df[train_mask].copy()
            test_df = data_df[test_mask].copy()

            model = train_fn(train_df, fold_info)
            result_df = predict_fn(model, test_df, fold_info['train_dates'][-1])
            score = score_fn(result_df, test_df)

            fold_results.append({
                'fold': fold_info['fold'],
                'train_start': fold_info['train_dates'][0].strftime('%Y-%m-%d'),
                'train_end': fold_info['train_dates'][-1].strftime('%Y-%m-%d'),
                'test_start': fold_info['test_dates'][0].strftime('%Y-%m-%d'),
                'test_end': fold_info['test_dates'][-1].strftime('%Y-%m-%d'),
                'score': score,
            })

            if verbose:
                print(f"  Fold {fold_info['fold']:2d}: "
                      f"{fold_info['train_dates'][0].date()} ~ {fold_info['train_dates'][-1].date()} "
                      f"-> test {fold_info['test_dates'][0].date()} ~ {fold_info['test_dates'][-1].date()} "
                      f"score={score:.6f}")

        scores = [r['score'] for r in fold_results]
        summary = {
            'n_folds': len(fold_results),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'sharpe_like': float(np.mean(scores) / (np.std(scores) + 1e-12)),
            'win_rate': float(sum(1 for s in scores if s > 0) / max(len(scores), 1)),
            'folds': fold_results,
        }

        if verbose:
            print(f"\n=== Walk-forward 汇总 ({summary['n_folds']} folds) ===")
            print(f"  均值: {summary['mean_score']:.6f} ± {summary['std_score']:.6f}")
            print(f"  范围: [{summary['min_score']:.6f}, {summary['max_score']:.6f}]")
            print(f"  稳定性(Sharpe-like): {summary['sharpe_like']:.2f}")
            print(f"  胜率: {summary['win_rate']:.1%}")

        return summary


def compute_final_score(result_df, test_df, buy_offset=1, sell_offset=6):
    """
    计算组合的 Final Score（与比赛评分口径一致）。

    对每只入选股票，用 test 数据的开盘价计算：
    return = (open_{T+sell_offset} - open_{T+buy_offset}) / open_{T+buy_offset}
    final_score = sum(return_i * weight_i)
    """
    if result_df is None or len(result_df) == 0:
        return 0.0

    score = 0.0
    stock_col = 'stock_id'
    weight_col = 'weight'
    if stock_col not in result_df.columns:
        stock_col = '股票代码'
    if weight_col not in result_df.columns:
        weight_col = '权重'

    for _, row in result_df.iterrows():
        sid = str(row[stock_col]).zfill(6)
        w = float(row[weight_col])
        stock_data = test_df[test_df['股票代码'].astype(str).str.zfill(6) == sid].sort_values('日期')

        if len(stock_data) < sell_offset:
            continue

        buy_price = float(stock_data.iloc[buy_offset - 1]['开盘']) if buy_offset <= len(stock_data) else float(stock_data.iloc[0]['开盘'])
        sell_price = float(stock_data.iloc[min(sell_offset - 1, len(stock_data) - 1)]['开盘'])
        ret = (sell_price - buy_price) / (buy_price + 1e-12)
        score += ret * w

    return score

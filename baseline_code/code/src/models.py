# code/src/models.py
# 多模型排序框架：XGBRanker + LGBMRanker + 模型融合
# 参考获奖队伍经验：
#   - O_O: XGBRanker做排序学习 + 均值方差优化
#   - 7355608: 多模型分数融合 + 后处理
#   - 柚子: 机器学习排序模型是核心优势

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb


# ============================================================
# 1. XGBoost Ranker 包装器
# ============================================================

class XGBRankerWrapper:
    """
    XGBoost 排序模型。

    目标函数 rank:ndcg 天然适合 Top-K 选股。
    获奖队伍 O_O 明确指出 XGBRanker 让"思路一下子就打开了"。
    """

    def __init__(self, params=None):
        default_params = {
            'objective': 'rank:pairwise',   # 支持连续标签
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 500,
            'early_stopping_rounds': 30,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0,
        }
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.best_ntree = None
        self.feature_names = None

    def fit(self, X_train, y_train, group_train, X_val=None, y_val=None, group_val=None):
        """
        训练 XGBRanker。

        参数:
        - X_train: (n_samples, n_features) 特征矩阵
        - y_train: (n_samples,) 标签（连续值，如超额收益）
        - group_train: list[int] 每个日期有多少只股票 [300, 298, 295, ...]
        - X_val, y_val, group_val: 验证集（可选）
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group_train)

        # 构建训练参数
        train_params = dict(self.params)

        # XGBoost 的 ranking objective + evals 会触发 ndcg 等需要整数标签的 metric
        # 显式指定 rmse 作为 eval_metric 避免报错
        if 'eval_metric' not in train_params:
            train_params['eval_metric'] = 'rmse'

        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None and group_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            dval.set_group(group_val)
            evals.append((dval, 'val'))

        early_stop = train_params.pop('early_stopping_rounds', None)
        n_estimators = train_params.pop('n_estimators', 500)
        callbacks = []
        if early_stop is not None and early_stop > 0:
            from xgboost.callback import EarlyStopping
            callbacks.append(EarlyStopping(rounds=early_stop))

        self.model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=50,
        )
        try:
            self.best_ntree = self.model.best_ntree_limit or n_estimators
        except Exception:
            self.best_ntree = n_estimators
        return self

    def predict(self, X):
        """预测排序分数"""
        dtest = xgb.DMatrix(X)
        ntree = self.best_ntree if self.best_ntree else 0
        # xgboost >= 2.0 使用 iteration_range 参数
        try:
            return self.model.predict(dtest, iteration_range=(0, ntree))
        except TypeError:
            return self.model.predict(dtest, ntree_limit=ntree)

    def get_feature_importance(self):
        """获取特征重要性（gain 类型）"""
        importance = self.model.get_score(importance_type='gain')
        return pd.Series(importance).sort_values(ascending=False)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = xgb.Booster(model_file=path)
        self.best_ntree = self.params['n_estimators']


# ============================================================
# 2. LightGBM Ranker 包装器
# ============================================================

class LGBRankerWrapper:
    """
    LightGBM 排序模型。

    使用 lambdarank 目标，与 XGBRanker 互补。
    两者树结构不同，融合效果好。
    """

    def __init__(self, params=None):
        default_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10, 30],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 500,
            'early_stopping_rounds': 30,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.best_iteration = None
        self.feature_names = None

    def fit(self, X_train, y_train, group_train, X_val=None, y_val=None, group_val=None):
        """
        训练 LGBMRanker。

        LightGBM lambdarank 需要整数标签，这里自动将连续标签
        转为组内排名（0-based, 越高越好）。
        """
        # 连续标签 → 组内整数排名
        y_train_int = _continuous_to_group_rank(y_train, group_train)
        y_val_int = None
        if y_val is not None and group_val is not None:
            y_val_int = _continuous_to_group_rank(y_val, group_val)

        dtrain = lgb.Dataset(X_train, label=y_train_int, group=group_train)
        valid_sets = [dtrain]
        valid_names = ['train']

        if X_val is not None and y_val_int is not None and group_val is not None:
            dval = lgb.Dataset(X_val, label=y_val_int, group=group_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('val')

        callbacks = [lgb.log_evaluation(50)]
        early_stop = self.params.get('early_stopping_rounds', 0)
        if early_stop and early_stop > 0:
            callbacks.insert(0, lgb.early_stopping(early_stop))

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration = self.model.best_iteration
        return self

    def predict(self, X):
        """预测排序分数"""
        return self.model.predict(X, num_iteration=self.best_iteration)

    def get_feature_importance(self):
        """获取特征重要性（gain 类型）"""
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
        self.best_iteration = self.params['n_estimators']


# ============================================================
# 3. Transformer 模型包装器（推理适配）
# ============================================================

class TransformerPredictor:
    """
    StockTransformer 的推理包装器，提供统一的 predict() 接口。

    Transformer 输出 3D 序列分数，这里将每日横截面分数展平为
    与树模型一致的 (n_samples,) 数组。
    """

    def __init__(self, model, device=None):
        self.model = model
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device

    def predict(self, sequences_tensor):
        """
        参数:
        - sequences_tensor: torch.Tensor 或 np.ndarray
                           [batch, n_stocks, seq_len, n_features]

        返回:
        - scores: np.ndarray [n_samples,] 展平的日截面分数
        """
        import torch
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            if isinstance(sequences_tensor, np.ndarray):
                sequences_tensor = torch.from_numpy(sequences_tensor).float()
            sequences_tensor = sequences_tensor.to(self.device)
            # 逐日处理（每天为一个 batch）
            for day_idx in range(len(sequences_tensor)):
                day_input = sequences_tensor[day_idx:day_idx + 1]
                day_scores = self.model(day_input).squeeze(0)
                all_scores.append(day_scores.cpu().numpy())
        return np.concatenate(all_scores)


# ============================================================
# 4. 多模型融合器
# ============================================================

class ModelEnsemble:
    """
    多模型分数融合器。

    策略：
    1. 每个模型对同一天的所有股票输出原始分数
    2. 每日截面内将分数转为排名百分位（消除模型量纲差异）
    3. 按配置权重加权平均各模型排名 → 融合分数
    4. 再按融合分数做日截面排名

    获奖队伍 7355608：分数融合 + 后处理是"比较重视的部分"。
    """

    def __init__(self, model_weights=None):
        self.model_weights = model_weights or {
            'xgboost': 0.40,
            'lightgbm': 0.35,
            'transformer': 0.25,
        }
        self.models = {}

    def add_model(self, name, model):
        """注册模型"""
        self.models[name] = model

    def predict_daily(self, daily_data, dates, stock_ids):
        """
        对多日数据逐日预测并融合。

        参数:
        - daily_data: dict, 键为模型名，值为该模型的输入数据
                      (每种模型可能有不同的输入格式)
        - dates: (n_samples,) 每条样本的日期
        - stock_ids: (n_samples,) 每条样本的股票代码

        返回:
        - DataFrame: [date, stock_id, ensemble_score, rank, xgb_score, lgb_score, ...]
        """
        df = pd.DataFrame({
            'date': dates,
            'stock_id': stock_ids,
        })

        # Step 1: 每个模型分别预测
        for name, model in self.models.items():
            if name in daily_data:
                raw_scores = model.predict(daily_data[name])
                df[f'{name}_score'] = raw_scores

        # Step 2: 日截面内排名百分位（消除量纲差异）
        for name in self.models:
            col = f'{name}_score'
            if col in df.columns:
                df[f'{name}_rank'] = df.groupby('date')[col].rank(pct=True)

        # Step 3: 加权融合
        df['ensemble_score'] = 0.0
        total_weight = 0.0
        for name, weight in self.model_weights.items():
            rank_col = f'{name}_rank'
            if rank_col in df.columns:
                df['ensemble_score'] += weight * df[rank_col]
                total_weight += weight

        if total_weight > 0:
            df['ensemble_score'] /= total_weight

        # Step 4: 融合后排名
        df['rank'] = df.groupby('date')['ensemble_score'].rank(ascending=False)

        return df.sort_values(['date', 'rank'])

    def predict_single_day(self, all_scores, stock_ids):
        """
        单日预测 + 融合（用于最终推理）。

        参数:
        - all_scores: dict, {model_name: np.array of shape (n_stocks,)}
        - stock_ids: list of str, 股票代码列表

        返回:
        - DataFrame: [stock_id, ensemble_score, rank, ...]
        """
        n = len(stock_ids)
        df = pd.DataFrame({'stock_id': stock_ids})

        # 各模型排名百分位
        rank_cols = []
        for name, scores in all_scores.items():
            if scores is None or len(scores) != n:
                continue
            df[f'{name}_score'] = scores
            df[f'{name}_rank'] = pd.Series(scores).rank(pct=True).values
            rank_cols.append(f'{name}_rank')

        # 加权融合
        df['ensemble_score'] = 0.0
        total_weight = 0.0
        for name, weight in self.model_weights.items():
            rank_col = f'{name}_rank'
            if rank_col in df.columns:
                df['ensemble_score'] += weight * df[rank_col]
                total_weight += weight

        if total_weight > 0:
            df['ensemble_score'] /= total_weight

        df['rank'] = df['ensemble_score'].rank(ascending=False)
        return df.sort_values('rank')


# ============================================================
# 5. 辅助：序列数据 → 表格数据转换
# ============================================================

def sequences_to_tabular(sequences, targets, stock_lists, method='last_day'):
    """
    将 3D 序列数据转换为 2D 表格数据，供 XGBoost/LightGBM 使用。

    参数:
    - sequences: list of (n_stocks, seq_len, n_features) arrays
    - targets: list of (n_stocks,) arrays (标签)
    - stock_lists: list of list of stock_ids
    - method: 'last_day' 取最后一天特征 | 'flatten' 展平全部 | 'stats' 统计量

    返回:
    - X: (n_samples, n_features_out) 特征矩阵
    - y: (n_samples,) 标签
    - groups: list[int] 每天的样本数
    - all_dates: list 日期标记
    - all_stock_ids: list 股票代码
    """
    X_list = []
    y_list = []
    groups = []
    all_dates = []
    all_stock_ids = []

    for day_idx, (day_seq, day_target, day_stocks) in enumerate(
        zip(sequences, targets, stock_lists)
    ):
        n_stocks = len(day_stocks)
        if n_stocks == 0:
            continue

        if method == 'last_day':
            # 取序列最后一天的特征
            features = day_seq[:, -1, :]  # (n_stocks, n_features)
            X_list.append(features)
        elif method == 'stats':
            # 取序列的统计量：最后1天 + 均值 + std + 趋势
            last = day_seq[:, -1, :]
            mean = day_seq.mean(axis=1)
            std = day_seq.std(axis=1)
            trend = day_seq[:, -1, :] - day_seq[:, -10, :] if day_seq.shape[1] >= 10 else np.zeros_like(last)
            features = np.concatenate([last, mean, std, trend], axis=1)
            X_list.append(features)
        else:
            # flatten: 展平全部60天特征
            features = day_seq.reshape(n_stocks, -1)
            X_list.append(features)

        y_list.append(day_target)
        groups.append(n_stocks)
        all_dates.extend([f'day_{day_idx}'] * n_stocks)
        all_stock_ids.extend(day_stocks)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)

    # 处理 NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, groups, all_dates, all_stock_ids


# ============================================================
# 6. 统一训练入口
# ============================================================

def train_tree_models(
    train_sequences, train_targets, train_stock_lists,
    val_sequences=None, val_targets=None, val_stock_lists=None,
    xgb_params=None, lgb_params=None,
    feature_method='last_day',
    verbose=True,
):
    """
    一键训练 XGBRanker + LGBMRanker。

    返回:
    - models: dict {'xgboost': XGBRankerWrapper, 'lightgbm': LGBRankerWrapper}
    - feature_importance: dict 各模型的特征重要性
    """
    # 数据转换
    X_train, y_train, groups_train, _, _ = sequences_to_tabular(
        train_sequences, train_targets, train_stock_lists,
        method=feature_method
    )

    X_val, y_val, groups_val = None, None, None
    if val_sequences is not None and val_targets is not None:
        X_val, y_val, groups_val, _, _ = sequences_to_tabular(
            val_sequences, val_targets, val_stock_lists,
            method=feature_method
        )

    models = {}
    importances = {}

    # XGBoost
    if verbose:
        print("=" * 50)
        print("训练 XGBoost Ranker ...")
        print("=" * 50)
    xgb_model = XGBRankerWrapper(params=xgb_params)
    xgb_model.fit(X_train, y_train, groups_train, X_val, y_val, groups_val)
    models['xgboost'] = xgb_model
    importances['xgboost'] = xgb_model.get_feature_importance()
    if verbose:
        print(f"XGBoost best_ntree: {xgb_model.best_ntree}")

    # LightGBM
    if verbose:
        print("=" * 50)
        print("训练 LightGBM Ranker ...")
        print("=" * 50)
    lgb_model = LGBRankerWrapper(params=lgb_params)
    lgb_model.fit(X_train, y_train, groups_train, X_val, y_val, groups_val)
    models['lightgbm'] = lgb_model
    importances['lightgbm'] = lgb_model.get_feature_importance()
    if verbose:
        print(f"LightGBM best_iteration: {lgb_model.best_iteration}")

    return models, importances


# ============================================================
# 7. 辅助工具
# ============================================================

def _continuous_to_group_rank(y, groups):
    """
    将连续标签转为组内整数排名（0 ~ n-1，越高越好）。

    LightGBM lambdarank 要求整数标签，
    这里将连续值映射为组内排名（rank）再传入。
    """
    y = np.asarray(y, dtype=np.float64)
    result = np.zeros_like(y, dtype=np.int32)
    start = 0
    for g in groups:
        end = start + g
        if g > 0:
            group_vals = y[start:end]
            ranks = np.argsort(np.argsort(group_vals))  # 0-based rank
            result[start:end] = ranks
        start = end
    return result

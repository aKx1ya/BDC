import os

# 配置参数
sequence_length = int(os.getenv('BDC_SEQUENCE_LENGTH', '60'))
feature_num = os.getenv('BDC_FEATURE_NUM', '158+39')
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'seed': 42,
    'd_model': int(os.getenv('BDC_D_MODEL', '256')),          # Transformer输入维度
    'nhead': int(os.getenv('BDC_NHEAD', '4')),             # 注意力头数量
    'num_layers': int(os.getenv('BDC_NUM_LAYERS', '3')),        # Transformer层数
    'dim_feedforward': int(os.getenv('BDC_DIM_FEEDFORWARD', '512')), # 前馈网络维度
    'batch_size': int(os.getenv('BDC_BATCH_SIZE', '2')),        # 4GB GPU 默认小 batch，配合梯度累积保持有效 batch
    'gradient_accumulation_steps': int(os.getenv('BDC_GRAD_ACCUM_STEPS', '2')),
    'num_epochs': int(os.getenv('BDC_NUM_EPOCHS', '15')),
    'min_epochs': int(os.getenv('BDC_MIN_EPOCHS', '5')),
    'early_stopping_patience': int(os.getenv('BDC_EARLY_STOPPING_PATIENCE', '3')),
    'learning_rate': float(os.getenv('BDC_LEARNING_RATE', '2e-5')),  # warm start 1 epoch微调步长
    'weight_decay': float(os.getenv('BDC_WEIGHT_DECAY', '1e-5')),
    'dropout': float(os.getenv('BDC_DROPOUT', '0.05')),
    'feature_num': feature_num,
    'max_grad_norm': float(os.getenv('BDC_MAX_GRAD_NORM', '5.0')),
    'grad_clip': os.getenv('BDC_GRAD_CLIP', '1') == '1',
    'use_amp': os.getenv('BDC_USE_AMP', '1') == '1',
    'num_workers': int(os.getenv('BDC_NUM_WORKERS', '4')),
    'pin_memory': os.getenv('BDC_PIN_MEMORY', '1') == '1',
    'oom_retry_limit': int(os.getenv('BDC_OOM_RETRY_LIMIT', '2')),

    'pairwise_weight': float(os.getenv('BDC_PAIRWISE_WEIGHT', '0.0')), # 配对损失权重
    'base_weight': float(os.getenv('BDC_BASE_WEIGHT', '1.0')), # 非top-k样本权重
    'top5_weight': float(os.getenv('BDC_TOP5_WEIGHT', '2.0')), # top-5样本权重（应大于base_weight）

    'output_dir': os.getenv('BDC_OUTPUT_DIR', f'./model/{sequence_length}_{feature_num}'),
    'data_path': './data',
    'golden_val_ratio': float(os.getenv('BDC_GOLDEN_VAL_RATIO', '0.2')),
    'golden_val_random_state': int(os.getenv('BDC_GOLDEN_VAL_RANDOM_STATE', '42')),
    'golden_val_split_path': os.getenv('BDC_GOLDEN_VAL_SPLIT_PATH', './data/golden_validation_dates.json'),
    'purge_trading_days': int(os.getenv('BDC_PURGE_TRADING_DAYS', '5')),
    'label_buy_offset': int(os.getenv('BDC_LABEL_BUY_OFFSET', '1')),
    'label_sell_offset': int(os.getenv('BDC_LABEL_SELL_OFFSET', '6')),
    # 标签类型: absolute_return | excess_return | rank | direction
    'label_type': os.getenv('BDC_LABEL_TYPE', 'excess_return'),
    # 辅助任务: direction（方向预测）, volatility（波动率预测）, 逗号分隔
    'aux_tasks': os.getenv('BDC_AUX_TASKS', 'direction,volatility'),
    # 辅助任务损失权重
    'aux_direction_weight': float(os.getenv('BDC_AUX_DIRECTION_WEIGHT', '0.1')),
    'aux_volatility_weight': float(os.getenv('BDC_AUX_VOLATILITY_WEIGHT', '0.05')),
    'use_aux_direction': os.getenv('BDC_USE_AUX_DIRECTION', '0') == '1',
    'use_aux_volatility': os.getenv('BDC_USE_AUX_VOLATILITY', '0') == '1',
    # XGBoost 排序模型参数
    'xgb_objective': os.getenv('BDC_XGB_OBJECTIVE', 'rank:pairwise'),
    'xgb_learning_rate': float(os.getenv('BDC_XGB_LR', '0.05')),
    'xgb_max_depth': int(os.getenv('BDC_XGB_MAX_DEPTH', '6')),
    'xgb_n_estimators': int(os.getenv('BDC_XGB_N_ESTIMATORS', '500')),
    'xgb_early_stopping': int(os.getenv('BDC_XGB_EARLY_STOPPING', '30')),
    # LightGBM 排序模型参数
    'lgb_objective': os.getenv('BDC_LGB_OBJECTIVE', 'lambdarank'),
    'lgb_learning_rate': float(os.getenv('BDC_LGB_LR', '0.05')),
    'lgb_num_leaves': int(os.getenv('BDC_LGB_NUM_LEAVES', '63')),
    'lgb_max_depth': int(os.getenv('BDC_LGB_MAX_DEPTH', '7')),
    'lgb_n_estimators': int(os.getenv('BDC_LGB_N_ESTIMATORS', '500')),
    'lgb_early_stopping': int(os.getenv('BDC_LGB_EARLY_STOPPING', '30')),
    # 模型融合权重
    'xgb_ensemble_weight': float(os.getenv('BDC_ENSEMBLE_XGB_WEIGHT', '0.40')),
    'lgb_ensemble_weight': float(os.getenv('BDC_ENSEMBLE_LGB_WEIGHT', '0.35')),
    'transformer_ensemble_weight': float(os.getenv('BDC_ENSEMBLE_TRANSFORMER_WEIGHT', '0.25')),
    # 是否启用树模型（Phase 4 完整支持，当前 Phase 2 仅预留配置）
    'enable_xgb_ranker': os.getenv('BDC_ENABLE_XGB_RANKER', '1') == '1',
    'enable_lgb_ranker': os.getenv('BDC_ENABLE_LGB_RANKER', '1') == '1',
    # 特征增强开关
    'enable_cross_sectional': os.getenv('BDC_ENABLE_CROSS_SECTIONAL', '1') == '1',
    'enable_industry_features': os.getenv('BDC_ENABLE_INDUSTRY', '0') == '1',
    'enable_market_features': os.getenv('BDC_ENABLE_MARKET', '0') == '1',
    # Walk-forward 验证参数
    'wf_train_window': int(os.getenv('BDC_WF_TRAIN_WINDOW', '252')),
    'wf_test_window': int(os.getenv('BDC_WF_TEST_WINDOW', '5')),
    'wf_step_size': int(os.getenv('BDC_WF_STEP_SIZE', '5')),
    'wf_n_splits': int(os.getenv('BDC_WF_N_SPLITS', '10')),
    # 精排参数
    'max_per_sector': int(os.getenv('BDC_MAX_PER_SECTOR', '2')),
    'max_correlation': float(os.getenv('BDC_MAX_CORRELATION', '0.3')),
    'max_single_weight': float(os.getenv('BDC_MAX_SINGLE_WEIGHT', '0.45')),
    'risk_aversion': float(os.getenv('BDC_RISK_AVERSION', '1.0')),
    'candidate_pool_size': int(os.getenv('BDC_CANDIDATE_POOL_SIZE', '30')),
    'max_drawdown_threshold': float(os.getenv('BDC_MAX_DRAWDOWN_THRESHOLD', '0.15')),
    'crash_threshold': float(os.getenv('BDC_CRASH_THRESHOLD', '0.07')),
    'local_benchmark_hurdle': float(os.getenv('BDC_LOCAL_BENCHMARK_HURDLE', '0.02517949121691857')),
    'pretrained_model_path': os.getenv('BDC_PRETRAINED_MODEL_PATH', './model/60_158+39/best_model.pth'),
    'eval_initial_model': os.getenv('BDC_EVAL_INITIAL_MODEL', '1') == '1',
    'eval_only': os.getenv('BDC_EVAL_ONLY', '0') == '1',
    'enable_extra_factors': os.getenv('BDC_ENABLE_EXTRA_FACTORS', '0') == '1',
    'portfolio_weighting': os.getenv('BDC_PORTFOLIO_WEIGHTING', 'rank_softmax'),
    'portfolio_temperature': float(os.getenv('BDC_PORTFOLIO_TEMPERATURE', '0.85')),
    'portfolio_max_weight_sum': float(os.getenv('BDC_PORTFOLIO_MAX_WEIGHT_SUM', '1.0')),
    'portfolio_min_weight': float(os.getenv('BDC_PORTFOLIO_MIN_WEIGHT', '0.03')),
    'portfolio_max_single_weight': float(os.getenv('BDC_PORTFOLIO_MAX_SINGLE_WEIGHT', '0.45')),
    'portfolio_rank_prior': float(os.getenv('BDC_PORTFOLIO_RANK_PRIOR', '0.35')),
    'portfolio_cash_z_threshold': float(os.getenv('BDC_PORTFOLIO_CASH_Z_THRESHOLD', '-999.0')),
    'use_causal_temporal_mask': os.getenv('BDC_USE_CAUSAL_TEMPORAL_MASK', '0') == '1',
    'temporal_residual_scale': float(os.getenv('BDC_TEMPORAL_RESIDUAL_SCALE', '1.0')),
    'cross_residual_scale': float(os.getenv('BDC_CROSS_RESIDUAL_SCALE', '1.0')),
}

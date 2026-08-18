import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import (
	ADDITIONAL_FACTOR_COLUMNS,
	add_extra_factor_features,
	engineer_features_39,
	engineer_features_158plus39,
)


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def select_feature_columns(feature_num):
	columns = list(feature_cloums_map[feature_num])
	if config.get('enable_extra_factors', False):
		columns.extend(ADDITIONAL_FACTOR_COLUMNS)
	return columns


def safe_torch_load(checkpoint_path, device):
	try:
		return torch.load(checkpoint_path, map_location=device, weights_only=True)
	except TypeError:
		return torch.load(checkpoint_path, map_location=device)


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = select_feature_columns(config['feature_num'])

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])
	if config.get('enable_extra_factors', False):
		processed = add_extra_factor_features(processed)

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def allocate_portfolio_weights(scores, top_indices):
	mode = config.get('portfolio_weighting', 'equal').lower()
	max_sum = min(max(float(config.get('portfolio_max_weight_sum', 1.0)), 0.0), 1.0)
	top_indices = np.asarray(top_indices, dtype=np.int64)
	if len(top_indices) == 0 or max_sum <= 0:
		return []
	if mode == 'equal':
		return [max_sum / len(top_indices)] * len(top_indices)

	scores = np.asarray(scores, dtype=np.float64)
	top_scores = scores[top_indices]
	temperature = max(float(config.get('portfolio_temperature', 0.85)), 1e-6)

	if mode in {'softmax', 'score_softmax'}:
		logits = (top_scores - top_scores.max()) / temperature
	elif mode == 'rank_decay':
		decay = max(float(config.get('portfolio_rank_prior', 0.35)), 1e-6)
		logits = -np.arange(len(top_indices), dtype=np.float64) * decay
	else:
		finite_scores = scores[np.isfinite(scores)]
		median = np.median(finite_scores) if finite_scores.size else 0.0
		mad = np.median(np.abs(finite_scores - median)) if finite_scores.size else 1.0
		scale = mad * 1.4826
		if not np.isfinite(scale) or scale < 1e-6:
			scale = np.std(finite_scores) if finite_scores.size else 1.0
		scale = max(float(scale), 1e-6)
		robust_z = np.clip((top_scores - median) / scale, -8.0, 8.0)
		rank_prior = -np.arange(len(top_indices), dtype=np.float64) * float(config.get('portfolio_rank_prior', 0.35))
		logits = robust_z / temperature + rank_prior

		# If the best prediction is not above the cross-sectional center, keep cash.
		cash_threshold = float(config.get('portfolio_cash_z_threshold', -999.0))
		if robust_z[0] < cash_threshold:
			max_sum *= max(0.0, min(1.0, np.exp(robust_z[0] - cash_threshold)))

	logits = logits - np.max(logits)
	weights = np.exp(logits)
	weights = weights / (weights.sum() + 1e-12)

	min_weight = max(float(config.get('portfolio_min_weight', 0.03)), 0.0)
	if min_weight > 0 and min_weight * len(weights) < max_sum:
		floor_total = min_weight * len(weights)
		weights = weights * (max_sum - floor_total) + min_weight
	else:
		weights = weights * max_sum

	max_single = min(max(float(config.get('portfolio_max_single_weight', 0.45)), 0.0), max_sum)
	for _ in range(len(weights)):
		over_mask = weights > max_single
		if not over_mask.any():
			break
		excess = float((weights[over_mask] - max_single).sum())
		weights[over_mask] = max_single
		under_mask = ~over_mask
		if not under_mask.any() or excess <= 0:
			break
		weights[under_mask] += excess * weights[under_mask] / (weights[under_mask].sum() + 1e-12)

	weights = np.clip(weights, 0.0, max_single if max_single > 0 else max_sum)
	total = weights.sum()
	if total > max_sum:
		weights *= max_sum / (total + 1e-12)
	return weights.tolist()


def main():
	data_file = os.path.join(config['data_path'], 'train.csv')
	output_dir = config['output_dir']
	model_path = os.path.join(output_dir, 'best_model.pth')
	scaler_path = os.path.join(output_dir, 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')
	candidate_path = os.path.join('./output/', 'candidate_top30.csv')

	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')
	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到Transformer模型: {model_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	scaled_features = scaler.transform(processed[features]).astype(np.float32)
	processed[features] = pd.DataFrame(scaled_features, index=processed.index, columns=features)

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed, features, sequence_length, stock_ids, latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	# ==========================================
	# Stage 1: 多模型预测 + 融合 → Top30 候选池
	# ==========================================
	print("=" * 50)
	print(f"Stage 1: 候选池召回 (Top{config.get('candidate_pool_size', 30)})")
	print("=" * 50)

	all_scores = {}
	n_stocks = len(sequence_stock_ids)

	# 1a. Transformer 预测
	transformer = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	state_dict = safe_torch_load(model_path, device)
	transformer.load_state_dict(state_dict, strict=False)
	transformer.to(device).eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
		scores_transformer = transformer(x).squeeze(0).detach().cpu().numpy()
	all_scores['transformer'] = scores_transformer
	print(f"  Transformer: {n_stocks} stocks scored")

	# 1b. XGBoost（如果已训练）
	xgb_path = os.path.join(output_dir, 'xgboost_ranker.json')
	if os.path.exists(xgb_path) and config.get('enable_xgb_ranker', False):
		try:
			from models import XGBRankerWrapper
			xgb_model = XGBRankerWrapper()
			xgb_model.load(xgb_path)
			# 取最后一天特征
			X_last = sequences_np[:, -1, :].astype(np.float32)
			scores_xgb = xgb_model.predict(X_last)
			all_scores['xgboost'] = scores_xgb
			print(f"  XGBoost: {len(scores_xgb)} stocks scored")
		except Exception as e:
			print(f"  XGBoost: 加载失败 ({e})")

	# 1c. LightGBM（如果已训练）
	lgb_path = os.path.join(output_dir, 'lightgbm_ranker.txt')
	if os.path.exists(lgb_path) and config.get('enable_lgb_ranker', False):
		try:
			from models import LGBRankerWrapper
			lgb_model = LGBRankerWrapper()
			lgb_model.load(lgb_path)
			X_last = sequences_np[:, -1, :].astype(np.float32)
			scores_lgb = lgb_model.predict(X_last)
			all_scores['lightgbm'] = scores_lgb
			print(f"  LightGBM: {len(scores_lgb)} stocks scored")
		except Exception as e:
			print(f"  LightGBM: 加载失败 ({e})")

	# 1d. 融合
	if len(all_scores) > 1:
		from models import ModelEnsemble
		weights = {}
		if 'xgboost' in all_scores:
			weights['xgboost'] = config.get('xgb_ensemble_weight', 0.40)
		if 'lightgbm' in all_scores:
			weights['lightgbm'] = config.get('lgb_ensemble_weight', 0.35)
		if 'transformer' in all_scores:
			weights['transformer'] = config.get('transformer_ensemble_weight', 0.25)
		# 归一化
		total = sum(weights.values())
		weights = {k: v / total for k, v in weights.items()}

		ensemble = ModelEnsemble(weights)
		result = ensemble.predict_single_day(all_scores, sequence_stock_ids)
		final_scores = result['ensemble_score'].values
		candidate_df = result.copy()
		print(f"  Ensemble: {len(all_scores)} models fused (weights={ {k: f'{v:.2f}' for k,v in weights.items()} })")
	else:
		final_scores = scores_transformer
		candidate_df = pd.DataFrame({
			'stock_id': sequence_stock_ids,
			'transformer_score': scores_transformer,
		})
		candidate_df['ensemble_score'] = candidate_df['transformer_score']
		candidate_df['ensemble_rank_pct'] = candidate_df['ensemble_score'].rank(pct=True)

	# 1e. 取 Top30
	pool_size = min(config.get('candidate_pool_size', 30), n_stocks)
	top_indices = np.argsort(final_scores)[::-1][:pool_size]
	candidate_top = candidate_df.iloc[top_indices].copy()
	candidate_top['rank'] = range(1, len(candidate_top) + 1)

	os.makedirs('./output/', exist_ok=True)
	candidate_top.to_csv(candidate_path, index=False)
	print(f"  候选池: {candidate_path} ({len(candidate_top)} stocks)")

	# ==========================================
	# Stage 2: 精排 Top30 → Top5
	# ==========================================
	print("=" * 50)
	print("Stage 2: 精排 Top5")
	print("=" * 50)

	try:
		from postprocess import fine_ranking
		from features import SW_INDUSTRY_MAP

		# 准备行业信息
		candidate_top['industry'] = candidate_top['stock_id'].map(SW_INDUSTRY_MAP).fillna('未知')

		# 准备收益矩阵（用于相关性计算）
		returns_df = None
		try:
			pivot = processed.pivot_table(
				index='日期', columns='股票代码', values='涨跌幅', aggfunc='first'
			)
			returns_df = pivot.astype(float).fillna(0.0) / 100.0  # 转为小数
		except Exception:
			pass

		result_df = fine_ranking(
			candidate_df=candidate_top,
			price_df=processed,
			returns_df=returns_df,
			sector_map=SW_INDUSTRY_MAP,
			config=config,
		)
		if result_df is None or len(result_df) == 0:
			raise ValueError("精排返回空结果")
	except Exception as e:
		print(f"  精排失败 ({e})，回退到直接 Top5")
		top5_indices = np.argsort(final_scores)[::-1][:5]
		# 清理 stock_id
		clean_stocks = []
		for i in top5_indices:
			s = str(sequence_stock_ids[i]).strip()
			if s.endswith('.0'):
				s = s[:-2]
			clean_stocks.append(s.zfill(6))
		result_df = pd.DataFrame({
			'stock_id': clean_stocks,
			'weight': [0.2] * 5,
		})

	result_df.to_csv(output_path, index=False)

	print(f"\n预测日期: {latest_date.date()}")
	print(f"参与排序股票数: {n_stocks}")
	print(f"最终结果 ({len(result_df)} stocks):")
	for _, row in result_df.iterrows():
		print(f"  {row['stock_id']}: {float(row['weight']):.4f}")
	print(f"结果已写入: {output_path}")


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()

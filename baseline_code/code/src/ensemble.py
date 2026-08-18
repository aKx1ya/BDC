import argparse
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from config import config
from model import StockTransformer
from predict import allocate_portfolio_weights, build_inference_sequences, preprocess_predict_data


def _safe_torch_load(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _load_config(model_dir):
    path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(path):
        return dict(config)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    merged = dict(config)
    merged.update(cfg)
    return merged


def _predict_scores(model_dir, raw_df, device):
    cfg = _load_config(model_dir)
    config.update(cfg)
    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    scaled_features = scaler.transform(processed[features]).astype(np.float32)
    processed[features] = pd.DataFrame(scaled_features, index=processed.index, columns=features)
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed,
        features,
        cfg['sequence_length'],
        stock_ids,
        raw_df['日期'].max(),
    )

    model = StockTransformer(input_dim=len(features), config=cfg, num_stocks=len(stock_ids))
    model.load_state_dict(_safe_torch_load(os.path.join(model_dir, 'best_model.pth'), device), strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        scores = model(x).squeeze(0).detach().cpu().numpy()
    return pd.Series(scores, index=sequence_stock_ids, dtype=float)


def _read_weight(model_dir):
    path = os.path.join(model_dir, 'final_score.txt')
    if not os.path.exists(path):
        return 1.0
    text = open(path, 'r', encoding='utf-8').read()
    for line in text.splitlines():
        if 'Best final_score:' in line:
            try:
                return max(float(line.split(':')[-1].strip()), 1e-6)
            except ValueError:
                return 1.0
    return 1.0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Blend successful BDC model ranks.')
    parser.add_argument('--model-dirs', nargs='+', required=True)
    parser.add_argument('--output', default='./output/result.csv')
    args = parser.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_df = pd.read_csv(config['data_path'] + '/train.csv', dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    blended = None
    total_weight = 0.0
    for model_dir in args.model_dirs:
        scores = _predict_scores(model_dir, raw_df, device)
        ranks = scores.rank(ascending=False, method='average')
        weight = _read_weight(model_dir)
        blended = ranks * weight if blended is None else blended.add(ranks * weight, fill_value=0.0)
        total_weight += weight

    if blended is None or total_weight <= 0:
        raise ValueError('No valid model predictions to ensemble.')

    blended = blended / total_weight
    ordered = blended.sort_values(ascending=True)
    top_ids = ordered.index[:5].tolist()
    pseudo_scores = -ordered.loc[top_ids].values
    top_indices = np.arange(len(top_ids))
    weights = allocate_portfolio_weights(pseudo_scores, top_indices)
    output_df = pd.DataFrame({'stock_id': top_ids, 'weight': weights})
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f'Ensemble result written to {args.output}')


if __name__ == '__main__':
    main(sys.argv[1:])

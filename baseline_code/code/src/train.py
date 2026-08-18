import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from label import build_labels, build_aux_labels
from loss import WeightedRankingLoss
from model import StockTransformer
from utils import (
    ADDITIONAL_FACTOR_COLUMNS,
    add_extra_factor_features,
    engineer_features_39,
    engineer_features_158plus39,
)
from utils import create_ranking_dataset_vectorized
import joblib
import json
import multiprocessing as mp
import random
import gc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def select_feature_columns(feature_num):
    columns = list(feature_cloums_map[feature_num])
    if config.get('enable_extra_factors', False):
        columns.extend(ADDITIONAL_FACTOR_COLUMNS)
    return columns


def _build_label_and_clean(processed, drop_small_open=True):
    """使用 label.py 统一构造标签并清洗无效样本。"""
    label_type = config.get('label_type', 'excess_return')
    buy_offset = config.get('label_buy_offset', 1)
    sell_offset = config.get('label_sell_offset', 6)

    processed = build_labels(
        processed,
        label_type=label_type,
        buy_offset=buy_offset,
        sell_offset=sell_offset,
        drop_small_open=drop_small_open,
    )

    # 构造辅助任务标签
    aux_tasks_str = config.get('aux_tasks', '')
    if aux_tasks_str:
        aux_tasks = [t.strip() for t in aux_tasks_str.split(',') if t.strip()]
        processed = build_aux_labels(
            processed,
            aux_tasks=aux_tasks,
            buy_offset=buy_offset,
            sell_offset=sell_offset,
        )

    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = select_feature_columns(config['feature_num'])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    if config.get('enable_extra_factors', False):
        processed = add_extra_factor_features(processed)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.as_tensor(np.array(self.sequences[idx], copy=True), dtype=torch.float32),
            'targets': torch.as_tensor(np.array(self.targets[idx], copy=True), dtype=torch.float32),
            'relevance': torch.as_tensor(np.array(self.relevance_scores[idx], copy=True), dtype=torch.long),
            'stock_indices': torch.as_tensor(np.array(self.stock_indices[idx], copy=True), dtype=torch.long)
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }

def _autocast_context(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _compute_masked_loss(outputs, targets, masks, criterion):
    masked_outputs = outputs * masks + (1 - masks) * (-1e9)
    masked_targets = targets * masks
    batch_loss = None
    batch_size = outputs.size(0)

    for i in range(batch_size):
        valid_indices = masks[i].nonzero().squeeze()
        if valid_indices.numel() == 0:
            continue
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)

        valid_pred = masked_outputs[i][valid_indices]
        valid_true = masked_targets[i][valid_indices]
        if len(valid_pred) > 1:
            loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
            batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

    if batch_loss is not None:
        batch_loss = batch_loss / batch_size
    return batch_loss, masked_outputs, masked_targets


# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, amp_scaler, device, epoch, writer, amp_enabled, grad_accum_steps):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask

        with _autocast_context(device, amp_enabled):
            outputs = model(sequences)  # [batch, max_stocks] 预测分数
        batch_loss, masked_outputs, masked_targets = _compute_masked_loss(outputs, targets, masks, criterion)

        if batch_loss is not None:
            amp_scaler.scale(batch_loss / grad_accum_steps).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(dataloader))
            if should_step:
                amp_scaler.unscale_(optimizer)
                if config.get('grad_clip', True):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                else:
                    grad_norm = 0.0
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / local_step if local_step > 0 else 0, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch, amp_enabled=False):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            with _autocast_context(device, amp_enabled):
                outputs = model(sequences)
                batch_loss, masked_outputs, masked_targets = _compute_masked_loss(outputs, targets, masks, criterion)
            
            if batch_loss is not None:
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def build_golden_validation_split(
    df,
    sequence_length,
    val_ratio=0.2,
    random_state=42,
    split_path='./data/golden_validation_dates.json',
):
    """Build chronological purged validation dates with no tail embargo."""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    all_dates = pd.Index(sorted(df['日期'].dt.normalize().unique()))
    sell_offset = config.get('label_sell_offset', 6)
    purge_days = config.get('purge_trading_days', 5)
    if len(all_dates) <= sequence_length + sell_offset + purge_days:
        raise ValueError("可用交易日不足，无法构造黄金验证集")

    eligible_dates = pd.Index(all_dates[sequence_length - 1:-sell_offset])
    if len(eligible_dates) == 0:
        raise ValueError("没有满足序列长度和 T+6 标签要求的候选日期")

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    val_count = max(1, int(round(len(eligible_dates) * val_ratio)))
    if len(eligible_dates) <= val_count + purge_days:
        raise ValueError("候选交易日不足，无法同时保留验证集和 5 日净化带")

    val_dates = pd.Index(eligible_dates[-val_count:])
    train_dates = pd.Index(eligible_dates[:-(val_count + purge_days)])
    purge_dates = pd.Index(eligible_dates[-(val_count + purge_days):-val_count])
    split = {
        'description': 'Chronological purged golden validation split. Do not randomize during the AutoML loop.',
        'schema_version': 'purged_chronological_v2',
        'val_ratio': val_ratio,
        'random_state': random_state,
        'sequence_length': sequence_length,
        'label': 'T+1 open buy to T+6 open sell',
        'label_buy_offset': config.get('label_buy_offset', 1),
        'label_sell_offset': sell_offset,
        'purge_trading_days': purge_days,
        'tail_embargo_trading_days': 0,
        'eligible_date_count': int(len(eligible_dates)),
        'train_date_count': int(len(train_dates)),
        'purge_date_count': int(len(purge_dates)),
        'val_date_count': int(len(val_dates)),
        'created_from': 'data/train.csv',
        'train_start': train_dates[0].strftime('%Y-%m-%d'),
        'train_end': train_dates[-1].strftime('%Y-%m-%d'),
        'purge_dates': [d.strftime('%Y-%m-%d') for d in purge_dates],
        'val_dates': [d.strftime('%Y-%m-%d') for d in val_dates],
    }
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    eligible_set = set(pd.to_datetime(eligible_dates).normalize())
    val_set = set(pd.to_datetime(val_dates).normalize())
    invalid_dates = sorted(val_set - eligible_set)
    if invalid_dates:
        raise ValueError(f"黄金验证集中存在不再合法的日期: {invalid_dates[:5]}")

    print(f"全量数据范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
    print(f"黄金验证集候选日期数: {len(eligible_dates)}")
    print(f"黄金验证集目标日期数: {len(val_dates)} (chronological, no tail embargo)")
    print(f"训练目标日期数: {len(train_dates)}")
    print(f"净化带交易日数: {len(purge_dates)}")
    print(f"黄金验证集文件: {split_path}")

    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
    return df, train_dates, val_dates


def seed_worker(worker_id):
    worker_seed = config.get('seed', 42) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def effective_num_workers(requested):
    requested = int(requested)
    if requested <= 0:
        return 0
    if os.name == 'nt' and os.getenv('BDC_ALLOW_WINDOWS_WORKERS', '0') != '1':
        print(
            f"Windows spawn DataLoader detected; using num_workers=0 instead of {requested}. "
            "Set BDC_ALLOW_WINDOWS_WORKERS=1 to force multiprocessing."
        )
        return 0
    return requested


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB")
        return device
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def safe_torch_load(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_compatible_state_dict(model, checkpoint_path, device):
    state_dict = safe_torch_load(checkpoint_path, device)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"已加载兼容权重: {len(compatible)} tensors from {checkpoint_path}")
    if skipped:
        print(f"跳过形状不兼容权重: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    if missing:
        print(f"新增/未加载权重: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"未预期权重: {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")


# 主程序
def run_once(run_config):
    config.update(run_config)
    set_seed(run_config.get('seed', 42))
    output_dir = run_config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = run_config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    device = get_device()
    amp_enabled = bool(run_config.get('use_amp', True) and device.type == 'cuda')
    amp_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    
    # 1. 数据加载
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    full_df, train_target_dates, val_target_dates = build_golden_validation_split(
        full_df,
        config['sequence_length'],
        val_ratio=config.get('golden_val_ratio', 0.2),
        random_state=config.get('golden_val_random_state', 42),
        split_path=config.get('golden_val_split_path', './data/golden_validation_dates.json'),
    )
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理。黄金验证集按窗口结束日期过滤，特征工程只做一次以降低内存峰值。
    all_data, features = preprocess_data(full_df, is_train=True, stockid2idx=stockid2idx)
    del full_df
    gc.collect()
    
    # 3. 标准化
    feature_scaler = StandardScaler()

    all_data[features] = all_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    all_data = all_data.dropna(subset=features).copy()
    # 然后再缩放
    date_key = pd.to_datetime(all_data['日期']).dt.normalize()
    train_date_set = set(pd.to_datetime(train_target_dates).normalize())
    train_scale_mask = date_key.isin(train_date_set)
    if not train_scale_mask.any():
        raise ValueError("训练日期掩码为空，无法拟合标准化器")

    feature_scaler.fit(all_data.loc[train_scale_mask, features].astype(np.float32))
    scaled_features = feature_scaler.transform(all_data[features].astype(np.float32)).astype(np.float32)
    all_data[features] = pd.DataFrame(scaled_features, index=all_data.index, columns=features)
    joblib.dump(feature_scaler, os.path.join(output_dir, 'scaler.pkl'))
    gc.collect()

    
    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        all_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path'),
        allowed_window_end_dates=train_target_dates,
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        all_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        allowed_window_end_dates=val_target_dates,
    )
    del all_data
    gc.collect()

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    num_workers = effective_num_workers(config.get('num_workers', 0))
    pin_memory = bool(config.get('pin_memory', True) and device.type == 'cuda')
    generator = torch.Generator()
    generator.manual_seed(config.get('seed', 42))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    pretrained_model_path = config.get('pretrained_model_path')
    if pretrained_model_path:
        if os.path.exists(pretrained_model_path):
            load_compatible_state_dict(model, pretrained_model_path, device)
            print(f"已加载预训练模型: {pretrained_model_path}")
        else:
            print(f"预训练模型不存在，跳过加载: {pretrained_model_path}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config['num_epochs']),
        eta_min=config['learning_rate'] * 0.05,
    )
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1

        if config.get('eval_initial_model', False):
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, None, -1, amp_enabled
            )
            print(f"Initial Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Initial Eval {k}: {v:.4f}")
            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存初始最佳模型 - final score: {best_score:.4f}")
            if config.get('eval_only', False):
                with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
                    f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")
                if writer:
                    writer.close()
                return best_score
        
        no_improve_epochs = 0
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model,
                train_loader,
                criterion,
                optimizer,
                amp_scaler,
                device,
                epoch,
                writer,
                amp_enabled,
                max(1, config.get('gradient_accumulation_steps', 1)),
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch, amp_enabled
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 保存最佳模型（基于final score）
            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - final score: {best_score:.4f}")
            else:
                no_improve_epochs += 1

            if (
                epoch + 1 >= config.get('min_epochs', 5)
                and no_improve_epochs >= config.get('early_stopping_patience', 3)
            ):
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"no improvement for {no_improve_epochs} epochs."
                )
                break
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\\nBest final_score: {best_score:.6f}\\n")

        # 9. 训练树模型（XGBoost + LightGBM）- 获奖队伍核心经验
        if config.get('enable_xgb_ranker', True) or config.get('enable_lgb_ranker', True):
            print("\n" + "=" * 50)
            print("训练树模型 (XGBoost Ranker + LightGBM Ranker)")
            print("=" * 50)
            try:
                from models import sequences_to_tabular, train_tree_models

                # 用验证集做 early stopping
                val_seq = val_sequences if val_sequences else None
                val_tgt = val_targets if val_targets else None
                val_stk = val_stock_indices if val_stock_indices else None

                xgb_p = {
                    'objective': config.get('xgb_objective', 'rank:pairwise'),
                    'learning_rate': config.get('xgb_learning_rate', 0.05),
                    'max_depth': config.get('xgb_max_depth', 6),
                    'n_estimators': config.get('xgb_n_estimators', 500),
                    'early_stopping_rounds': config.get('xgb_early_stopping', 30),
                    'verbosity': 1,
                    'n_jobs': -1,
                    'tree_method': 'hist',
                    'random_state': config.get('seed', 42),
                }
                lgb_p = {
                    'objective': config.get('lgb_objective', 'lambdarank'),
                    'learning_rate': config.get('lgb_learning_rate', 0.05),
                    'num_leaves': config.get('lgb_num_leaves', 63),
                    'max_depth': config.get('lgb_max_depth', 7),
                    'n_estimators': config.get('lgb_n_estimators', 500),
                    'early_stopping_rounds': config.get('lgb_early_stopping', 30),
                    'verbosity': 1,
                    'n_jobs': -1,
                    'random_state': config.get('seed', 42),
                }

                tree_models, tree_importances = train_tree_models(
                    train_sequences, train_targets, train_stock_indices,
                    val_sequences=val_seq, val_targets=val_tgt, val_stock_lists=val_stk,
                    xgb_params=xgb_p, lgb_params=lgb_p,
                    feature_method='last_day',
                    verbose=True,
                )

                # 保存树模型
                for name, m in tree_models.items():
                    ext = 'json' if name == 'xgboost' else 'txt'
                    path = os.path.join(output_dir, f'{name}_ranker.{ext}')
                    m.save(path)
                    print(f"  {name} 已保存: {path}")

                # 打印特征重要性
                for name, imp in tree_importances.items():
                    if len(imp) > 0:
                        print(f"  {name} Top5 特征: {list(imp.head(5).index)}")

            except Exception as e:
                print(f"  树模型训练失败（不影响 Transformer 结果）: {e}")
                import traceback
                traceback.print_exc()

        if writer:
            writer.close()

        return best_score


def _is_cuda_oom(error):
    return isinstance(error, RuntimeError) and 'out of memory' in str(error).lower()


def main():
    run_config = dict(config)
    attempts = 0
    while True:
        try:
            return run_once(run_config)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or attempts >= run_config.get('oom_retry_limit', 2):
                raise
            attempts += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            old_batch = max(1, int(run_config.get('batch_size', 1)))
            new_batch = max(1, old_batch // 2)
            if new_batch == old_batch and old_batch == 1:
                raise
            run_config['batch_size'] = new_batch
            run_config['gradient_accumulation_steps'] = int(
                run_config.get('gradient_accumulation_steps', 1)
            ) * 2
            print(
                "CUDA OOM retry: "
                f"batch_size {old_batch} -> {new_batch}, "
                f"gradient_accumulation_steps -> {run_config['gradient_accumulation_steps']}"
            )


if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")

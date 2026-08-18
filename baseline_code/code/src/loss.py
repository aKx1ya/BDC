import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedRankingLoss(nn.Module):
    """
    Robust listwise ranking loss for same-day cross-section selection.

    y_true is the T+1 open to T+6 open return. The target distribution is built
    from median/MAD-normalized returns with Huber-style clipping so a single
    extreme stock cannot dominate the full cross-section update.
    """

    def __init__(
        self,
        temperature=1.0,
        k=5,
        weight_factor=2.0,
        pairwise_weight=0.0,
        base_weight=1.0,
        huber_delta=3.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.huber_delta = huber_delta

    @staticmethod
    def _row_median(values):
        sorted_values = values.sort(dim=1).values
        width = sorted_values.size(1)
        mid = width // 2
        if width % 2:
            return sorted_values[:, mid:mid + 1]
        return 0.5 * (sorted_values[:, mid - 1:mid] + sorted_values[:, mid:mid + 1])

    def _robust_targets(self, y_true):
        median = self._row_median(y_true)
        centered = y_true - median
        mad = self._row_median(centered.abs()).clamp_min(1e-6)
        robust = centered / mad
        return torch.clamp(robust, min=-self.huber_delta, max=self.huber_delta)

    def listwise_loss(self, y_pred, y_true, weights):
        pred_log_probs = F.log_softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(self._robust_targets(y_true) / self.temperature, dim=1)
        weighted_ce = -(target_probs * pred_log_probs * weights)
        return (weighted_ce.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)).mean()

    def pairwise_loss(self, y_pred, y_true, weights):
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise = F.softplus(-pred_diff * torch.sign(true_diff))
        num_pairs = mask.sum(dim=[1, 2]).clamp_min(1)
        return (pairwise * mask * weight_matrix).sum(dim=[1, 2]).div(num_pairs).mean()

    def forward(self, y_pred, y_true):
        _, top_indices = torch.topk(y_true, min(self.k, y_true.size(1)), dim=1)
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        weights.scatter_(1, top_indices, self.weight_factor)
        loss = self.listwise_loss(y_pred, y_true, weights)
        if self.pairwise_weight > 0:
            loss = loss + self.pairwise_weight * self.pairwise_loss(y_pred, y_true, weights)
        return loss


def compute_multitask_loss(outputs, aux_outputs, targets, aux_targets, criterion, config):
    """
    多任务联合损失。

    参数:
    - outputs: (batch, n_stocks) 主任务排序分数
    - aux_outputs: dict {'direction': (batch, n_stocks), 'volatility': ...}
    - targets: (batch, n_stocks) 主标签
    - aux_targets: dict {'direction': (batch, n_stocks), 'volatility': ...}
    - criterion: WeightedRankingLoss 实例
    - config: 配置字典

    返回: total_loss (scalar tensor)
    """
    total_loss = criterion(outputs, targets)

    # 辅助任务1：方向预测（BCE loss）
    if 'direction' in aux_outputs and 'direction' in aux_targets:
        dir_pred = aux_outputs['direction']
        dir_true = aux_targets['direction'].float()
        valid = ~torch.isnan(dir_true)
        if valid.any():
            dir_loss = F.binary_cross_entropy_with_logits(
                dir_pred[valid], dir_true[valid]
            )
            w = config.get('aux_direction_weight', 0.1)
            total_loss = total_loss + w * dir_loss

    # 辅助任务2：波动率预测（Huber loss）
    if 'volatility' in aux_outputs and 'volatility' in aux_targets:
        vol_pred = aux_outputs['volatility']
        vol_true = aux_targets['volatility'].float()
        valid = ~torch.isnan(vol_true)
        if valid.any():
            vol_loss = F.smooth_l1_loss(vol_pred[valid], vol_true[valid])
            w = config.get('aux_volatility_weight', 0.05)
            total_loss = total_loss + w * vol_loss

    return total_loss

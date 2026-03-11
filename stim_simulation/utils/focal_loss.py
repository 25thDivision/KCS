import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, Num_Qubits] (logits)
        # targets: [Batch, Num_Qubits] (0 or 1)
        
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # p_t
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Custom weighted BCE loss with better control
    """
    def __init__(self, pos_weight, neg_weight=1.0):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, inputs, targets):
        # Manual implementation for better control
        max_val = torch.clamp(inputs, min=0)
        loss = inputs - inputs * targets + max_val + torch.log(torch.exp(-max_val) + torch.exp(-inputs - max_val))
        
        # Apply weights
        weight = targets * self.pos_weight + (1 - targets) * self.neg_weight
        loss = loss * weight
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines BCE with additional penalty for all-zero predictions
    """
    def __init__(self, pos_weight, diversity_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.diversity_weight = diversity_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        # Penalty for predicting all zeros (encourages diversity)
        probs = torch.sigmoid(inputs)
        
        # 더 강력한 diversity penalty: 실제 error가 있는데 예측 안하면 큰 패널티
        error_mask = (targets == 1)
        if error_mask.sum() > 0:
            # Error가 있는 위치에서 sigmoid 값이 낮으면 패널티
            diversity_penalty = -torch.mean(probs[error_mask])
        else:
            diversity_penalty = -torch.mean(probs)
        
        return bce_loss + self.diversity_weight * diversity_penalty

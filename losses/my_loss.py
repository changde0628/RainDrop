import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            # Ensure tensors have at least 2 dimensions
            if S2_fea[i].dim() == 1:
                S2_tensor = S2_fea[i].unsqueeze(0)  # Add batch dimension
            else:
                S2_tensor = S2_fea[i]
                
            if S1_fea[i].dim() == 1:
                S1_tensor = S1_fea[i].detach().unsqueeze(0)  # Add batch dimension
            else:
                S1_tensor = S1_fea[i].detach()
            
            # Now apply softmax on the right dimension
            S2_distance = F.log_softmax(S2_tensor / self.temperature, dim=1)
            S1_distance = F.softmax(S1_tensor / self.temperature, dim=1)
            
            loss_KD_dis += F.kl_div(
                        S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_tensor, S1_tensor)
        
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs

# 在losses/my_loss.py中添加
@LOSS_REGISTRY.register()
class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, pred, target):
        # 計算FFT
        pred_freq = torch.fft.rfft2(pred)
        target_freq = torch.fft.rfft2(target)
        
        # 計算頻率域差異
        freq_diff = torch.abs(pred_freq - target_freq)
        
        # 高頻部分加權
        batch, channel, h, w = pred.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w//2+1))
        y_grid, x_grid = y_grid.to(pred.device), x_grid.to(pred.device)
        
        # 距離中心的距離作為權重（高頻權重更大）
        center_y, center_x = h//2, 0
        dist = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        weight = torch.clip(dist / max(h, w) * 2.0, 0.1, 1.0)
        
        # 應用權重
        weighted_diff = freq_diff * weight.unsqueeze(0).unsqueeze(0)
        
        # 計算損失
        if self.reduction == 'mean':
            loss = torch.mean(weighted_diff)
        elif self.reduction == 'sum':
            loss = torch.sum(weighted_diff)
        else:
            loss = weighted_diff
            
        return self.loss_weight * loss
import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, inputs, target, mask=None):
        if mask is not None:
            return self.mse(inputs * mask, target * mask)
        else:
            return self.mse(inputs, target)


class CELoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1):
        super(CELoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)

    def forward(self, inputs, target, mask=None):
        #logits: [B, C, L], target: [B, L], mask: [B, L]
        if mask is not None:
            target[mask == 0] = -1
        return self.ce(inputs, target)

import torch
import torch.nn as nn

__all__ = ['MseLoss']
    
class MseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        MSE = ((input - target) ** 2).sum() / input.data.nelement()
        return MSE
    
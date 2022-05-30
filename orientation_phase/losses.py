import torch
import torch.nn as nn

__all__ = ['CEL']
    
class CEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input, target):
        Loss = self.loss(input, target)
        return Loss

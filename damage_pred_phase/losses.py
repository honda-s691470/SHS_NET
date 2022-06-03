""" 
Copyright (c) 2019 XyChen  
The original source code for kl_divergence is released under the MIT License  
https://github.com/chxy95/Deep-Mutual-Learning/blob/master/LICENSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

__all__ = ['CEL', 'kl_divergence']

    
class CEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input, target):
        Loss = self.loss(input, target)
        return Loss
    
class kl_divergence(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)
    def forward(self, logits_1, logits_2):
        softmax_1 = self.softmax_1(logits_1)
        softmax_2 = self.softmax_2(logits_2)
        kl = (softmax_2 * torch.log((softmax_2 / (softmax_1+1e-10)) + 1e-10)).sum(dim=1)
        return kl.mean()
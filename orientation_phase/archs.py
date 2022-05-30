import torch
from torch import nn
import timm
from torchvision import models
import torch.nn.functional as F

__all__ = ['Effnet']
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
    
    
class Effnet(nn.Module):
    def __init__(self):
        super(Effnet,self).__init__()
        effnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.fc1 = nn.Linear(1280, 12)
        self.avg_pool = GlobalAvgPool2d()
        
    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        y = self.fc1(im)
        y = torch.log_softmax(y, dim=-1)
        return y
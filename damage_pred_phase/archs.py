"""
Copyright (c) 2018 Ryuichiro Hataya  
the source code of SELayer is released under the MIT License  
https://github.com/moskomule/senet.pytorch/blob/master/LICENSE

We built our base CNN model using timm(https://github.com/rwightman/pytorch-image-models).  
Copyright (c) 2019 Ross Wightman  
the source code of timm is released under the Apache License 2.0  
https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE

"""

import torch
from torch import nn
import timm
from torchvision import models
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

__all__ = ['ResNet34_Erosion_CE_MC_MUL', 'ResNet34_Erosion_CE_MC_MUL_withoutSE12', 'Effnet_Erosion_CE_MC_MUL', 'ResNet34_Erosion_CE_withoutSE12', 'Effnet_Erosion_CE_MC_MUL_without_pretrain', 'Effnet_JSN_CE', 'Effnet_JSN_CE_without_pretrain', 'ResNet34_JSN_CE', 'ResNet34_JSN_CE_withoutSE12', 'Effnet_Erosion_CE_MC_MUL_withoutSE12', 'Effnet_JSN_CE_RA_CAM', 'Effnet_JSN_CE_RA_CAM_withoutSE12', 'ResNet34_JSN_CE_RA_CAM', 'ResNet34_JSN_CE_RA_CAM_withoutSE', 'Effnet_JSN_CE_RA_CAM_without_pretrain', 'Effnet_Erosion_CE', 'Effnet_Erosion_CE_without_pretrain', 'Resnet34_Erosion_CE', 'Effnet_Erosion_CE_withoutSE', 'Effnet_JSN_CE_withoutSE12']


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self,
                 device='cpu'):
        super().__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
    
class ResNet34_Erosion_CE_MC_MUL(nn.Module):
    def __init__(self):
        super(ResNet34_Erosion_CE_MC_MUL, self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.resnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        return pred1, pred2   
    
class ResNet34_Erosion_CE_MC_MUL_withoutSE12(nn.Module):
    def __init__(self):
        super(ResNet34_Erosion_CE_MC_MUL_withoutSE12, self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        return pred1, pred2   
    
class Effnet_Erosion_CE_MC_MUL(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE_MC_MUL, self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        return pred1, pred2  
    
class ResNet34_Erosion_CE_withoutSE12(nn.Module):
    def __init__(self):
        super(ResNet34_Erosion_CE_withoutSE12,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )

    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class Effnet_Erosion_CE_MC_MUL_without_pretrain(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE_MC_MUL_without_pretrain, self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=None)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        return pred1, pred2       
    
class Effnet_JSN_CE(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class Effnet_JSN_CE_without_pretrain(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE_without_pretrain,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=None)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class ResNet34_JSN_CE(nn.Module):
    def __init__(self):
        super(ResNet34_JSN_CE,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.resnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class ResNet34_JSN_CE_withoutSE12(nn.Module):
    def __init__(self):
        super(ResNet34_JSN_CE_withoutSE12,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )

    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class Effnet_Erosion_CE_MC_MUL_withoutSE12(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE_MC_MUL_withoutSE12,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )
    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        return pred1, pred2  
    
class Effnet_JSN_CE_RA_CAM(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE_RA_CAM,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        pred3 = self.classifier3(im)
        return pred1, pred2, pred3

class Effnet_JSN_CE_RA_CAM_withoutSE12(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE_RA_CAM_withoutSE12,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        pred3 = self.classifier3(im)
        return pred1, pred2, pred3

class ResNet34_JSN_CE_RA_CAM(nn.Module):
    def __init__(self):
        super(ResNet34_JSN_CE_RA_CAM,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.resnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        pred3 = self.classifier3(im)
        return pred1, pred2, pred3  
    
class ResNet34_JSN_CE_RA_CAM_withoutSE(nn.Module):
    def __init__(self):
        super(ResNet34_JSN_CE_RA_CAM_withoutSE,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
    def forward(self,im):
        im = self.resnet(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        pred3 = self.classifier3(im)
        return pred1, pred2, pred3    
    
class Effnet_JSN_CE_RA_CAM_without_pretrain(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE_RA_CAM_without_pretrain,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=None)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier1 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )
    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        pred1 = self.classifier1(im)
        pred2 = self.classifier2(im)
        pred3 = self.classifier3(im)
        return pred1, pred2, pred3

    
class Effnet_Erosion_CE(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class Effnet_Erosion_CE_without_pretrain(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE_without_pretrain,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        self.selayer1 = SELayer(3,3)
        effnet = timm.create_model('efficientnet_b3', pretrained=False)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.effnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y    
    
class Resnet34_Erosion_CE(nn.Module):
    def __init__(self):
        super(Resnet34_Erosion_CE,self).__init__()
        self.liner_num1= 512
        self.liner_num2= 256
        self.selayer1 = SELayer(3,3)
        resnet = timm.create_model('resnet34', pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.selayer2 = SELayer(self.liner_num1)
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )

    def forward(self,im):
        im = self.selayer1(im)
        im = self.resnet(im)
        im = self.selayer2(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y    
    
class Effnet_Erosion_CE_withoutSE(nn.Module):
    def __init__(self):
        super(Effnet_Erosion_CE_withoutSE,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 6)
        )

    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
class Effnet_JSN_CE_withoutSE12(nn.Module):
    def __init__(self):
        super(Effnet_JSN_CE_withoutSE12,self).__init__()
        self.liner_num1= 1536
        self.liner_num2= 480
        effnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.effnet = nn.Sequential(*list(effnet.children())[:-2])
        self.avg_pool = GlobalAvgPool2d()

        self.classifier = nn.Sequential(
            nn.Linear(self.liner_num1, self.liner_num2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.liner_num2, 5)
        )

    def forward(self,im):
        im = self.effnet(im)
        im = self.avg_pool(im)
        
        y = self.classifier(im)
        return y
    
import os
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import albumentations as A
from albumentations.core.composition import Compose, OneOf
import random
from matplotlib import pyplot as plt
from PIL import Image

#https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
#https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#resizing-transforms-augmentationsgeometricresize

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
            
        mask = np.dstack(mask)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']           
            mask = augmented['mask']
        maskall=np.max(mask,axis=2)

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}
    
class Dataset_img_only(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, num_classes, transform=None):

        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        
        return img, {'img_id': img_id}
    
def make_loader(train_img_ids, test_img_ids, config): 
    img_dir=os.path.join(config['img_path'], config['dataset'], 'images')
    print(img_dir)
    train_transform_withAlbu = Compose([
        A.geometric.rotate.RandomRotate90(),
        A.Flip(),
        OneOf([
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
            A.RandomContrast(),
        ], p=1),
        A.geometric.resize.Resize(config['input_h'], config['input_w'], interpolation=3),
        A.Normalize()])

    test_transform_withAlbu = Compose([
        A.geometric.resize.Resize(config['input_h'], config['input_w'], interpolation=3),
        A.Normalize()])
    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['img_path'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['img_path'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform_withAlbu,
        )
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(config['img_path'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['img_path'], config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform_withAlbu,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    return train_loader, test_loader

def make_loader_test_only(test_img_ids, config, config2): 

    test_transform_withAlbu = Compose([
        A.geometric.resize.Resize(config['input_h'], config['input_w'], interpolation=3),
        A.Normalize()])
    
    test_dataset = Dataset_img_only(
        img_ids=test_img_ids,
        img_dir=os.path.join(config2['img_path'] + config2['data_dir']),
        img_ext=config2['img_ext'],
        num_classes=config['num_classes'],
        transform=test_transform_withAlbu,
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    return test_loader

import os
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, labels, img_ext, num_tasks, transform=None):
        self.num_tasks = num_tasks
        self.img_ids = img_ids
        self.img_dir = img_dir
        
        if self.num_tasks == 1:
            self.labels1 = labels[0]
        elif self.num_tasks == 2:
            self.labels1 = labels[0]
            self.labels2 = labels[1]
        elif self.num_tasks == 3:
            self.labels1 = labels[0]
            self.labels2 = labels[1]
            self.labels3 = labels[2]
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        if self.num_tasks == 1:
            labels1 = self.labels1[idx]
            labels = [float(labels1)]
        elif self.num_tasks == 2:
            labels1 = self.labels1[idx]
            labels2 = self.labels2[idx]
            labels = [float(labels1), float(labels2)]
        elif self.num_tasks == 3:
            labels1 = self.labels1[idx]
            labels2 = self.labels2[idx]
            labels3 = self.labels3[idx]
            labels = [float(labels1), float(labels2), float(labels3)]
            
        img = Image.open(os.path.join(self.img_dir, img_id + self.img_ext))
        img = img.convert("RGB") 
        img = self.transform(img)
        
        return img, labels, {'img_id': img_id} 

class Dataset_pred_only(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, num_tasks, transform=None):
        self.num_tasks = num_tasks
        self.img_ids = img_ids
        self.img_dir = img_dir
        #self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = Image.open(os.path.join(self.img_dir, img_id))
        img = img.convert("RGB") 
        img = self.transform(img)
        
        return img, {'img_id': img_id}     
    
def make_loader(train_ids, val_ids, img_ids_test, train_labels, val_labels, test_labels, config): 
    
    train_transform = transforms.Compose([
        transforms.Resize((config['input_h'],config['input_w']), interpolation=3),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.9),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    val_transform = transforms.Compose([
        transforms.Resize((config['input_h'],config['input_w']), interpolation=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['input_h'],config['input_w']), interpolation=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = Dataset(
        img_ids=train_ids,
        img_dir=os.path.join(config['check_path'], config['dataset']),
        labels=train_labels,
        img_ext=config['img_ext'],
        num_tasks=config['num_tasks'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_ids,
        img_dir=os.path.join(config['check_path'], config['dataset']),
        labels=val_labels,
        img_ext=config['img_ext'],
        num_tasks=config['num_tasks'],
        transform=val_transform)

    test_dataset = Dataset(
        img_ids=img_ids_test,
        img_dir=os.path.join(config['check_path'], config['dataset']),
        labels=test_labels,
        img_ext=config['img_ext'],
        num_tasks=config['num_tasks'],
        transform=test_transform)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    return train_loader, val_loader, test_loader

def make_loader_for_ScoreCAM(val_ids, config, config2): 
    
    val_transform = transforms.Compose([
        transforms.Resize((config['input_h'], config['input_w']), interpolation=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_dataset = Dataset_pred_only(
        img_ids=val_ids,
        img_dir=os.path.join(config['check_path'], config2['joint_dir_name']),
        num_tasks = config['num_tasks'],
        transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    return val_loader
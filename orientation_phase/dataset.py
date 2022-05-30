import os
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, labels, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.labels = labels
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        labels = self.labels[idx]
        image = Image.open(os.path.join(self.img_dir, img_id + self.img_ext))
        image= image.convert("RGB") 
        image = self.transform(image)

        return image, labels, {'img_id': img_id} 

def make_loader(img_ids_train, img_ids_test, img_labels_train, img_labels_test, config): 
    
    train_transform = transforms.Compose([
        transforms.Resize((config['input_h'],config['input_w']), interpolation=3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.9),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    test_transform = transforms.Compose([
        transforms.Resize((config['input_h'],config['input_w']), interpolation=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_labels = img_labels_train 
    test_labels = img_labels_test     

    train_dataset = Dataset(
        img_ids=img_ids_train,
        img_dir=os.path.join(config['img_path'], config['dataset']),
        labels=img_labels_train,
        img_ext=config['img_ext'],
        transform=train_transform)

    test_dataset = Dataset(
        img_ids=tuple(img_ids_test),
        img_dir=os.path.join(config['img_path'], config['dataset']),
        labels=img_labels_test,
        img_ext=config['img_ext'],
        transform=test_transform)
        
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


class Dataset_for_orientation(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = Image.open(os.path.join(self.img_dir, img_id + self.img_ext))
        image= image.convert("RGB") 
        image = self.transform(image)

        return image, {'img_id': img_id} 
    
def make_loader_for_orientation(img_ids_test, config, config2): 
    
    test_transform = transforms.Compose([
        transforms.Resize((config['input_h'] ,config['input_h']), interpolation=3),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])   

    test_dataset = Dataset_for_orientation(
        img_ids=tuple(img_ids_test),
        img_dir=os.path.join(config2['img_path'], config2['data_dir']),
        img_ext=config2['img_ext'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    return test_loader
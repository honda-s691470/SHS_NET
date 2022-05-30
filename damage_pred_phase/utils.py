import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import adabound
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def scheduler_maker(optimizer, config):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']/5, eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler

def optim_maker(params, optimizer, config):
    if optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif optimizer == 'Adabound':
        optimizer = adabound.AdaBound(params, lr=config['lr'], final_lr=0.5, amsbound=True)
    else:
        raise NotImplementedError
    return optimizer

def label_maker(config):
    img_id_label = pd.read_csv(config['check_path'] + config['all_label'])

    if config['num_tasks'] == 1:
        img_id_label["avg"] = img_id_label.iloc[0:,1]
        img_id_label_train_val, img_id_label_test = train_test_split(img_id_label, test_size=config['test_ratio'], random_state=1, stratify=img_id_label["avg"])
        
    elif config['num_tasks'] == 2:        
        img_id_label["avg"] = ((img_id_label.iloc[0:,1] + img_id_label.iloc[0:,2])/2).astype(int)
        img_id_label_train_val, img_id_label_test = train_test_split(img_id_label, test_size=config['test_ratio'], random_state=1, stratify=img_id_label["avg"])
        
    elif config['num_tasks'] == 3:        
        img_id_label["avg"] = ((img_id_label.iloc[0:,1] + img_id_label.iloc[0:,2]+ img_id_label.iloc[0:,3])/3).astype(int)
        img_id_label_train_val, img_id_label_test = train_test_split(img_id_label, test_size=config['test_ratio'], random_state=1, stratify=img_id_label["avg"])

    else:
        print("Please check num_tasks in config")
        
    img_id_label_train_val.reset_index(inplace=True, drop=True)
    img_id_label_test.reset_index(inplace=True, drop=True)
    print("number of images before sample reduction", len(img_id_label_train_val))
    img_id_label_train_val = img_id_label_train_val.sample(frac=config['inclusion_ratio'], random_state=0).reset_index(drop=True)
    print("number of images after sample reduction", len(img_id_label_train_val))
    img_ids, img_ids_test = img_id_label_train_val.iloc[0:,0], img_id_label_test.iloc[0:,0]
    
    img_labels = []
    img_labels_test =[]
    for i in range(config['num_tasks']):
        img_labels.append(img_id_label_train_val.iloc[0:,i+1])
        img_labels_test.append(tuple(img_id_label_test.iloc[0:,i+1]))

    return img_ids, img_ids_test, img_labels, img_labels_test, img_id_label_train_val, img_id_label_test

def over_sampling (img_id_label_train_val, train_index, over_samp_param, task_type):
    
    img_id_label_train = img_id_label_train_val.iloc[train_index]
    img_id_label_train_list= []
    print("before_oversampling")
    if task_type == 'erosion':
        for i in range (6):
            img_id_label_train_list.append(i)
            img_id_label_train_list[i] = img_id_label_train.index[img_id_label_train["avg"]==i].tolist()
            print("score" + f'{i}',len(img_id_label_train_list[i]))
    elif task_type == 'narrowing':
        for i in range (5):
            img_id_label_train_list.append(i)
            img_id_label_train_list[i] = img_id_label_train.index[img_id_label_train["avg"]==i].tolist()
            print("score" + f'{i}',len(img_id_label_train_list[i]))

    if over_samp_param== 0:
        pass
    else:
        print("after_oversampling")
        if task_type == 'erosion':
            for i in range (6-1):
                img_id_label_train_list[i+1] = img_id_label_train_list[i+1]*(round((len(img_id_label_train_list[0])/(len(img_id_label_train_list[i+1]))*over_samp_param)))
                print("score"+ f'{i+1}',len(img_id_label_train_list[i+1]))
            #make new train index    
            train_index=[]
            for i in range(6):
                train_index.extend(img_id_label_train_list[i])
                
        if task_type == 'narrowing':
            for i in range (5-1):
                img_id_label_train_list[i+1] = img_id_label_train_list[i+1]*(round((len(img_id_label_train_list[0])/(len(img_id_label_train_list[i+1]))*over_samp_param)))
                print("score"+ f'{i+1}',len(img_id_label_train_list[i+1]))
            #make new train index    
            train_index=[]
            for i in range(5):
                train_index.extend(img_id_label_train_list[i])
    return train_index

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x

def cm_visualize(label_list_merge, score_pred_model, config, val_test, i):
    labels = sorted(list(set(label_list_merge[i])))
    cmx_data = confusion_matrix(label_list_merge[i], score_pred_model[i], labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig = plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx/(np.sum(df_cmx)+1.0e-5), annot=True, annot_kws={"fontsize":16}, cmap='Blues', fmt='.1%')
    plt.xlabel("Predict-labels", fontsize=18)
    plt.ylabel("True-labels", fontsize=18)
    plt.show()
    fig.savefig(config['check_path'] + 'damage_pred_log/models/%s/conf_mat_%s_%s.png' % (config['name'], val_test, i))


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result
import argparse
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2
import adabound
from torch.optim import lr_scheduler
import torch.optim as optim
import collections
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
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs']/2, eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler

def optim_maker(params, config):
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adabound':
        optimizer = adabound.AdaBound(params, lr=config['lr'], final_lr=0.5, amsbound=True)
    else:
        raise NotImplementedError
    return optimizer

def label_maker(config):
    img_id_label_csv = pd.read_csv(config['label'])
    print("number of label", collections.Counter(img_id_label_csv["label"]))
    img_label = img_id_label_csv.iloc[0:,1]

    img_id_label_train, img_id_label_test = train_test_split(img_id_label_csv, test_size=config['test_ratio'], random_state=42, stratify=img_label)
          
    img_id_label_train.reset_index(inplace=True, drop=True)
    img_id_label_test.reset_index(inplace=True, drop=True)
    
    img_ids_train, img_ids_test = list(img_id_label_train.iloc[0:,0]), list(img_id_label_test.iloc[0:,0])
    img_labels_train, img_labels_test = list(img_id_label_train.iloc[0:,1]), list(img_id_label_test.iloc[0:,1])
            
    return img_ids_train, img_ids_test, img_labels_train, img_labels_test

def fig_maker(df_log):
    plt.figure()
    plt.plot(df_log["epoch"], df_log["loss"], color='blue', linestyle='-', label='train_loss')
    plt.plot(df_log["epoch"], df_log["val_loss"], color='green', linestyle='--', label='val_loss')
    plt.plot(df_log["epoch"], df_log["val_acc"], color='orange', linestyle='--', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()  

def cor_orient(pred_dataframe, config2):
    for i, output in enumerate(pred_dataframe.iloc[0:,0]):
        img = cv2.imread(config2['img_path'] + config2['data_dir'] + "/" +  pred_dataframe.iloc[i,1] + config2['img_ext'])

        if output >= 4:
            if output==4 or output==8:
                pass
            elif output==5 or output==9:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif output==6 or output==10:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif output==7 or output==11:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                pass
            y, x, c = img.shape

            if config2['trimming_one_hand'] == True:
                img = img[int(y*config2['trim_ratio_one_hand_h']): int(y-y*config2['trim_ratio_one_hand_h']), 
                          int(x*config2['trim_ratio_one_hand_w']): int(x-x*config2['trim_ratio_one_hand_w'])]
                y, x, c = img.shape
                
            if config2['resize_one_hand'] == True:
                scale_x = config2['output_w']/x
                scale_y = config2['output_h']/y
                if(scale_x < scale_y): 
                    resize_img = cv2.resize(img, dsize=None, fx=scale_x, fy=scale_x, interpolation = cv2.INTER_AREA)
                    y, x, c = resize_img.shape
                else:
                    resize_img = cv2.resize(img, dsize=None, fx=scale_y, fy=scale_y, interpolation = cv2.INTER_AREA)
                    y, x, c = resize_img.shape  
                img = np.zeros((config2['output_h'], config2['output_w'], 3), dtype = np.uint8)
                img[0:y, 0:x] = resize_img
                y, x, c = img.shape
                
            if config2['output_h'] - y >= 0:
                img = cv2.copyMakeBorder(img, int((config2['output_h']-y)/2), int((config2['output_h']-y)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
            else:
                print("please trimming the height of the x-ray image of one hand", " Pixels that need to be trimmed", (config2['output_h']-y))
                break
            if config2['output_w']-x >= 0:
                img = cv2.copyMakeBorder(img, 0, 0 ,int((config2['output_w']-x)/2), int((config2['output_w']-x)/2), cv2.BORDER_CONSTANT, (0,0,0))
            else:
                print("please trimming the width of the x-ray image of one hand", " Pixels that need to be trimmed", (config2['output_w']-x))
                break
            if output >=4 and output<8:
                name = pred_dataframe.iloc[i,1] + "_L"
                cv2.imwrite(config2['img_path'] + config2['output_dir'] + "/" + name + config2['img_ext'], img)
            elif output >=8 and output<12:
                name = pred_dataframe.iloc[i,1] + "_R"
                cv2.imwrite(config2['img_path'] + config2['output_dir'] + "/" + name + config2['img_ext'], img)
                
            print("==== ", name, "====")
            plt.imshow(img)
            plt.show()
        else:
            if output==0:
                pass
            elif output==1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif output==2:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif output==3:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                pass
            y, x, c = img.shape
            if config2['trimming_two_hand'] == True:
                img = img[int(y*config2['trim_ratio_two_hand_h']): int(y-y*config2['trim_ratio_two_hand_h']), 
                          int(x*config2['trim_ratio_two_hand_w']): int(x-x*config2['trim_ratio_two_hand_w'])]
                y, x, c = img.shape
                
            imgL = img[int(0): int(y), 0: int(x/2)]
            imgR = img[int(0): int(y), int(x/2): int(x)]
            Ly, Lx, Lc = imgL.shape
            Ry, Rx, Rc = imgR.shape

            if config2['resize_two_hand'] == True:
                scale_Lx = config2['output_w']/Lx
                scale_Ly = config2['output_h']/Ly
                scale_Rx = config2['output_w']/Rx
                scale_Ry = config2['output_h']/Ry
                if(scale_Lx < scale_Ly): 
                    resize_imgL = cv2.resize(imgL, dsize=None, fx=scale_Lx, fy=scale_Lx, interpolation = cv2.INTER_AREA)
                    Ly, Lx, Lc = resize_imgL.shape
                else:
                    resize_imgL = cv2.resize(imgL, dsize=None, fx=scale_Ly, fy=scale_Ly, interpolation = cv2.INTER_AREA)
                    Ly, Lx, Lc = resize_imgL.shape 
                    
                if(scale_Rx < scale_Ry): 
                    resize_imgR = cv2.resize(imgR, dsize=None, fx=scale_Rx, fy=scale_Rx, interpolation = cv2.INTER_AREA)
                    Ry, Rx, Rc = resize_imgR.shape
                else:
                    resize_imgR = cv2.resize(imgR, dsize=None, fx=scale_Ry, fy=scale_Ry, interpolation = cv2.INTER_AREA)
                    Ry, Rx, Rc = resize_imgR.shape  
                    
                imgL = np.zeros((config2['output_h'], config2['output_w'], 3), dtype = np.uint8)
                imgR = np.zeros((config2['output_h'], config2['output_w'], 3), dtype = np.uint8)
                imgL[0:Ly, 0:Lx] = resize_imgL
                imgR[0:Ry, 0:Rx] = resize_imgR
                Ly, Lx, Lc = imgL.shape  
                Ry, Rx, Rc = imgR.shape 
 
            if (config2['output_h']) - Ly >= 0 or (config2['output_h'] - Ry) >= 0:
                imgL = cv2.copyMakeBorder(imgL, int((config2['output_h']-Ly)/2), int((config2['output_h']-Ly)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
                imgR = cv2.copyMakeBorder(imgR, int((config2['output_h']-Ry)/2), int((config2['output_h']-Ry)/2), 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
            else:
                print("please trimming the height of the x-ray image of two hand", " Pixels that need to be trimmed", 
                      (config2['output_h']-Ly), (config2['output_h']-Ry))
                break
            if (config2['output_w'] -Lx) >= 0 or (config2['output_w'] - Rx) >= 0:
                imgL = cv2.copyMakeBorder(imgL, 0, 0 ,int((config2['output_w']-Lx)/2), int((config2['output_w']-Lx)/2), cv2.BORDER_CONSTANT, (0,0,0))
                imgR = cv2.copyMakeBorder(imgR, 0, 0 ,int((config2['output_w']-Rx)/2), int((config2['output_w']-Rx)/2), cv2.BORDER_CONSTANT, (0,0,0))
            else:
                print("please trimming the width of the x-ray image of two hand", " Pixels that need to be trimmed", 
                      (config2['output_w']-Lx), (config2['output_w']-Rx))
                break
                
            nameL = pred_dataframe.iloc[i,1] + "_L"
            nameR = pred_dataframe.iloc[i,1] + "_R"
            cv2.imwrite(config2['img_path'] + config2['output_dir'] + "/" + nameL + config2['img_ext'], imgL)
            cv2.imwrite(config2['img_path'] + config2['output_dir'] + "/" + nameR + config2['img_ext'], imgR)  

            print("==== ", nameL, "====")
            plt.imshow(imgL)
            plt.show()
            print("==== ", nameR, "====")
            plt.imshow(imgR)
            plt.show()
            
            
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


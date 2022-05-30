import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
from albumentations.augmentations import transforms
import cv2
import numpy as np
import seaborn as sns; sns.set() 
from matplotlib import pyplot as plt
import pandas as pd


def scheduler_maker(optimizer, config):
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
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

def finger(config, config2, coord_df, img, i):
    for j in range(10):  
        coord_x = int(coord_df.iloc[i][1+j]/config['input_w']*config2['origin_w'])
        coord_y = int(coord_df.iloc[i][16+j]/config['input_h']*config2['origin_h'])  
        img_crop = img[max(0,coord_y-36):max(0,coord_y+36), max(0,coord_x-36):max(0,coord_x+36)]
        finger_img_id = coord_df.iloc[i,0].replace('_L', '_L_'+coord_df.columns[1+j].split('_')[0]).replace('_R', '_R_'+coord_df.columns[1+j].split('_')[0])
        cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "finger" + "/" + finger_img_id, img_crop)

def mc_mul_cmc(config, config2, coord_df, img, i):

    #make mc_mul image
    mc_mul_coord_x = int(coord_df.iloc[i][11]/config['input_w']*config2['origin_w'])
    mc_mul_coord_y = int(coord_df.iloc[i][26]/config['input_h']*config2['origin_h'])
    img_crop_mc_mul = img[max(0,mc_mul_coord_y-36):max(0,mc_mul_coord_y+36), max(0,mc_mul_coord_x-36):max(0,mc_mul_coord_x+36)]
    mc_mul_img_id = coord_df.iloc[i,0].replace('_L', '_L_MC_MUL').replace('_R', '_R_MC_MUL')
    cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "mc_mul" + "/" + mc_mul_img_id, img_crop_mc_mul)

    #make cmc image
    cmc_coord_x = int(coord_df.iloc[i][12]/config['input_w']*config2['origin_w'])
    cmc_coord_y = int(coord_df.iloc[i][27]/config['input_h']*config2['origin_h'])
    img_crop_cmc = img[max(0,cmc_coord_y-36):max(0,cmc_coord_y+36), max(0,cmc_coord_x-48):max(0,cmc_coord_x+48)]
    cmc_img_id = coord_df.iloc[i,0].replace('_L', '_L_CMC').replace('_R', '_R_CMC')
    cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "cmc" + "/" + cmc_img_id, img_crop_cmc)

def ulna_radi_luna_navi_racam(config, config2, coord_df, img, i):
    if "LF" in coord_df.iloc[i,0]:
        #ulna_radi
        ulna_radi_x_min = int(coord_df.iloc[i][13]/config['input_w']*config2['origin_w'])
        ulna_radi_x_max = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w'])
        ulna_radi_y_min = int(coord_df.iloc[i][30]/config['input_h']*config2['origin_h'])
        ulna_radi_y_max = int(coord_df.iloc[i][28]/config['input_h']*config2['origin_h'])
        ulna_radi_ypoint = int((ulna_radi_y_max+ulna_radi_y_min)/2)
        ulna_radi_xpoint = int((ulna_radi_x_max+ulna_radi_x_min)/2)     

        #Luna_navi
        luna_navi_x_min = int(coord_df.iloc[i][14]/config['input_w']*config2['origin_w'])
        luna_navi_x_max = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w'])
        luna_navi_y_min = int(coord_df.iloc[i][26]/config['input_h']*config2['origin_h'])
        luna_navi_y_max = int(coord_df.iloc[i][28]/config['input_h']*config2['origin_h'])    
        luna_navi_ypoint = int((luna_navi_y_max+luna_navi_y_min)/2)
        luna_navi_xpoint = int((luna_navi_x_max+luna_navi_x_min)/2)  

        #racam
        racam_y_min = int(coord_df.iloc[i][26]/config['input_h']*config2['origin_h'])
        racam_y_max = int(coord_df.iloc[i][30]/config['input_h']*config2['origin_h'])    
        racam_ypoint = int(((racam_y_max-racam_y_min)/3)*2+racam_y_min)
        racam_xpoint = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w']) 
    else:
        ulna_radi_x_min = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w'])
        ulna_radi_x_max = int(coord_df.iloc[i][13]/config['input_w']*config2['origin_w'])
        ulna_radi_y_min = int(coord_df.iloc[i][30]/config['input_h']*config2['origin_h'])
        ulna_radi_y_max = int(coord_df.iloc[i][28]/config['input_h']*config2['origin_h'])
        ulna_radi_ypoint = int((ulna_radi_y_max+ulna_radi_y_min)/2)
        ulna_radi_xpoint = int((ulna_radi_x_max+ulna_radi_x_min)/2)

        luna_navi_x_min = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w'])
        luna_navi_x_max = int(coord_df.iloc[i][14]/config['input_w']*config2['origin_w'])
        luna_navi_y_min = int(coord_df.iloc[i][26]/config['input_h']*config2['origin_h'])
        luna_navi_y_max = int(coord_df.iloc[i][28]/config['input_h']*config2['origin_h'])     
        luna_navi_ypoint = int((luna_navi_y_max+luna_navi_y_min)/2)
        luna_navi_xpoint = int((luna_navi_x_max+luna_navi_x_min)/2)  

        racam_y_min = int(coord_df.iloc[i][26]/config['input_h']*config2['origin_h'])
        racam_y_max = int(coord_df.iloc[i][30]/config['input_h']*config2['origin_h'])    
        racam_ypoint = int(((racam_y_max-racam_y_min)/3)*2+racam_y_min)
        racam_xpoint = int(coord_df.iloc[i][15]/config['input_w']*config2['origin_w']) 

    img_crop_ulna_radi = img[max(0,ulna_radi_ypoint-60):max(0,ulna_radi_ypoint+60), max(0,ulna_radi_xpoint-125):max(0,ulna_radi_xpoint+125)] 
    img_crop_luna_navi = img[max(0,luna_navi_ypoint-60):max(0,luna_navi_ypoint+60), max(0,luna_navi_xpoint-100):max(0,luna_navi_xpoint+100)]
    img_crop_racam = img[max(0,racam_ypoint-50):max(0,racam_ypoint+50), max(0,racam_xpoint-50):max(0,racam_xpoint+50)] 

    ulna_radi_img_id = coord_df.iloc[i,0].replace('_L', '_L_ULNA_RADI').replace('_R', '_R_ULNA_RADI')
    luna_navi_img_id = coord_df.iloc[i,0].replace('_L', '_L_LUNA_NAVI').replace('_R', '_R_LUNA_NAVI')
    racam_img_id = coord_df.iloc[i,0].replace('_L', '_L_RACAM').replace('_R', '_R_RACAM')
    cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "ulna_radi" + "/" + ulna_radi_img_id, img_crop_ulna_radi)
    cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "luna_navi" + "/" + luna_navi_img_id, img_crop_luna_navi)
    cv2.imwrite(config2['img_path'] + config2['data_dir'] + "/" + "racam" + "/" + racam_img_id, img_crop_racam)
    
def RLdiscrim(df_img):    
    #One-hot-encoding as follows: right = 0, left = 1
    orientation=[]
    for i in range(len(df_img)):
        index = df_img[i].find("_L")
        if index != -1:
            orientation.append(int(1))
        else: 
            orientation.append(int(0))   
    return orientation
    
def coord_reconst(config, config2, dfx, dfy):
    """
    The coordinates in coord_final.csv are the coordinate data of the image resized to 224 x 224. 
    Therefore, convert to the coordinate data for the size of the original image (630 x 910 by default).
    """
    resizeshape = (config['input_w'], config['input_h'])
    imgshape = (config2['origin_w'], config2['origin_h'])
    ptarray_x=np.array(dfx)
    ptarray_y=np.array(dfy)
    ptarray_stdx=(ptarray_x-(resizeshape[0]/2))/(resizeshape[0])
    ptarray_stdy=(ptarray_y-(resizeshape[1]/2))/(resizeshape[1])
    ptarray_x=ptarray_stdx*imgshape[0]+imgshape[0]/2
    ptarray_y=ptarray_stdy*imgshape[1]+imgshape[1]/2
    return ptarray_x, ptarray_y

def calc_angle(ptarray_x, ptarray_y, orientation):
    rad_list = []
    degree_list = []
    degree_list2 = []
    for i in range (len(ptarray_x)):
        PIP4 = np.array([ptarray_x[i,3],ptarray_y[i,3]])
        MCP4 = np.array([ptarray_x[i,8],ptarray_y[i,8]])
        Wrist2 = np.array([ptarray_x[i,11],ptarray_y[i,11]])
        # Define vectors
        vec_MP = PIP4 - MCP4
        vec_MW = Wrist2 - MCP4
        vec_zero = [1, 0]
        # Calculation of Cosine
        length_vec_MP = np.linalg.norm(vec_MP)
        length_vec_MW = np.linalg.norm(vec_MW)
        length_vec_zero = np.linalg.norm(vec_zero)
        inner_productPW = np.inner(vec_MP, vec_MW)
        inner_productPZ = np.inner(vec_MP, vec_zero)
        inner_productWZ = np.inner(vec_MW, vec_zero)

        cosPW = inner_productPW / (length_vec_MP * length_vec_MW)
        cosPZ = inner_productPZ / (length_vec_MP * length_vec_zero)
        cosWZ = inner_productWZ / (length_vec_MW * length_vec_zero)
        radPW = np.arccos(cosPW)
        radPZ = np.arccos(cosPZ)
        radWZ = np.arccos(cosWZ)

        if orientation[i]==1:
            degree_list.append(180-np.rad2deg(radPZ)-np.rad2deg(radWZ))
        else:
            degree_list.append(-180+np.rad2deg(radPZ)+np.rad2deg(radWZ))
    
    sns.distplot(
        degree_list, bins=20, color='#123456', label='number of data', axlabel='ulnar_deviation',
        kde=False,
        rug=False
    )

    plt.legend() 
    plt.show()    
    
    degree =pd.DataFrame(degree_list)
    degree = degree.set_axis(['rad'], axis='columns')
    return degree

def calc_ratio(ptarray_x, ptarray_y):
    ratio_list = []
    for i in range (len(ptarray_x)):
        MC4d = np.array([ptarray_x[i,8],ptarray_y[i,8]])
        MC4p = np.array([ptarray_x[i,11],ptarray_y[i,11]])
        Ossa = np.array([ptarray_x[i,13],ptarray_y[i,13]])
        # Define vectors
        vec_MC4 = MC4d-MC4p
        vec_Ossa = MC4p-Ossa
        # Calculating the norm
        length_vec_MC4 = np.linalg.norm(vec_MC4)
        length_vec_Ossa = np.linalg.norm(vec_Ossa)
        ratio = length_vec_Ossa/length_vec_MC4
        ratio_list.append(ratio)
        
    sns.distplot(
        ratio_list, bins=20, color='#123456', label='number of data', axlabel='ratio',
        kde=False,
        rug=False
    )

    plt.legend()
    plt.show()
    
    ratio =pd.DataFrame(ratio_list)
    ratio = ratio.set_axis(['ratio'], axis='columns')
    return ratio
    
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
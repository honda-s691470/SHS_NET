import numpy as np
import torch
import torch.nn.functional as F


def sdr(output, target):
    num = target.size(0)
    c = target.size(1)
    SDR_all=0
    R_all = 0
    for i in range(num):

        for chan in range(c):
            img = output[i, chan]
            mask = target[i, chan]
            #Calculating the remainder with fmod and obtaining the x-coordinate.
            img_x, img_y = torch.fmod(torch.argmax(img), img.size(1)), torch.argmax(img)/img.size(1)
            mask_x, mask_y = torch.fmod(torch.argmax(mask), mask.size(1)), torch.argmax(mask)/mask.size(1)
            mse_x, mse_y = (img_x-mask_x)**2, (img_y-mask_y)**2
            mse = mse_x+mse_y
            R = torch.sqrt(mse.float())
            R_all += R
            if R < 4:
                SDR_all += 1
     
    SDR = SDR_all/(num*c)
    R_ave = R_all/(num*c)
    return SDR, R_ave

def sdr_stratified(output, target):
    num = target.size(0)
    c = target.size(1)
    SDR_2=0
    SDR_4=0
    SDR_6=0
    SDR_8=0
    SDR_10=0 
    R_all = 0
    
    for i in range(num):

        for chan in range(c):
            img = output[i, chan]
            mask = target[i, chan]
            #Calculating the remainder with fmod and obtaining the x-coordinate.
            img_x, img_y = torch.fmod(torch.argmax(img), img.size(1)), torch.argmax(img)/img.size(1)
            mask_x, mask_y = torch.fmod(torch.argmax(mask), mask.size(1)), torch.argmax(mask)/mask.size(1)
            mse_x, mse_y = (img_x-mask_x)**2, (img_y-mask_y)**2
            mse = mse_x+mse_y
            R = torch.sqrt(mse.float())
            R_all += R
            if R < 2:
                SDR_2 += 1
            if R < 4:
                SDR_4 += 1 
            if R < 6:
                SDR_6 += 1
            if R < 8:
                SDR_8 += 1
            if R < 10:
                SDR_10 += 1     
    SDR_2_all, SDR_4_all, SDR_6_all, SDR_8_all, SDR_10_all = SDR_2/(num*c), SDR_4/(num*c), SDR_6/(num*c), SDR_8/(num*c), SDR_10/(num*c)
    R_ave = R_all/(num*c)
    return  SDR_2_all, SDR_4_all, SDR_6_all, SDR_8_all, SDR_10_all, R_ave

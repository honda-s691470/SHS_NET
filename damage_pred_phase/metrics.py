import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import correlation
from sklearn.metrics import mean_squared_error

def Pearson(pred, label):
    coef = 1 - correlation(pred, label)
    return coef

def RMSE(pred, label):
    rmse = np.sqrt(mean_squared_error(pred, label))
    return rmse

from utils import AverageMeter, str2bool
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import archs
from collections import OrderedDict
import pandas as pd
import numpy as np


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter()}
    model.train()
        
    pbar = tqdm(total=len(train_loader))
    for input, label, _ in train_loader:
        input = input.cuda()
        labels = label.type(torch.cuda.LongTensor)
        pred_model = model(input)
        
        loss = criterion(pred_model, labels) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        avg_meters['loss'].update(loss.item(), input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return postfix 
    
def val(val_loader, model, criterion, epoch, scheduler, config, train_log, log, best_val_acc):

    avg_meters = {'val_loss': AverageMeter(),'val_acc': AverageMeter()}
    model.eval() 
              
    pred_list = []
    label_list = []
    ID_list = []
    val_acc = 0
    miss_num = 0
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))     
        for input, label, ID in val_loader:
            input = input.cuda()
            labels = label.type(torch.cuda.LongTensor)
            ID = ID["img_id"]
            pred_model = model(input)
            
            loss = criterion(pred_model, labels)
            val_acc += (pred_model.max(1)[1] == labels).sum()/len(val_loader.dataset)
            miss_num += (pred_model.max(1)[1] != labels).sum()
            val_acc = val_acc.item()

            pred = pred_model.max(1)[1].to('cpu').detach().numpy().copy()  
            label_model = labels.to('cpu').detach().numpy().copy()  
            
            avg_meters['val_loss'].update(loss.item(), input.size(0))
            val_log = OrderedDict([('val_loss', avg_meters['val_loss'].avg), ('val_acc', val_acc)])
            
            pred_list.extend(pred)
            label_list.extend(label_model)
            ID_list.extend(ID)
            
            pbar.set_postfix(val_log)
            pbar.update(1)           
        pbar.close()                
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['val_loss']) 

        ID_df = pd.DataFrame(ID_list).set_axis(['ID'], axis='columns', inplace=False)
        label_model_df = pd.DataFrame(label_list).set_axis(['label'], axis='columns', inplace=False)
        pred_model_df = pd.DataFrame(pred_list).set_axis(['pred'], axis='columns', inplace=False)
        All_dataframe=pd.concat([pred_model_df, label_model_df, ID_df], axis=1)    

        print('loss %.4f - val_loss %.4f - val_acc %.4f - Number of mistakes %.0f' 
              % (train_log['loss'], val_log['val_loss'], val_log['val_acc'], miss_num))

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(val_log['val_loss'])
        log['val_acc'].append(val_log['val_acc'])


        if val_log['val_acc'] > best_val_acc:
            torch.save(model.state_dict(), 'orientation_pred_log/models/' + config['name'] + f'/model.pth')
            best_score1 = [val_log['val_loss'], val_log['val_acc']]
            print("=> saved best model")
            best_val_acc = val_log['val_acc']
            All_dataframe.to_csv(f'orientation_pred_log/models/%s/all_label_pred.csv' % config['name'], index=False)
            
        df_log = pd.DataFrame(log)
        df_log.to_csv(f'orientation_pred_log/models/%s/log.csv' % config['name'], index=False)
           
    return best_val_acc, df_log

def pred_orient(test_loader, model, config, config2):
    print("===load_model===")
    model.load_state_dict(torch.load(config2['img_path'] + 'orientation_pred_log/models/' + config['name'] + f'/model.pth'))
    model.eval()       
    
    pred_list = []
    ID_list = []
    with torch.no_grad():    
        for input, ID in test_loader:
            input = input.cuda()
            ID = ID["img_id"]
            pred_model = model(input)
            pred = pred_model.max(1)[1].to('cpu').detach().numpy().copy()  
               
            pred_list.extend(pred)
            ID_list.extend(ID)
        print("ID_list",ID_list)    
        ID_df = pd.DataFrame(ID_list).set_axis(['ID'], axis='columns', inplace=False)
        pred_model_df = pd.DataFrame(pred_list).set_axis(['pred'], axis='columns', inplace=False)
        pred_dataframe=pd.concat([pred_model_df, ID_df], axis=1)    
        
        torch.cuda.empty_cache()
    return pred_dataframe


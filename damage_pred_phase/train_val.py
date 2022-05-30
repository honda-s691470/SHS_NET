from utils import AverageMeter, str2bool, cm_visualize
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import archs
from collections import OrderedDict
from metrics import Pearson
from metrics import RMSE
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def backpropagation1(optimizer, loss1, avg_meter_loss, input, pbar):
    optimizer[0].zero_grad()
    loss1.backward()
    optimizer[0].step() 

    avg_meter_loss.update(loss1.item(), input.size(0))

    postfix = OrderedDict([('loss1', avg_meter_loss.avg)])
    pbar.set_postfix(postfix)
    pbar.update(1)
    return postfix

def backpropagation2(optimizer, loss1, loss2, avg_meter_loss1, avg_meter_loss2, input, pbar):      
    optimizer[0].zero_grad()
    loss1.backward()
    optimizer[0].step() 

    optimizer[1].zero_grad()
    loss2.backward()
    optimizer[1].step()

    avg_meter_loss1.update(loss1.item(), input.size(0))
    avg_meter_loss2.update(loss2.item(), input.size(0))

    postfix = OrderedDict([('loss1', avg_meter_loss1.avg), ('loss2', avg_meter_loss2.avg)])
    pbar.set_postfix(postfix)
    pbar.update(1)
    return postfix

def model_output(model, input, config):
    pred_model1 = [0,0,0]
    pred_model2 = [0,0,0] 
    if config['mutual'] == 'yes':     
        if config['num_tasks']==1:
            pred_model1[0] = model[0](input)
            pred_model2[0] = model[1](input)         
        elif config['num_tasks']==2:
            pred_model1[0], pred_model1[1] = model[0](input)
            pred_model2[0], pred_model2[1] = model[1](input)                
        elif config['num_tasks']==3:
            pred_model1[0], pred_model1[1], pred_model1[2] = model[0](input)
            pred_model2[0], pred_model2[1], pred_model2[2] = model[1](input) 
        return pred_model1, pred_model2
    else:
        if config['num_tasks']==1:
            pred_model1[0] = model[0](input)
        elif config['num_tasks']==2:
            pred_model1[0], pred_model1[1] = model[0](input)             
        elif config['num_tasks']==3:
            pred_model1[0], pred_model1[1], pred_model1[2] = model[0](input) 
        return pred_model1
    
    
def model_save(coef_model, best_coef, model, at_best_list, K_num, All_dataframe, config, model_num):
    if coef_model > best_coef:
        torch.save(model.state_dict(), config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_{model_num}.pth')
        best_coef = coef_model
        print(f'=> saved best model{model_num}')
        at_best_list = All_dataframe.values.tolist()
    return at_best_list, best_coef
    

def train(train_loader, model, criterion, optimizer, config):
    if config['mutual'] == 'yes': 
        avg_meters = {'loss1': AverageMeter(),'loss2': AverageMeter()}
        model[0].train()
        model[1].train()
    else:
        avg_meters = {'loss1': AverageMeter()}
        model[0].train()
            
    labels = [0,0,0]
    pbar = tqdm(total=len(train_loader))
    for input, label, _ in train_loader:
        input = input.cuda()

        for i in range (config['num_tasks']):
            labels[i] = label[i].type(torch.cuda.LongTensor)
 
        if config['mutual'] == 'yes':   
            pred_model1, pred_model2 = model_output(model, input, config)
            loss1_sum = 0
            loss2_sum = 0
            for i in range(config['num_tasks']):
                loss1 = criterion[0](pred_model1[i], labels[i]) + criterion[1](pred_model1[i], pred_model2[i].detach()) 
                loss2 = criterion[0](pred_model2[i], labels[i]) + criterion[1](pred_model2[i], pred_model1[i].detach())   
                loss1_sum += loss1
                loss2_sum += loss2
            postfix = backpropagation2(optimizer, loss1_sum, loss2_sum, avg_meters['loss1'], avg_meters['loss2'], input, pbar)
            
        else:
            pred_model1 = model_output(model, input, config)
            loss1_sum = 0
            for i in range(config['num_tasks']):
                loss1 = criterion[0](pred_model1[i], labels[i]) 
                loss1_sum += loss1
            postfix = backpropagation1(optimizer, loss1_sum, avg_meters['loss1'], input, pbar)

    pbar.close()
    return postfix 
    
def validate(val_loader, model, criterion, epoch, scheduler, lr, config, train_log, log, K_num, best_coef, at_best_list, restart):
    if restart == 'off':
        if config['mutual'] == 'yes': 
            avg_meters = {'val_loss1': AverageMeter(),'val_loss2': AverageMeter()}
            model[0].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_1.pth'))
            model[0].eval()
            model[1].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_2.pth'))
            model[1].eval()
        else:
            avg_meters = {'val_loss1': AverageMeter()}
            model[0].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_1.pth'))
            model[0].eval()   
    else:
        if config['mutual'] == 'yes': 
            avg_meters = {'val_loss1': AverageMeter(),'val_loss2': AverageMeter()}
            model[0].eval() 
            model[1].eval()
        else:
            avg_meters = {'val_loss1': AverageMeter()}
            model[0].eval() 
              
    pred1_list = []
    pred2_list = []
    label_list = []
    ID_list = []
    joint_num_list =[]
    labels = [0,0,0]
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))     
        for input, label, ID in val_loader:
            input = input.cuda()
            for i in range (config['num_tasks']):
                labels[i] = label[i].type(torch.cuda.LongTensor)
                
            if config['mutual'] == 'yes':   
                pred_model1, pred_model2 = model_output(model, input, config)
                loss1_sum = 0
                loss2_sum = 0
                pred1=[]
                pred2=[]
                label_model=[]
                for i in range(config['num_tasks']):
                    loss1 = criterion[0](pred_model1[i], labels[i])  
                    loss2 = criterion[0](pred_model2[i], labels[i])   
                    loss1_sum += loss1
                    loss2_sum += loss2
                    pred1.extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())
                    pred2.extend(pred_model2[i].max(1)[1].to('cpu').detach().numpy().copy())
                    label_model.extend(labels[i].to('cpu').detach().numpy().copy())
                pred1_list.extend(pred1)
                pred2_list.extend(pred2) 
                avg_meters['val_loss1'].update(loss1_sum.item(), input.size(0))
                avg_meters['val_loss2'].update(loss2_sum.item(), input.size(0))  
                val_log = OrderedDict([('val_loss1', avg_meters['val_loss1'].avg),('val_loss2', avg_meters['val_loss2'].avg)])                
            else:
                pred_model1 = model_output(model, input, config)
                loss1_sum = 0
                pred1=[]
                label_model=[]
                for i in range(config['num_tasks']):
                    loss1 = criterion[0](pred_model1[i], labels[i]) 
                    loss1_sum += loss1                
                    pred1.extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())   
                    label_model.extend(labels[i].to('cpu').detach().numpy().copy())
                avg_meters['val_loss1'].update(loss1_sum.item(), input.size(0))
                val_log = OrderedDict([('val_loss1', avg_meters['val_loss1'].avg)])
                pred1_list.extend(pred1)
            label_list.extend(label_model)
            
            for i in range (config['num_tasks']):
                ID_ext = ID["img_id"]
                joint_num = np.ones(len(ID_ext))*(i+1)
                ID_list.extend(ID_ext)
                joint_num_list.extend(joint_num)
            pbar.set_postfix(val_log)
            pbar.update(1)           
        pbar.close()     
        
        ID_df = pd.DataFrame(ID_list).set_axis(['ID'], axis='columns', inplace=False)
        joint_num_df = pd.DataFrame(joint_num_list).set_axis(['joint_num'], axis='columns', inplace=False)
        label_model_df = pd.DataFrame(label_list).set_axis(['label'], axis='columns', inplace=False)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['val_loss1']) 

        if config['mutual'] == 'yes':
            coef_model1 = Pearson(pred1_list, label_list)
            pred_model1_df = pd.DataFrame(pred1_list).set_axis(['pred1'], axis='columns', inplace=False)
            All_dataframe1=pd.concat([pred_model1_df, label_model_df, ID_df, joint_num_df], axis=1)    
      
            coef_model2 = Pearson(pred2_list, label_list)
            pred_model2_df = pd.DataFrame(pred2_list).set_axis(['pred2'], axis='columns', inplace=False)
            All_dataframe2=pd.concat([pred_model2_df, label_model_df, ID_df, joint_num_df], axis=1)     
         
            print('loss1 %.4f - loss2 %.4f - val_loss1 %.4f - val_loss2 %.4f - coef1 %.4f - coef2 %.4f' 
                  % (train_log['loss1'], train_log['loss2'], val_log['val_loss1'],  val_log['val_loss2'], coef_model1, coef_model2))

            log['epoch'].append(epoch)
            log['lr'].append(lr)
            log['loss1'].append(train_log['loss1'])
            log['loss2'].append(train_log['loss2'])
            log['val_loss1'].append(val_log['val_loss1'])
            log['val_loss2'].append(val_log['val_loss2'])
            log['coef1'].append(coef_model1)
            log['coef2'].append(coef_model2)
            
            at_best_list[0], best_coef[0] = model_save(coef_model1, best_coef[0], model[0], at_best_list[0], K_num, All_dataframe1, config, 1)
            at_best_list[1], best_coef[1] = model_save(coef_model2, best_coef[1], model[1], at_best_list[1], K_num, All_dataframe2, config, 2)
        else:
            coef_model1 = Pearson(pred1_list, label_list)
            pred_model1_df = pd.DataFrame(pred1_list).set_axis(['pred1'], axis='columns', inplace=False)
            All_dataframe1=pd.concat([pred_model1_df, label_model_df, ID_df, joint_num_df], axis=1)    
       
            print('loss1 %.4f - val_loss1 %.4f - coef1 %.4f' 
                  % (train_log['loss1'], val_log['val_loss1'], coef_model1))

            log['epoch'].append(epoch)
            log['lr'].append(lr)
            log['loss1'].append(train_log['loss1'])
            log['val_loss1'].append(val_log['val_loss1'])
            log['coef1'].append(coef_model1)
            
            at_best_list[0], best_coef[0] = model_save(coef_model1, best_coef[0], model[0], at_best_list[0], K_num, All_dataframe1, config, 1) 
             
        pd.DataFrame(log).to_csv(config['check_path'] + f'damage_pred_log/models/%s/log{K_num}.csv' %
                                 config['name'], index=False)   
        torch.cuda.empty_cache()
    
    return at_best_list

def validate_bestmodel(at_best_list, config, K_num, val_loader, model):
    DF1 = pd.DataFrame(at_best_list[0]).set_axis(['pred', 'label', 'img_id', 'joint_number'], axis='columns', inplace=False)
    DF1.to_csv(config['check_path'] + 'damage_pred_log/models/%s/pred1_labeldata_K%s.csv' % (config['name'], K_num), index=False)
    model[0].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_1.pth'))
    model[0].eval()
    if config['mutual'] == 'yes': 
        DF2 = pd.DataFrame(at_best_list[1]).set_axis(['pred', 'label', 'img_id', 'joint_number'], axis='columns', inplace=False)     
        DF2.to_csv(config['check_path'] + 'damage_pred_log/models/%s/pred2_labeldata_K%s.csv' % (config['name'], K_num), index=False)
        model[1].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_2.pth'))
        model[1].eval()
     
    ID_list = []
    labels = [0,0,0]
    label_list = [[],[],[]]
    joint_num_list =[]
    pred_model1_list = [[],[],[]]
    pred_model2_list = [[],[],[]]
    pred_ensem = [[],[],[]]
    pred_ensem_list = [[],[],[]]
    coef_model1 = [0,0,0]
    coef_model2 = [0,0,0]
    coef_ensem = [0,0,0]
    rmse_model1 = [0,0,0]
    rmse_model2 = [0,0,0]
    rmse_ensem = [0,0,0]
    cm = [0,0,0]
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))     
        for input, label, ID in val_loader:
            input = input.cuda()
            for i in range (config['num_tasks']):
                labels[i] = label[i].to('cpu').detach().numpy().copy()
                label_list[i].extend(labels[i])
            if config['mutual'] == 'yes': 
                pred_model1, pred_model2 = model_output(model, input, config)

                for i in range (config['num_tasks']):
                    pred_ensem[i] = (pred_model1[i] + pred_model2[i])/2
                    pred_model1_list[i].extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())
                    pred_model2_list[i].extend(pred_model2[i].max(1)[1].to('cpu').detach().numpy().copy())
                    pred_ensem_list[i].extend(pred_ensem[i].max(1)[1].to('cpu').detach().numpy().copy())
            else:
                pred_model1 = model_output(model, input, config)

                for i in range (config['num_tasks']):
                    pred_ensem_list[i].extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())

            for i in range (config['num_tasks']):
                ID_ext = ID["img_id"]
                joint_num = np.ones(len(ID_ext))*(i+1)
                ID_list.extend(ID_ext)
                joint_num_list.extend(joint_num)
                
            pbar.update(1)
        pbar.close()

    for i in range (config['num_tasks']): 
        if config['mutual'] == 'yes':
            coef_model1[i] = Pearson(pred_model1_list[i], label_list[i])
            coef_model2[i] = Pearson(pred_model2_list[i], label_list[i])
            coef_ensem[i] = Pearson(pred_ensem_list[i], label_list[i])
            rmse_model1[i] = RMSE(pred_model1_list[i], label_list[i])
            rmse_model2[i] = RMSE(pred_model2_list[i], label_list[i])
            rmse_ensem[i] = RMSE(pred_ensem_list[i], label_list[i])
            cm[i] = confusion_matrix(pred_ensem_list[i],label_list[i])
            print(f'===confusion matrix in ensem model label{i+1}===')
            print(cm[i])
            print('model1', 'coef_model1_%s: %.4f' % (i+1, coef_model1[i]))
            print('model2', 'coef_model2_%s: %.4f' % (i+1, coef_model2[i]))
            print('ensem', 'coef_ensem_%s: %.4f' % (i+1, coef_ensem[i]))
            print('model1', 'rmse_model1_%s: %.4f' % (i+1, rmse_model1[i]))
            print('model2', 'rmse_model2_%s: %.4f' % (i+1, rmse_model2[i]))
            print('ensem', 'rmse_ensem_%s: %.4f' % (i+1, rmse_ensem[i]))
            best_score = coef_model1[0:config['num_tasks']] + coef_model2[0:config['num_tasks']] + coef_ensem[0:config['num_tasks']] + rmse_model1[0:config['num_tasks']] + rmse_model2[0:config['num_tasks']] + rmse_ensem[0:config['num_tasks']]
            
        else:
            coef_ensem[i] = Pearson(pred_ensem_list[i], label_list[i])
            rmse_ensem[i] = RMSE(pred_ensem_list[i], label_list[i])
            cm[i] = confusion_matrix(pred_ensem_list[i],label_list[i])
            print(f'===confusion matrix label{i+1}===')
            print(cm[i])
            print('coef%s: %.4f' % (i+1, coef_ensem[i]))
            print('rmse%s: %.4f' % (i+1, rmse_ensem[i]))
            best_score = coef_ensem[0:config['num_tasks']] + rmse_ensem[0:config['num_tasks']]
    print("best_score",best_score)        
      
    return best_score, pred_model1_list, pred_model2_list, pred_ensem_list, label_list

def test(model, config, test_loader, best_score_list):
    col_list = pd.read_csv(config['check_path'] + config['all_label']).columns
    model = [[],[]]
    model_list = [[],[]]
    for K_n in range(config['fold_num']):
        if config['mutual'] == 'yes':
            mutual_num=2
        else:
            mutual_num=1
        for i in range(mutual_num):
            model[i].append(K_n)
            model[i][K_n] = archs.__dict__[config['arch']]()
            model[i][K_n] = model[i][K_n].cuda()
            model[i][K_n].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_n}_{i+1}.pth'))
            model_list[i].append(model[i][K_n])
            model_list[i][K_n].eval()
            
    pred_model1 = [[],[],[]]
    pred_model2 = [[],[],[]]
    labels = []
    label_list = [[],[],[]]
    preds = []
    pred_list = [[],[],[]]
    joint_num_list = []  
    ID_list = []
    coef = [0,0,0]
    rmse = [0,0,0]
    cm = [0,0,0]
    with torch.no_grad():
        for input, label, ID in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()    

            for i in range (config['num_tasks']):
                label_list[i].extend(label[i].to('cpu').detach().numpy().copy()) 
                
            if config['mutual'] == 'yes':
                pred_ensem = [0,0,0]
                for K in range(config['fold_num']): 
                    if config['num_tasks'] == 1:
                        pred_model1[0] = model_list[0][K](input)
                        pred_model2[0] = model_list[1][K](input)
                    
                    elif config['num_tasks'] == 2:
                        pred_model1[0], pred_model1[1] = model_list[0][K](input)
                        pred_model2[0], pred_model2[1]= model_list[1][K](input)                     
                        
                    elif config['num_tasks'] == 3:
                        pred_model1[0], pred_model1[1], pred_model1[2] = model_list[0][K](input)
                        pred_model2[0], pred_model2[1], pred_model2[2] = model_list[1][K](input)

                    for i in range (config['num_tasks']):
                        pred_ensem[i] = pred_ensem[i] + (pred_model1[i] + pred_model2[i])
                        
                for i in range (config['num_tasks']):
                    pred_ensem[i] = pred_ensem[i].max(1)[1].to('cpu').detach().numpy().copy()
                    pred_list[i].extend(pred_ensem[i])
                    
            else:
                pred = [0,0,0]
                for K in range(config['fold_num']): 
                    if config['num_tasks'] == 1:
                        pred_model1[0] = model_list[0][K](input)
                        pred_list1_1[K].extend(pred_model1[0].max(1)[1].to('cpu').detach().numpy().copy())
                    elif config['num_tasks'] == 2:
                        pred_model1[0], pred_model1[1] = model_list[0][K](input)
                        pred_list1_1[K].extend(pred_model1[0].max(1)[1].to('cpu').detach().numpy().copy())
                        pred_list1_2[K].extend(pred_model1[1].max(1)[1].to('cpu').detach().numpy().copy())
                    elif config['num_tasks'] == 3:
                        pred_model1[0], pred_model1[1], pred_model1[2] = model_list[0][K](input)     
                        pred_list1_1[K].extend(pred_model1[0].max(1)[1].to('cpu').detach().numpy().copy())
                        pred_list1_2[K].extend(pred_model1[1].max(1)[1].to('cpu').detach().numpy().copy())
                        pred_list1_3[K].extend(pred_model1[2].max(1)[1].to('cpu').detach().numpy().copy())
                        
                    #pred[0]にはcolumn1の予測値、pred[1]にはcolumn2の予測値がK-fold分蓄積されていく
                    for i in range (config['num_tasks']):
                        pred[i] = pred[i] + pred_model1[i]
                for i in range (config['num_tasks']):    
                    pred[i] = pred[i].max(1)[1].to('cpu').detach().numpy().copy()
                    pred_list[i].extend(pred[i])
            ID_list.extend(ID["img_id"])

        val_test='test'    
        for i in range (config['num_tasks']): 
            coef[i] = Pearson(pred_list[i], label_list[i])
            rmse[i] = RMSE(pred_list[i], label_list[i])
            cm[i] = confusion_matrix(pred_list[i],label_list[i])
            print("=========joint_name", col_list[i+1], "in test set===========")
            print(cm[i])
            print('coef%s: %.4f' % (i+1, coef[i]))
            print('rmse%s: %.4f' % (i+1, rmse[i]))
            cm_visualize(pred_list, label_list, config, val_test, i)

        for i in range (config['num_tasks']):
            labels.extend(label_list[i])
            preds.extend(pred_list[i])
            joint_num = np.ones(len(label_list[i]))*(i+1)
            joint_num_list.extend(joint_num)
            
        ID_list = ID_list*(config['num_tasks'])
        ID_list = pd.DataFrame(ID_list).set_axis(['img_id'], axis='columns', inplace=False)            
        label_list_df = pd.DataFrame(labels).set_axis(['label'], axis='columns', inplace=False)    
        joint_num_df = pd.DataFrame(joint_num_list).set_axis(['joint_num'], axis='columns', inplace=False)
        pred_list_df = pd.DataFrame(preds).set_axis(['pred'], axis='columns', inplace=False)

        DF_ensem=pd.concat([pred_list_df, label_list_df, ID_list, joint_num_df], axis=1)
        DF_ensem.to_csv(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/pred_label_ensem_test.csv', index=False)
        best_score_test = coef[0:config['num_tasks']] + rmse[0:config['num_tasks']]
            
    
    ##############pd.DataFrame(best_score_test)##############
    print("============All results===========")
    col_list_ext = col_list[1:config['num_tasks']+1]
    best_score_list = np.array(best_score_list)
    if config['mutual'] == 'yes':
        col_list_model1 = [f'{x}_model1' for x in col_list_ext]
        col_list_model2 = [f'{x}_model2' for x in col_list_ext]
        col_list_ensem = [f'{x}_ensem' for x in col_list_ext]
        col_list_all = col_list_model1 + col_list_model2 + col_list_ensem
        best_score_coef = best_score_list[0:,0:config['num_tasks']*3]
        best_score_rmse = best_score_list[0:,config['num_tasks']*3:config['num_tasks']*6]
    else:
        col_list_all = col_list_ext
        best_score_coef = best_score_list[0:,0:config['num_tasks']]
        best_score_rmse = best_score_list[0:,config['num_tasks']:config['num_tasks']*2]
    col_list_coef = [f'{x}_coef' for x in col_list_all]    
    col_list_rmse = [f'{x}_rmse' for x in col_list_all] 
    best_score_coef = pd.DataFrame(best_score_coef).set_axis(col_list_coef, axis='columns', inplace=False)
    best_score_rmse = pd.DataFrame(best_score_rmse).set_axis(col_list_rmse, axis='columns', inplace=False)
    best_score_coef_des = best_score_coef.describe()
    best_score_rmse_des = best_score_rmse.describe()
        
    print("============best score of coef in each cross validation===========")
    print(best_score_coef)
    print(best_score_coef_des)    
    print("============best score of rmse in each cross validation===========")
    print(best_score_rmse)
    print(best_score_rmse_des)      
        
    print("============best score of all K-folds ensemble model for test set===========")
    col_list_coef = [f'{x}_coef' for x in col_list_ext]    
    col_list_rmse = [f'{x}_rmse' for x in col_list_ext]  
    col_list = col_list_coef + col_list_rmse
    best_score_test = pd.DataFrame([best_score_test]).set_axis(col_list, axis='columns', inplace=False)
    print(best_score_test)
    
    best_score_val = pd.concat([best_score_coef, best_score_rmse],axis=1)
    best_score_val_des = best_score_val.describe()
    best_score_val.to_csv(config['check_path'] + 'damage_pred_log/models/%s/score_val.csv' % config['name'] , mode='a' )
    best_score_val_des.to_csv(config['check_path'] + 'damage_pred_log/models/%s/score_val.csv' % config['name'] , mode='a')

    best_score_test.to_csv(config['check_path'] + 'damage_pred_log/models/%s/score_test.csv' % config['name'] , mode='a' ,index=False, header=None)

    torch.cuda.empty_cache()  
    
def validate_bestmodel_restart(config, K_num, val_loader, model):
    col_list = pd.read_csv(config['check_path'] + config['all_label']).columns
    model[0].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_1.pth'))
    model[0].eval()
    if config['mutual'] == 'yes': 
        model[1].load_state_dict(torch.load(config['check_path'] + 'damage_pred_log/models/' + config['name'] + f'/model{K_num}_2.pth'))
        model[1].eval()
     
    ID_list = []
    labels = [0,0,0]
    label_list = [[],[],[]]
    joint_num_list =[]
    pred_model1_list = [[],[],[]]
    pred_model2_list = [[],[],[]]
    pred_ensem = [[],[],[]]
    pred_ensem_list = [[],[],[]]
    coef_model1 = [0,0,0]
    coef_model2 = [0,0,0]
    coef_ensem = [0,0,0]
    rmse_model1 = [0,0,0]
    rmse_model2 = [0,0,0]
    rmse_ensem = [0,0,0]
    cm = [0,0,0]
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))     
        for input, label, ID in val_loader:
            input = input.cuda()
            for i in range (config['num_tasks']):
                labels[i] = label[i].to('cpu').detach().numpy().copy()
                label_list[i].extend(labels[i])
            if config['mutual'] == 'yes': 
                pred_model1, pred_model2 = model_output(model, input, config)

                for i in range (config['num_tasks']):
                    pred_ensem[i] = (pred_model1[i] + pred_model2[i])/2
                    pred_model1_list[i].extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())
                    pred_model2_list[i].extend(pred_model2[i].max(1)[1].to('cpu').detach().numpy().copy())
                    pred_ensem_list[i].extend(pred_ensem[i].max(1)[1].to('cpu').detach().numpy().copy())
            else:
                pred_model1 = model_output(model, input, config)

                for i in range (config['num_tasks']):
                    pred_ensem_list[i].extend(pred_model1[i].max(1)[1].to('cpu').detach().numpy().copy())

            for i in range (config['num_tasks']):
                ID_ext = ID["img_id"]
                joint_num = np.ones(len(ID_ext))*(i+1)
                ID_list.extend(ID_ext)
                joint_num_list.extend(joint_num)
                
            pbar.update(1)
        pbar.close()

    for i in range (config['num_tasks']): 
        if config['mutual'] == 'yes':
            coef_model1[i] = Pearson(pred_model1_list[i], label_list[i])
            coef_model2[i] = Pearson(pred_model2_list[i], label_list[i])
            coef_ensem[i] = Pearson(pred_ensem_list[i], label_list[i])
            rmse_model1[i] = RMSE(pred_model1_list[i], label_list[i])
            rmse_model2[i] = RMSE(pred_model2_list[i], label_list[i])
            rmse_ensem[i] = RMSE(pred_ensem_list[i], label_list[i])
            cm[i] = confusion_matrix(pred_ensem_list[i],label_list[i])
            print("=========joint_name", col_list[i+1], "===========")
            print(f'===confusion matrix in ensem model label{i+1}===')
            print(cm[i])            
            print('model1', 'coef_model1_%s: %.4f' % (i+1, coef_model1[i]))
            print('model2', 'coef_model2_%s: %.4f' % (i+1, coef_model2[i]))
            print('ensem', 'coef_ensem_%s: %.4f' % (i+1, coef_ensem[i]))
            print('model1', 'rmse_model1_%s: %.4f' % (i+1, rmse_model1[i]))
            print('model2', 'rmse_model2_%s: %.4f' % (i+1, rmse_model2[i]))
            print('ensem', 'rmse_ensem_%s: %.4f' % (i+1, rmse_ensem[i]))
            best_score = coef_model1[0:config['num_tasks']] + coef_model2[0:config['num_tasks']] + coef_ensem[0:config['num_tasks']] + rmse_model1[0:config['num_tasks']] + rmse_model2[0:config['num_tasks']] + rmse_ensem[0:config['num_tasks']]
            
        else:
            coef_ensem[i] = Pearson(pred_ensem_list[i], label_list[i])
            rmse_ensem[i] = RMSE(pred_ensem_list[i], label_list[i])
            cm[i] = confusion_matrix(pred_ensem_list[i],label_list[i])
            print("=========joint_name", col_list[i+1], "===========")
            print(f'===confusion matrix label{i+1}===')
            print(cm[i])
            print('coef%s: %.4f' % (i+1, coef_ensem[i]))
            print('rmse%s: %.4f' % (i+1, rmse_ensem[i]))
            best_score = coef_ensem[0:config['num_tasks']] + rmse_ensem[0:config['num_tasks']]
    print("best_score",best_score)       
       
    return best_score, pred_model1_list, pred_model2_list, pred_ensem_list, label_list

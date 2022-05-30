from metrics import sdr, sdr_stratified
from utils import AverageMeter
from tqdm import tqdm
from collections import OrderedDict
import torch
import pandas as pd
import cv2
from matplotlib import pyplot as plt

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return postfix


def val(config, val_loader, model, criterion, scheduler, log, train_log, best_R, trigger, epoch):
    avg_meters = {'val_loss': AverageMeter(),'SDR': AverageMeter(),'R_ave': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            SDR, R_ave = sdr(output, target) 
                
            avg_meters['val_loss'].update(loss.item(), input.size(0))
            avg_meters['SDR'].update(SDR, input.size(0))
            avg_meters['R_ave'].update(R_ave.item(), input.size(0))       

            output = torch.sigmoid(output).cpu().numpy()
            target = torch.sigmoid(target).cpu().numpy()

            maskall = []
            postfix = OrderedDict([('val_loss', avg_meters['val_loss'].avg),('SDR', avg_meters['SDR'].avg),('R_ave', avg_meters['R_ave'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['val_loss'].append(postfix['val_loss'])
        log['SDR'].append(postfix['SDR'])
        log['R_ave'].append(postfix['R_ave'])
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(postfix['val_loss'])

        print('loss %.4f - val_loss %.4f - SDR %.4f - R_ave %.4f'
              % (train_log['loss'], postfix['val_loss'], postfix['SDR'], postfix['R_ave']))



        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1
   
        if postfix['R_ave'] < best_R:
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_R = postfix['R_ave']
            print("=> saved best model")
            trigger = 0        
        
    return postfix, best_R

def test(config, config2, test_loader, model):
    coord_col = pd.read_csv(config2['img_path'] + "/" + config2['coord_list'])
    coord_col_list = coord_col.columns
    with torch.no_grad():
        coord_data = []
        img_name = []
        for img, idx in  tqdm(test_loader):
            img=img.cuda()

            output = model(img)
            num = output.size(0)
            c = output.size(1)
            
            for j in range(num):
                one_data_coord_x = []
                one_data_coord_y = []
                for chan in range(c):
                    img = output[j, chan]
                    img_x, img_y = (torch.fmod(torch.argmax(img), img.size(1))).item(), (torch.argmax(img)/img.size(1)).item()

                    one_data_coord_x.append(int(img_x))
                    one_data_coord_y.append(int(img_y))
                if config2["visualize"]=="yes":
                    im_show = cv2.imread(config2['img_path'] + config2['data_dir'] + "/" + f'{idx[coord_col.columns[0]][j]}' + config2['img_ext'])#########
                    im_show = cv2.resize(im_show, (config['input_h'], config['input_w']))
                    for k in range(len(one_data_coord_x)):
                        cv2.circle(im_show, (one_data_coord_x[k], one_data_coord_y[k]), 3, (255, 255, 0), thickness=-1)
                    plt.imshow(im_show)
                    plt.show()
                    print("img_name", idx[coord_col.columns[0]][j])
                    print("=================================")
                one_data_coord = one_data_coord_x + one_data_coord_y
                coord_data.append(one_data_coord)
                img_name.append(f'{idx[coord_col.columns[0]][j]}' + config2['img_ext'])####################

        df1 = pd.DataFrame(coord_data)
        df2 = pd.DataFrame(img_name)


        df = pd.concat([df2, df1], axis=1).set_axis(coord_col_list, axis='columns')

        df.to_csv(config2['img_path'] + config2['data_dir'] + "/" + "coord_final.csv", index=False)
    torch.cuda.empty_cache()
    
def val_sdr_stratified(config, config2, val_loader, model):
    avg_meters = {'SDR_2': AverageMeter(),
                  'SDR_4': AverageMeter(),
                  'SDR_6': AverageMeter(),
                  'SDR_8': AverageMeter(),
                  'SDR_10': AverageMeter(),
                  'R_ave': AverageMeter()}
    
    log = OrderedDict([('SDR_2', []),('SDR_4', []),('SDR_6', []),('SDR_8', []),('SDR_10', []),('R_ave', [])])
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            SDR_2_all, SDR_4_all, SDR_6_all, SDR_8_all, SDR_10_all, R_ave = sdr_stratified(output, target)
                
            avg_meters['SDR_2'].update(SDR_2_all, input.size(0))
            avg_meters['SDR_4'].update(SDR_4_all, input.size(0))
            avg_meters['SDR_6'].update(SDR_6_all, input.size(0))
            avg_meters['SDR_8'].update(SDR_8_all, input.size(0))
            avg_meters['SDR_10'].update(SDR_10_all, input.size(0))
            avg_meters['R_ave'].update(R_ave.item(), input.size(0))       

            output = torch.sigmoid(output).cpu().numpy()
            target = torch.sigmoid(target).cpu().numpy()

            maskall = []
            postfix = OrderedDict([('SDR_2', avg_meters['SDR_2'].avg),
                                   ('SDR_4', avg_meters['SDR_4'].avg),
                                   ('SDR_6', avg_meters['SDR_6'].avg),
                                   ('SDR_8', avg_meters['SDR_8'].avg),
                                   ('SDR_10', avg_meters['SDR_10'].avg),
                                   ('R_ave', avg_meters['R_ave'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        log['SDR_2'].append(postfix['SDR_2'])
        log['SDR_4'].append(postfix['SDR_4'])
        log['SDR_6'].append(postfix['SDR_6'])
        log['SDR_8'].append(postfix['SDR_8'])
        log['SDR_10'].append(postfix['SDR_10'])
        log['R_ave'].append(postfix['R_ave'])
        
        print('SDR_2 %.4f - SDR_4 %.4f - SDR_6 %.4f - SDR_8 %.4f - SDR_10 %.4f -R_ave %.4f'
              % (postfix['SDR_2'], postfix['SDR_4'], postfix['SDR_6'], postfix['SDR_8'], postfix['SDR_10'],postfix['R_ave']))

        pd.DataFrame(log).to_csv('models/%s/log_sdr_stratify.csv' % config2['name'], index=False)
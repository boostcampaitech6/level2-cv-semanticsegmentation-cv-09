import datetime 
from tqdm.auto import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from omegaconf import OmegaConf
import argparse

from dataset import preprocessing, data_aug, XRayDataset
import models
from utils import dice_coef, set_seed, save_model

import warnings
warnings.filterwarnings('ignore')

scaler = GradScaler()
def train(model,file_name, epochs, data_loader, val_loader, criterion, optimizer,patience,device,save_dir):
    print(f'Start training..')
    
    n_class = 29
    best_dice = 0.
    best_loss = 9999
    i = 0
    for epoch in range(epochs):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.to(device), masks.to(device)
            
            with autocast():
                outputs = model(images)
                if not type(outputs) == torch.Tensor:
                    outputs = outputs['out']
                # loss를 계산합니다.
                loss = criterion(outputs, masks)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 10 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{step+1}/{len(data_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        
        dice, val_loss = validation(epoch + 1, model, val_loader, criterion, device)
        
        if best_loss > val_loss:
            print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f},  {best_loss:.4f} -> {val_loss:.4f}")
            print(f"Save model in {save_dir}")
            best_dice = dice
            best_loss = val_loss
            save_model(model,save_dir, file_name)
            i = 0
        else:
            i += 1
            
            if i == patience:
                print('Early Stopping')
                break

def validation(epoch, model, data_loader, criterion, device, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = 29
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.to(device), masks.to(device)         
    
            
            outputs = model(images)
            if not type(outputs) == torch.Tensor: 
                outputs = outputs['out']
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    # dice_str = [
    #     f"{c:<12}: {d.item():.4f}"
    #     for c, d in zip(CLASSES, dices_per_class)
    # ]
    # dice_str = "\n".join(dice_str)
    # print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    final_loss = total_loss / len(data_loader)
    return avg_dice, final_loss


def main(configs):
    IMAGE_ROOT = configs['dir']['img_dir']
    LABEL_ROOT = configs['dir']['label_dir']
    SAVE_DIR = configs['dir']['save_dir']

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    pngs, jsons = preprocessing(IMAGE_ROOT, LABEL_ROOT)
    transform = data_aug()
    trainset = XRayDataset(IMAGE_ROOT, LABEL_ROOT, pngs, jsons, True, transform)
    validset = XRayDataset(IMAGE_ROOT, LABEL_ROOT, pngs, jsons, False, transform)

    BATCH_SIZE = configs['batch_size']

    train_loader = DataLoader(
    dataset=trainset, 
    batch_size=BATCH_SIZE,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=validset, 
        batch_size=16,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    device = torch.device('cuda')
    model_name = configs['model_name']

    if model_name.lower() == 'FCN'.lower():
        model = models.FCN()
        model.to(device)
        file_name = 'FCN.pt'

    elif model_name.lower() == 'SegNet'.lower():
        model = models.SegNet()
        model.to(device)
        file_name = 'SegNet.pt'

    elif model_name.lower() == 'dilatednet':
        model = models.dilated_net()
        model.to(device)
        file_name = 'DilatedNet.pt'

    elif model_name.lower() == 'DeepLabv3+'.lower():
        model = models.DeepLabV3Plus()
        model.to(device)
        file_name = 'DeepLabv3Plus.pt'


    LR = configs['learning_rate']
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    RANDOM_SEED = configs['random_seed']
    set_seed(RANDOM_SEED)

    patience = configs['patience']
    epochs = configs['epochs']

    train(model, file_name, epochs, train_loader, valid_loader, criterion, optimizer, patience,device,SAVE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    main(configs)




import models
import torch 
from utils import encode_mask_to_rle, decode_rle_to_mask
from dataset import XRayInferenceDataset, data_aug_for_infer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from omegaconf import OmegaConf
import argparse
import os
from tqdm import tqdm
import pandas as pd 

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

IND2CLASS = {v: k for k, v in CLASS2IND.items()}

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = 29

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def main(configs):

    
    model_name = configs['model_name']
    if model_name == 'FCN':
        model = models.FCN()

    save_path = configs['path']['save_dir']
    model = torch.load(save_path)

    img_dir = configs['path']['img_dir']

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=img_dir)
        for root, _dirs, files in os.walk(img_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    tf = data_aug_for_infer()

    testset = XRayInferenceDataset(pngs, img_dir, tf)

    test_loader = DataLoader(
        dataset=testset, 
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    rles, filename_and_class = test(model, test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    output_dir = configs['path']['output_dir']
    df.to_csv(output_dir,index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/infer.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = OmegaConf.load(f)
    main(configs)
from datetime import datetime
import argparse
from omegaconf import OmegaConf

import lightning as L

from dataset import HandBoneDataModule
from model import HandBoneModel

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument(
        "--config", type=str, default="./configs/infer.yaml"
    )
args = parser.parse_args()
with open(args.config, 'r') as f:
    configs = OmegaConf.load(f)

L.seed_everything(configs.seed, workers=True)

now = datetime.now()
now = now.strftime('%m-%d_%H-%M-%S')

IMAGE_ROOT, save_dir = configs['dir']['img_dir'], configs['dir']['save_dir']
output_dir = configs['dir']['output_dir']
batch_size, img_size = configs.batch_size, configs.img_size
model_name, encoder = configs.model_name, configs.encoder_name


datamodule = HandBoneDataModule(
    IMAGE_ROOT=IMAGE_ROOT, batch_size=batch_size, img_size=img_size)
datamodule.setup(stage='predict')

model = HandBoneModel.load_from_checkpoint(
    f"./checkpoints/{save_dir}.ckpt", model_name=model_name,
    encoder=encoder, output_dir=output_dir)

trainer = L.Trainer()

trainer.predict(
    model,
    datamodule=datamodule
)

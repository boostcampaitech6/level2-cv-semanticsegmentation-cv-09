from datetime import datetime
import argparse
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from dataset import HandBoneDataModule
from model import HandBoneModel

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument(
        "--config", type=str, default="./configs/train.yaml"
    )
args = parser.parse_args()
with open(args.config, 'r') as f:
    configs = OmegaConf.load(f)

L.seed_everything(configs.seed, workers=True)
wandb_logger = WandbLogger(project="handbone_seg")

now = datetime.now()
now = now.strftime('%m-%d_%H-%M-%S')

checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{now}",
        save_top_k=3,
        every_n_epochs=1,
        monitor="val_dice",
        mode='max',
        filename="{epoch}_{val_loss:2f}_{val_dice:2f}"
)

IMAGE_ROOT, LABEL_ROOT = configs['dir']['img_dir'], configs['dir']['label_dir']
data_split, img_size = configs.split, configs.img_size
batch_size, lr = configs.batch_size, configs.learning_rate
model_name, encoder = configs.model_name, configs.encoder_name
patience = configs.patience

datamodule = HandBoneDataModule(
    IMAGE_ROOT=IMAGE_ROOT, LABEL_ROOT=LABEL_ROOT, data_split=data_split,
    batch_size=batch_size, img_size=img_size)
datamodule.setup(stage='fit')

model = HandBoneModel(
    model_name=model_name, encoder=encoder, scheduler=False, lr=lr)
early_stopping = EarlyStopping(
    monitor="val_dice", min_delta=0.00, patience=patience,
    verbose=False, mode="max")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    precision=16,
    max_epochs=configs.epochs,
    logger=wandb_logger,
    fast_dev_run=False
)

trainer.fit(
    model,
    datamodule=datamodule
)

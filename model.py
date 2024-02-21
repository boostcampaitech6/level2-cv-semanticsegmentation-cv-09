from typing import Union, Optional, Tuple
import os
import pandas as pd

import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightning as L
import segmentation_models_pytorch as smp
# from lion_pytorch import Lion

from models.upernet import UperNet_swin
from utils import dice_score, encode_mask_to_rle, calc_loss

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


class HandBoneModel(L.LightningModule):
    def __init__(
        self, model_name: str = 'Unet', encoder: str = 'resnet50', img_size: int = 512,
        scheduler: bool = False, lr: float = 0.001, output_dir: str = '',
        threshold: float = 0.5
    ) -> None:
        super().__init__()
        if model_name != 'SwinTransformerV2':
            self.model = smp.create_model(
                arch=model_name,
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=len(CLASSES),
            )
        else:
            self.model = UperNet_swin(
                size='swinv2_base_window16_256', img_size=img_size,
                num_classes=len(CLASSES))
        self.threshold = threshold
        self.scheduler = scheduler
        self.lr = lr
        self.output_dir = output_dir
        self.train_loss_outputs = []
        self.val_loss_outputs = []
        self.val_dice_outputs = []
        self.pred_rles = []
        self.fname_and_class = []
        # self.loss_fn = smp.losses.FocalLoss(mode='MULTICLASS_MODE')
        self.loss_fn = calc_loss

    def forward(self, x: Union[Tensor]) -> Union[Tensor]:
        return self.model(x)

    def training_step(self, batch: Union[Tensor], idx: int) -> torch.Tensor:
        image, mask = batch
        pred_mask = self(image)
        loss = self.loss_fn(pred_mask, mask)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, logger=True)
        self.train_loss_outputs.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = torch.stack(self.train_loss_outputs).mean()
        self.log(
            "train_loss_epoch", train_loss, on_epoch=True,
            prog_bar=True, logger=True)
        self.train_loss_outputs.clear()

    def validation_step(self, batch: Union[Tensor], idx: int) -> None:
        image, mask = batch
        pred_mask = self(image)
        loss = self.loss_fn(pred_mask, mask)

        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > self.threshold).detach().cpu()
        mask = mask.detach().cpu()

        dice = dice_score(pred_mask, mask)
        score = dice.mean()
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True,
            prog_bar=True, logger=True)
        self.log(
            "val_dice", score, on_step=True, on_epoch=True,
            prog_bar=True, logger=True)

        self.val_loss_outputs.append(loss)
        self.val_dice_outputs.append(dice)

    def on_validation_epoch_end(self) -> None:
        val_loss = torch.stack(self.val_loss_outputs).mean()
        dices = torch.cat(self.val_dice_outputs, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_dict = {}
        for cls, dice in zip(CLASSES, dices_per_class):
            dice_dict[cls] = dice.item()
        avg_dice = torch.mean(dices_per_class).item()
        self.log("val_loss_epoch", val_loss, on_epoch=True, logger=True)
        self.log("val_dice_epoch", avg_dice, on_epoch=True, logger=True)
        self.log_dict(dice_dict, on_epoch=True, logger=True)
        self.val_loss_outputs.clear()
        self.val_dice_outputs.clear()

    def predict_step(self, batch: Union[Tensor], idx: int) -> None:
        image, image_names = batch
        pred_mask = self(image)
        pred_mask = F.interpolate(
            pred_mask, size=(2048, 2048), mode="bilinear")
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > self.threshold).detach().cpu()

        for pred, image_name in zip(pred_mask, image_names):
            for c, seg in enumerate(pred):
                rle = encode_mask_to_rle(seg)
                self.pred_rles.append(rle)
                self.fname_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    def on_predict_end(self) -> None:
        classes, filename = zip(*[x.split("_") for x in self.fname_and_class])
        image_name = [os.path.basename(f) for f in filename]

        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": self.pred_rles,
        })
        df.to_csv(self.output_dir, index=False)

    def configure_optimizers(self) -> Optional[
            Union[Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler],
                  optim.Optimizer]]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=1, eta_min=0)
        if self.scheduler:
            return [optimizer], [scheduler]
        return optimizer

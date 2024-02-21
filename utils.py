from typing import Callable, Union
import numpy as np

import albumentations as A
# from albumentations.pytorch import ToTensorV2

import torch
from torch import Tensor
import torch.nn.functional as F


def dice_score(y_true: Union[Tensor], y_pred: Union[Tensor]) -> Union[Tensor]:
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


def encode_mask_to_rle(mask: np.ndarray) -> str:
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_train_aug(img_size: int = 512) -> Callable:
    train_transform = [
        A.HorizontalFlip(p=0.25),
        A.Rotate(30, p=0.3),
        A.ElasticTransform(p=0.3),
        # A.VerticalFlip(p=0.25),
        # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.25, border_mode=0),
        # A.OneOf(
        #     [
        #         A.RandomBrightnessContrast(p=0.25),
        #         A.HueSaturationValue(p=0.25),
        #     ],
        #     p=1,
        # ),
        # A.OneOf(
        #     [
        #         A.RandomBrightness(p=0.25, limit=0.35),
        #         A.ColorJitter(p=0.25),
        #     ],
        #     p=1,
        # ),
        # A.OneOf(
        #     [
        #         A.ChannelShuffle(p=0.25),
        #         A.RandomGamma(p=0.25),
        #     ],
        #     p=1,
        # ),
        A.Resize(img_size, img_size),
        # ToTensorV2(transpose_mask=True)
    ]
    return A.Compose(train_transform)


def get_valid_aug(img_size: int = 512) -> Callable:
    valid_transform = [
        A.Resize(img_size, img_size),
        # ToTensorV2(transpose_mask=True)
    ]
    return A.Compose(valid_transform)


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

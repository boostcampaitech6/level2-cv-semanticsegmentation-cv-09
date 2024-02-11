from typing import Callable, List, Tuple
import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from utils import get_train_aug, get_valid_aug


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


def preprocess(
    img_dir: os.PathLike = None, label_dir: os.PathLike = None,
    data_split: int = 0
) -> Tuple[List, List, List, List]:

    pngs = {
        os.path.relpath(os.path.join(root, fname), start=img_dir)
        for root, _dirs, files in os.walk(img_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=label_dir)
        for root, _dirs, files in os.walk(label_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = np.array(sorted(pngs))
    jsons = np.array(sorted(jsons))
    groups = [os.path.dirname(fname) for fname in pngs]
    ys = [0 for fname in pngs]

    groupkfold = GroupKFold(n_splits=5)
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []

    for i, (x, y) in enumerate(groupkfold.split(pngs, ys, groups)):
        if i == data_split:
            valid_images += list(pngs[y])
            valid_labels += list(jsons[y])
        else:
            train_images += list(pngs[y])
            train_labels += list(jsons[y])

    return train_images, train_labels, valid_images, valid_labels


def preprocess_predict(img_dir: os.PathLike = None) -> np.ndarray:
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=img_dir)
        for root, _dirs, files in os.walk(img_dir)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    }
    predict_images = np.array(sorted(pngs))
    return predict_images


class HandBoneDataset(Dataset):
    def __init__(
        self, img_dir: os.PathLike, label_dir: os.PathLike,
        images: List = None, labels: List = False,
        transform: Callable = None, is_train: bool = False
    ) -> None:
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = images
        self.labels = labels
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        image_path = os.path.join(self.img_dir, image_name)
        image = cv2.imread(image_path)
        image = image / 255.

        if self.is_train:
            label_name = self.labels[idx]
            label_path = os.path.join(self.label_dir, label_name)
            label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
            mask = np.zeros(label_shape, dtype=np.uint8)

            with open(label_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]

            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])

                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                mask[..., class_ind] = class_label

        if self.is_train:
            if self.transform is not None:
                inputs = {"image": image, "mask": mask}
                result = self.transform(**inputs)
                image = result["image"]
                mask = result["mask"]
            image = image.transpose(2, 0, 1)
            mask = mask.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()
            return image, mask
        else:
            inputs = {"image": image}
            result = self.transform(**inputs)
            image = result["image"]
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            return image, image_name


class HandBoneDataModule(L.LightningDataModule):
    def __init__(
        self, IMAGE_ROOT: os.PathLike = None, LABEL_ROOT: os.PathLike = None,
        data_split: int = 0, batch_size: int = 16, img_size: int = 512
    ) -> None:
        super().__init__()
        self.IMAGE_ROOT = IMAGE_ROOT
        self.LABEL_ROOT = LABEL_ROOT
        self.data_split = data_split
        self.batch_size = batch_size
        self.img_size = img_size

    def setup(self, stage: str = 'fit') -> None:
        self.num_workers = os.cpu_count() // 2
        if stage == 'fit':
            self.train_images, self.train_labels, self.valid_images, self.valid_labels = preprocess(
                img_dir=self.IMAGE_ROOT, label_dir=self.LABEL_ROOT,
                data_split=self.data_split)
            self.train_data = HandBoneDataset(
                img_dir=self.IMAGE_ROOT, label_dir=self.LABEL_ROOT,
                images=self.train_images, labels=self.train_labels,
                transform=get_train_aug(self.img_size), is_train=True)
            self.val_data = HandBoneDataset(
                img_dir=self.IMAGE_ROOT, label_dir=self.LABEL_ROOT,
                images=self.valid_images, labels=self.valid_labels,
                transform=get_valid_aug(self.img_size), is_train=True)
        elif stage == 'predict':
            self.predict_flist = preprocess_predict(img_dir=self.IMAGE_ROOT)
            self.predict_data = HandBoneDataset(
                img_dir=self.IMAGE_ROOT, label_dir=None,
                images=self.predict_flist, labels=None,
                transform=get_valid_aug(self.img_size), is_train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=True, drop_last=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.batch_size // 2,
            num_workers=self.num_workers, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_data, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False, drop_last=False)

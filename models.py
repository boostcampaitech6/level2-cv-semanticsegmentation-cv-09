import torch.nn as nn
from torchvision import models

LEN_CLASS = 29

def FCN():
    model = model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, LEN_CLASS, kernel_size=1)
    return model
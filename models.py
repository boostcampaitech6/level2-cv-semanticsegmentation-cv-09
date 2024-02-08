import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

LEN_CLASS = 29

def FCN():
    model = model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, LEN_CLASS, kernel_size=1)
    return model

class SegNet(nn.Module):
    def __init__(self, num_classes=LEN_CLASS):
        super(SegNet, self).__init__()
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        # conv1
        self.cbr1_1 = CBR(3, 64, 3, 1, 1)
        self.cbr1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv2
        self.cbr2_1 = CBR(64, 128, 3, 1, 1)
        self.cbr2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv3
        self.cbr3_1 = CBR(128, 256, 3, 1, 1)
        self.cbr3_2 = CBR(256, 256, 3, 1, 1)
        self.cbr3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv4
        self.cbr4_1 = CBR(256, 512, 3, 1, 1)
        self.cbr4_2 = CBR(512, 512, 3, 1, 1)
        self.cbr4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # conv5
        self.cbr5_1 = CBR(512, 512, 3, 1, 1)
        self.cbr5_2 = CBR(512, 512, 3, 1, 1)
        self.cbr5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True, return_indices=True)

        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr5_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr4_3 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_2 = CBR(512, 512, 3, 1, 1)
        self.dcbr4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr3_3 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_2 = CBR(256, 256, 3, 1, 1)
        self.dcbr3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr2_2 = CBR(128, 128, 3, 1, 1)
        self.dcbr2_1 = CBR(128, 64, 3, 1, 1)

        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dcbr1_2 = CBR(64, 64, 3, 1, 1)
        self.dcbr1_1 = CBR(64, 64, 3, 1, 1)
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size = 1)

    def forward(self, x):
        h = self.cbr1_1(x)
        h = self.cbr1_2(h)
        dim1 = h.size()
        h, pool1_indices = self.pool1(h)

        h = self.cbr2_1(h)
        h = self.cbr2_2(h)
        dim2 = h.size()
        h, pool2_indices = self.pool2(h)

        h = self.cbr3_1(h)
        h = self.cbr3_2(h)
        h = self.cbr3_3(h)
        dim3 = h.size()
        h, pool3_indices = self.pool3(h)

        h = self.cbr4_1(h)
        h = self.cbr4_2(h)
        h = self.cbr4_3(h)
        dim4 = h.size()
        h, pool4_indices = self.pool4(h)

        h = self.cbr5_1(h)
        h = self.cbr5_2(h)
        h = self.cbr5_3(h)
        dim5 = h.size()
        h, pool5_indices = self.pool5(h)

        h = self.unpool5(h, pool5_indices, output_size = dim5)
        h = self.dcbr5_3(h)
        h = self.dcbr5_2(h)
        h = self.dcbr5_1(h)

        h = self.unpool4(h, pool4_indices, output_size = dim4)
        h = self.dcbr4_3(h)
        h = self.dcbr4_2(h)
        h = self.dcbr4_1(h)

        h = self.unpool3(h, pool3_indices, output_size = dim3)
        h = self.dcbr3_3(h)
        h = self.dcbr3_2(h)
        h = self.dcbr3_1(h)

        h = self.unpool2(h, pool2_indices, output_size = dim2)
        h = self.dcbr2_2(h)
        h = self.dcbr2_1(h)

        h = self.unpool1(h, pool1_indices, output_size = dim1)
        h = self.dcbr1_2(h)
        h = self.dcbr1_1(h)
        h = self.score_fr(h)
        return h
    
    
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=False):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_ch, out_ch//4, 1, 1, 0),
            conv_block(out_ch//4, out_ch//4, 3, stride, dilation, dilation),
            conv_block(out_ch//4, out_ch, 1, 1, 0, relu=False)
        )
        self.downsample = nn.Sequential(
            conv_block(in_ch, out_ch, 1, stride, 0, 1, False)
        ) if downsample else None

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, dilation, num_layers):
        super().__init__()
        block = []
        for i in range(num_layers):
            block.append(Bottleneck(
                in_ch if i==0 else out_ch,
                out_ch,
                stride if i==0 else 1,
                dilation,
                True if i==0 else False
            ))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ResNet101(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_channels, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            ResBlock(64, 256, 1, 1, num_layers=3),
            ResBlock(256, 512, 2, 1, num_layers=4),
            ResBlock(512, 1024, 1, 2, num_layers=23),
            ResBlock(1024, 2048, 1, 4, num_layers=3)
        )

    def forward(self, x):
        return self.block(x)


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, out_ch, 1, 1, padding=0)
        self.block2 = conv_block(in_ch, out_ch, 3, 1, padding=6, dilation=6)
        self.block3 = conv_block(in_ch, out_ch, 3, 1, padding=12, dilation=12)
        self.block4 = conv_block(in_ch, out_ch, 3, 1, padding=18, dilation=18)
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5, size=upsample_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return out


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=LEN_CLASS):
        super().__init__()
        self.backbone = ResNet101(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048, 256)
        self.conv1 = conv_block(256*5, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        out = self.conv1(aspp_out)
        out = self.conv2(out)

        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out

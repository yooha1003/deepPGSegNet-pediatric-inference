"""Checkpoint-compatible four-level 3D U-Net for pituitary segmentation."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


def convolution(in_channels, out_channels):
    return nn.Sequential(OrderedDict([
        ('groupnorm', nn.GroupNorm(1 if in_channels < 8 else 8, in_channels)),
        ('conv', nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)),
        ('ReLU', nn.ReLU(inplace=True)),
    ]))


def double_convolution(in_channels, out_channels, encoder):
    middle = max(in_channels, out_channels // 2) if encoder else out_channels
    return nn.Sequential(OrderedDict([
        ('SingleConv1', convolution(in_channels, middle)),
        ('SingleConv2', convolution(middle, out_channels)),
    ]))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, pool):
        super().__init__()
        self.pooling = nn.MaxPool3d(2) if pool else nn.Identity()
        self.basic_module = double_convolution(in_channels, out_channels, True)

    def forward(self, x):
        return self.basic_module(self.pooling(x))


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.basic_module = double_convolution(in_channels, out_channels, False)

    def forward(self, skip, x):
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        return self.basic_module(torch.cat((skip, x), dim=1))


class UNet3D(nn.Module):
    """GroupNorm–Conv–ReLU blocks with widths 64, 128, 256 and 512."""
    def __init__(self):
        super().__init__()
        widths = [64, 128, 256, 512]
        self.encoders = nn.ModuleList([
            Encoder(1 if i == 0 else widths[i-1], w, i > 0)
            for i, w in enumerate(widths)
        ])
        self.decoders = nn.ModuleList([
            Decoder(widths[i] + widths[i-1], widths[i-1])
            for i in range(3, 0, -1)
        ])
        self.final_conv = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = decoder(skip, x)
        return self.final_conv(x)

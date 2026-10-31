import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleBUSI(nn.Module):

    def __init__(self, num_classes=3, pretrained=False, image_size=224):
        super().__init__()
        self.image_size = image_size

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        base = models.efficientnet_b0(weights=weights)
        self.encoder = base.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.lat8 = nn.Conv2d(1280, 128, 1)
        self.lat4 = nn.Conv2d(80, 128, 1)
        self.lat3 = nn.Conv2d(40, 128, 1)
        self.lat2 = nn.Conv2d(24, 128, 1)

        self.refine4 = ConvBlock(128, 128)
        self.refine3 = ConvBlock(128, 128)
        self.refine2 = ConvBlock(128, 128)

        self.seg_head = nn.Sequential(
            ConvBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):

        x = self.encoder[0](x)
        x = self.encoder[1](x)

        f2 = self.encoder[2](x)
        f3 = self.encoder[3](f2)
        f4 = self.encoder[4](f3)

        x = self.encoder[5](f4)
        x = self.encoder[6](x)
        x = self.encoder[7](x)
        f8 = self.encoder[8](x)

        pooled = self.pool(f8).flatten(1)
        class_logits = self.classifier(pooled)

        p8 = self.lat8(f8)

        p4 = self.lat4(f4) + F.interpolate(
            p8,
            size=f4.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        p4 = self.refine4(p4)

        p3 = self.lat3(f3) + F.interpolate(
            p4,
            size=f3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        p3 = self.refine3(p3)

        p2 = self.lat2(f2) + F.interpolate(
            p3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        p2 = self.refine2(p2)

        seg_features = F.interpolate(
            p2,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        seg_logits = self.seg_head(seg_features)

        return class_logits, seg_logits, f8


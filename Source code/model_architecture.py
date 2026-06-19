import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNetHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.transformer_bridge = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )

        self.dec1 = DecoderBlock(512, 1024, 256)
        self.dec2 = DecoderBlock(256, 512, 128)
        self.dec3 = DecoderBlock(128, 256, 64)
        self.dec4 = DecoderBlock(64, 0, 32)

        self.final_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.encoder1(x)
        c2 = self.encoder2(c1)
        c3 = self.encoder3(c2)
        c4 = self.encoder4(c3)
        c5 = self.encoder5(c4)

        bottleneck = self.transformer_bridge(c5)

        x = self.dec1(bottleneck, c4)
        x = self.dec2(x, c3)
        x = self.dec3(x, c2)
        x = self.dec4(x)

        out = self.final_head(x)
        out = nn.functional.interpolate(out, size=(512, 512), mode='bilinear', align_corners=True)

        return out

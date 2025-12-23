import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet34, ResNet34_Weights
    _HAS_TV = True
except Exception:
    _HAS_TV = False


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv1 = ConvBNReLU(c_in + c_skip, c_out, 3, 1, 1)
        self.conv2 = ConvBNReLU(c_out, c_out, 3, 1, 1)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetUNetGraspNoWidth(nn.Module):
    """
    ResNet34 encoder + UNet-like decoder
    Outputs ONLY:
      pos_logits, cos2, sin2
    """
    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        if not _HAS_TV:
            raise RuntimeError("torchvision not available. Install torchvision first.")

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        enc = resnet34(weights=weights)

        old_conv = enc.conv1
        enc.conv1 = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # init first conv weights if pretrained
        if pretrained:
            with torch.no_grad():
                if in_channels == 3:
                    enc.conv1.weight.copy_(old_conv.weight)
                elif in_channels > 3:
                    enc.conv1.weight[:, :3].copy_(old_conv.weight)
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                    for c in range(3, in_channels):
                        enc.conv1.weight[:, c:c+1].copy_(mean_w)
                else:
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    enc.conv1.weight.copy_(mean_w)

        self.enc = enc

        # encoder stages
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2
        self.maxpool = enc.maxpool                               # /4
        self.e1 = enc.layer1                                     # /4
        self.e2 = enc.layer2                                     # /8
        self.e3 = enc.layer3                                     # /16
        self.e4 = enc.layer4                                     # /32

        # decoder
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 64, 64)

        self.final_conv = nn.Sequential(
            ConvBNReLU(64, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 1)
        )

        # heads
        self.head_pos = nn.Conv2d(64, 1, 1)  # logits
        self.head_cos = nn.Conv2d(64, 1, 1)  # tanh
        self.head_sin = nn.Conv2d(64, 1, 1)  # tanh

    def forward(self, x):
        s0 = self.stem(x)          # /2
        x1 = self.maxpool(s0)      # /4
        s1 = self.e1(x1)           # /4
        s2 = self.e2(s1)           # /8
        s3 = self.e3(s2)           # /16
        s4 = self.e4(s3)           # /32

        d3 = self.up3(s4, s3)      # /16
        d2 = self.up2(d3, s2)      # /8
        d1 = self.up1(d2, s1)      # /4
        d0 = self.up0(d1, s0)      # /2

        d = F.interpolate(d0, scale_factor=2.0, mode="bilinear", align_corners=False)  # /1
        d = self.final_conv(d)

        pos_logits = self.head_pos(d)
        cos2 = torch.tanh(self.head_cos(d))
        sin2 = torch.tanh(self.head_sin(d))
        return pos_logits, cos2, sin2

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv_downsample = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.residual_connection(x) + self.conv_downsample(x)


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

        self.conv_upsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.residual_connection(x) + self.conv_upsample(x)


class UNetResBlockUp(nn.Module):
    def __init__(self, in_channels, skip_connection_channels, out_channels):
        super().__init__()

        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

        self.upsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_connection_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, skip_connection):
        upsampled = self.upsample(x)
        upsampled = torch.cat([upsampled, skip_connection], dim=1)
        return self.residual_connection(x) + self.conv(upsampled)

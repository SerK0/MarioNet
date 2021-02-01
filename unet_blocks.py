import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Basic U-Net blocks implementation.
Implementation is based on the original paper, but differs in some details:
- convolutions use padding to keep image dimensions intact
- up-conv operation uses 3x3 padded convolution as opposed to 2x2 in original paper
- batch norm is used between convolutions in conv block
First two differences were dictated by the ease of TargetEncoder (U-Net based architecture)
  implementation and the 3rd one is supposed to improve performance.
"""


class DoubleConvBlock(nn.Module):
    """
    Common to downsampling and upsampling blocks part.
    Consists of two consecutive convolutions separated by ReLU (original paper) and
      batch norm (custom modification).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """
    Custom U-Net upsampling operation.
    Consists of bilinear upsampling followed by 3x3 (in original paper 2x2) convolution
      that halves the number of feature channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # TODO(binpord): will 1x1 conv be better?
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block. Quote from original paper:
    'It consists of the repeated application of two 3x3 convolutions (unpadded convolutions),
      each followed by a rectified linear unit (ReLU)[here should probably be a comma] and
      a 2x2 max pooling operation with stride 2 for downsampling.'
    Current implementation uses padded convolutions and adds batch norm after each convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        return self.downsample(x)


class UpBlock(nn.Module):
    """
    Upsampling block. Quote from original paper:
    'Every step in the expansive path consists of an upsampling of the feature map followed
      by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a
      concatenation with the correspondingly cropped feature map from the contracting path,
      and two 3x3 convolutions, each followed by a ReLU.'
    Current implementation uses padded convolutions and adds batch norm after each convolution.
    Also, up-conv module uses padded 3x3 convolution for the ease of implementation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = UpConv(in_channels, out_channels=in_channels // 2)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

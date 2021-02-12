import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MarioNetModule, pairwise, warp_image
from unet_blocks import DownBlock, UpBlock


class TargetEncoder(MarioNetModule):
    """
    Target encoder. Quote from original paper:
    'The target encoder Ey(y, ry) adopts a U-Net architecture to extract style information
      from the target input and generates target feature map zy along with the warped target
      feature maps S.'
    """

    def __init__(self, config):
        super(TargetEncoder, self).__init__(config)
        self.input_conv = nn.Conv2d(
            self.config.image_channels + self.config.landmark_channels,
            out_channels=self.config.downsampling_channels[0],
            kernel_size=3,
            padding=1,
        )

        # '...adopts a U-Net style architecture including five downsampling blocks
        #   and four upsampling blocks with skip connections'
        assert len(self.config.downsampling_channels) == 6
        downsampling_blocks = []
        for in_channels, out_channels in pairwise(self.config.downsampling_channels):
            downsampling_blocks.append(DownBlock(in_channels, out_channels))

        self.downsampling_blocks = nn.ModuleList(downsampling_blocks)

        assert len(self.config.upsampling_channels) == 5
        upsampling_blocks = []
        for in_channels, out_channels in pairwise(self.config.upsampling_channels):
            upsampling_blocks.append(UpBlock(in_channels, out_channels))

        self.upsampling_blocks = nn.ModuleList(upsampling_blocks)

        # TODO(binpord): will 1x1 conv be better?
        self.output_conv = nn.Conv2d(
            self.config.upsampling_channels[-1],
            out_channels=2,
            kernel_size=3,
            padding=1,
        )

    def forward(self, target_image, landmark_image):
        x = torch.cat([target_image, landmark_image], dim=1)
        x = F.relu(self.input_conv(x))

        feature_maps = []
        for block in self.downsampling_blocks:
            x = block(x)
            feature_maps.append(x)

        for i, block in enumerate(self.upsampling_blocks):
            x = block(x, feature_maps[-2 - i])

        optical_flow = torch.tanh(self.output_conv(x))
        *s, zy = feature_maps
        s = [warp_image(image, optical_flow) for image in s]
        return s, zy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MarioNetModule, pairwise
from unet_blocks import DownBlock, UpBlock


class TargetEncoder(MarioNetModule):
    """
    Target encoder. Quote from original paper:
    'The target encoder Ey(y, ry) adopts a U-Net architecture to extract style information
      from the target input and generates target feature map zy along with the warped target
      feature maps S.'
    """

    def __init__(self, config):
        super().__init__(config)
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

    def warp_image(self, image, optical_flow):
        """
        Warp image according to optical flow map.
        Heavily influenced by https://github.com/AliaksandrSiarohin/monkey-net/blob/master/modules/generator.py#L51
        """
        _, _, flow_h, flow_w = optical_flow.size()
        _, _, image_h, image_w = image.size()
        # TODO(binpord): MarioNETte authors use average pooling instead of nearest interpolation
        #   as opposed to the referenced paper.
        optical_flow = F.interpolate(
            optical_flow, size=(image_h, image_w), mode="nearest"
        )
        optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(image, optical_flow)

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
        s = [self.warp_image(image, optical_flow) for image in s]
        return s, zy

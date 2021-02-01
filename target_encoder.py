import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_blocks import DownBlock, UpBlock


class TargetEncoder(nn.Module):
    """
    Target encoder. Quote from original paper:
    'The target encoder Ey(y, ry) adopts a U-Net architecture to extract style information
      from the target input and generates target feature map zy along with the warped target
      feature maps S.'
    """

    def __init__(self, image_channels=3, landmark_channels=2):
        super().__init__()
        self.input_conv = nn.Conv2d(
            image_channels + landmark_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )

        # '...adopts a U-Net style architecture including five downsampling blocks
        #   and four upsampling blocks with skip connections'
        self.downsampling_blocks = nn.ModuleList(
            [
                DownBlock(64, 128),
                DownBlock(128, 256),
                DownBlock(256, 512),
                DownBlock(512, 1024),
                DownBlock(1024, 2048),
            ]
        )

        self.upsampling_blocks = nn.ModuleList(
            [
                UpBlock(2048, 1024),
                UpBlock(1024, 512),
                UpBlock(512, 256),
                UpBlock(256, 128),
            ]
        )

        # TODO(binpord): will 1x1 conv be better?
        self.output_conv = nn.Conv2d(128, out_channels=2, kernel_size=3, padding=1)

    def warp_image(self, image, optical_flow):
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

        optical_flow = F.tanh(self.output_conv(x))
        *s, zy = feature_maps
        s = [self.warp_image(image, optical_flow) for image in s]
        return s, zy

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

from .common.utils import pairwise, warp_image
from .common.warp_alignment import WarpAlignmentBlock
from .common.blocks import (
    ResBlockDown,
    ResBlockUp,
    UNetResBlockUp,
    DecoderBlock,
    SelfAttentionBlock,
)


class MarioNetModule(nn.Module):
    """
    Base class for all MarioNet modules.

    Parses config and sets self.config to be config.model.__class__name__ on __init__.
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(MarioNetModule, self).__init__()
        self.config = config["model"][self.__class__.__name__]


class TargetEncoder(MarioNetModule):
    """
    Target encoder. Quote from original paper:
    'The target encoder Ey(y, ry) adopts a U-Net architecture to extract style information
      from the target input and generates target feature map zy along with the warped target
      feature maps S.'
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
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
            downsampling_blocks.append(ResBlockDown(in_channels, out_channels))

        self.downsampling_blocks = nn.ModuleList(downsampling_blocks)

        assert len(self.config.upsampling_channels) == 5
        upsampling_blocks = []
        channels = zip(
            self.config.upsampling_channels[:-1],
            self.config.downsampling_channels[-2:0:-1],
            self.config.upsampling_channels[1:],
        )
        for in_channels, skip_connection_channels, out_channels in channels:
            upsampling_blocks.append(
                UNetResBlockUp(in_channels, skip_connection_channels, out_channels)
            )

        self.upsampling_blocks = nn.ModuleList(upsampling_blocks)

        # TODO(binpord): will 1x1 conv be better?
        self.output_conv = nn.Conv2d(
            self.config.upsampling_channels[-1],
            out_channels=2,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, target_image: torch.Tensor, landmark_image: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Forward pass of TargetEncoder.

        Parses target image and its landmarks to produce output feature maps and optical flow
        for warp alignment.

        Normally parses single target image at a time, however, can be tricked with temporal
        increase of batch size.

        :param torch.Tensor target_image: target image, shape [B, I, W, H]
        :param torch.Tensor landmark_image: target image landmarks, shape [B, L, W, H]
        :returns:
          - list[torch.Tensor] S - feature maps to be used in decoder
          - torch.Tensor zy - TargetEncoder output feature map
        :rtype: tuple[list[torch.Tensor], torch.Tensor]

        Here B - batch size, I - image_channels, L - landmark_channels, W - width, H - height.
        """
        x = torch.cat([target_image, landmark_image], dim=1)
        x = F.relu(self.input_conv(x))

        feature_maps = []
        for block in self.downsampling_blocks:
            x = block(x)
            feature_maps.append(x)

        for block, skip_connection in zip(self.upsampling_blocks, feature_maps[-2::-1]):
            x = block(x, skip_connection)

        optical_flow = torch.tanh(self.output_conv(x))
        *unwarped_s, zy = feature_maps
        reversed_s = []
        for feature_map in reversed(unwarped_s):
            reversed_s.append(warp_image(feature_map, optical_flow))
            optical_flow = F.avg_pool2d(optical_flow, kernel_size=2)

        return list(reversed(reversed_s)), zy


class Decoder(MarioNetModule):
    """
    Decoder. Quote from the paper:
    'The decoder Q(zxy, Si (i=1...K)) consists of four warp-alignment blocks followed by residual
      upsampling blocks. Note that the last upsampling block is followed by an additional
      convolution layer and a hyperbolic tangent activation function.'
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(Decoder, self).__init__(config)

        # feature maps are target encoder downsampling path outputs from all layers except last
        feature_map_channels = config.model.TargetEncoder.downsampling_channels[1:-1]
        assert len(feature_map_channels) == 4
        assert len(self.config.channels) == 5
        channels = zip(
            self.config.channels[:-1], feature_map_channels, self.config.channels[1:]
        )

        decoder_blocks = []
        for in_channels, feature_map_channels, out_channels in channels:
            decoder_blocks.append(
                DecoderBlock(in_channels, feature_map_channels, out_channels)
            )

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_conv = nn.Conv2d(
            in_channels=self.config.channels[-1],
            out_channels=self.config.output_channels,
            kernel_size=1,
        )

    def forward(
        self,
        blender_output: torch.Tensor,
        target_encoder_feature_maps: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Decoder forward pass.

        :param torch.Tensor blender_output: Blender output
        :param list[torch.Tensor] target_encoder_feature_maps: TargetEncoder feature maps (S in doc)
        :returns: decoder output
        :rtype: torch.Tensor
        """
        x = blender_output
        for block, feature_map in zip(
            self.decoder_blocks, reversed(target_encoder_feature_maps)
        ):
            x = block(x, feature_map)

        return torch.tanh(self.output_conv(x))


class DriverEncoder(MarioNetModule):
    def __init__(self, config):
        """
        Downsample Encoder of input driver image
        """
        super(DriverEncoder, self).__init__(config)

        input_feature_dim = self.config["input_feature_dim"]
        hidden_features_dim = self.config["hidden_features_dim"]

        assert self.config["depth"] == len(
            self.config["hidden_features_dim"]
        ), "inconsistent depth of driver encoder and len of hidden dims"

        self.block1 = ResBlockDown(input_feature_dim, hidden_features_dim[0])

        self.blocks = nn.Sequential(
            ResBlockDown(
                hidden_features_dim[idx],
                hidden_features_dim[idx + 1],
            )
            for idx, hidden_dim in enumerate(hidden_features_dim[:-1])
        )

    def forward(self, rx):
        x = self.block1(rx)
        x = self.blocks(x)
        return x


class Blender(MarioNetModule):
    def __init__(self, config):
        super(Blender, self).__init__(config)

        self.self_attnblock = SelfAttentionBlock(
            self.config["driver_feature_dim"],
            self.config["target_feature_dim"],
            self.config["attention_feature_dim"],
        )

        self.inst_norm1 = nn.InstanceNorm2d(self.config["driver_feature_dim"])

        self.conv = nn.Conv2d(
            self.config["driver_feature_dim"],
            self.config["driver_feature_dim"],
            kernel_size=3,
            padding=1,
        )

        self.inst_norm2 = nn.InstanceNorm2d(self.config["driver_feature_dim"])

    def forward(self, zx, Zy):
        mixed_feature = self.self_attnblock(zx, Zy)
        normed = self.inst_norm1(mixed_feature)
        return self.inst_norm2(normed + self.conv(normed))

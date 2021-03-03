import torch
import torch.nn as nn
import torch.nn.functional as F

import typing as tp

from ..config import Config

from .common.utils import pairwise, warp_image
from .common.blocks import (
    ResBlockDown,
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


class ConvMerger(MarioNetModule):
    """
    Merges image and landmark and projects them into intermediate tensor representation.
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(ConvMerger, self).__init__(config)
        self.conv_projection = nn.Conv2d(
            in_channels=self.config.image_channels + self.config.landmark_channels,
            out_channels=self.config.tensor_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, image: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor image: image
        :param torch.Tenosr landmarks: landmarks
        :returns: intermediate representation
        :rtype: torch.Tensor
        """
        merged = torch.cat([image, landmarks], dim=1)
        return self.conv_projection(merged)


class DriverEncoder(MarioNetModule):
    """
    MarioNet DriverEncoder - consist of five residual downsampling blocks
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(DriverEncoder, self).__init__(config)

        assert len(self.config.hidden_features_dim) == 5
        blocks = []
        for in_channels, out_channels in pairwise(self.config.hidden_features_dim):
            blocks.append(ResBlockDown(in_channels, out_channels))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, driver_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor driver_tensor: driver tensor, shape [B, C, W, H]
        :rtype: torch.Tensor

        Here B - batch size, C - tensor channels, W/H - image width/height.
        """
        return self.blocks(driver_tensor)


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
        self,
        target_tensor: torch.Tensor,
    ) -> tp.Tuple[tp.List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of TargetEncoder.

        Parses target image and its landmarks to produce output feature maps and optical flow
        for warp alignment.

        Normally parses single target image at a time, however, can be tricked with temporal
        increase of batch size.

        :param torch.Tensor target_tensor: target tensor, shape [B, C, W, H]
        :returns:
          - list[torch.Tensor] S - feature maps to be used in decoder
          - torch.Tensor zy - TargetEncoder output feature map
        :rtype: tuple[list[torch.Tensor], torch.Tensor]

        Here B - batch size, C - tensor channels, W - width, H - height.
        Target tensor is the product of convolutional layer from concatenated target image and
        landmarks.
        """

        feature_maps = []
        for block in self.downsampling_blocks:
            target_tensor = block(target_tensor)
            feature_maps.append(target_tensor)

        for block, skip_connection in zip(self.upsampling_blocks, feature_maps[-2::-1]):
            target_tensor = block(target_tensor, skip_connection)

        optical_flow = torch.tanh(self.output_conv(target_tensor))
        *unwarped_s, zy = feature_maps
        reversed_s = []
        for feature_map in reversed(unwarped_s):
            reversed_s.append(warp_image(feature_map, optical_flow))
            optical_flow = F.avg_pool2d(optical_flow, kernel_size=2)

        return list(reversed(reversed_s)), zy


class Blender(MarioNetModule):
    """
    Blender: mixes driver and target feature maps
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(Blender, self).__init__(config)

        self.self_attnblock = SelfAttentionBlock(
            self.config.driver_feature_dim,
            self.config.target_feature_dim,
            self.config.attention_feature_dim,
        )

        self.inst_norm1 = nn.InstanceNorm2d(self.config.driver_feature_dim)

        self.conv = nn.Conv2d(
            self.config.driver_feature_dim,
            self.config.driver_feature_dim,
            kernel_size=3,
            padding=1,
        )

        self.inst_norm2 = nn.InstanceNorm2d(self.config.driver_feature_dim)

    def forward(self, zx: torch.Tensor, Zy: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor zx: driver encoder output
        :param torch.Tensor zy: target encoder output
        :return: mixed feature map with size equal to zx.size()
        :rtype: torch.Tensor
        """
        mixed_feature = self.self_attnblock(zx, Zy)
        normed = self.inst_norm1(mixed_feature)
        return self.inst_norm2(normed + self.conv(normed))


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
            self.config.channels[:-1],
            feature_map_channels,
            self.config.channels[1:],
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
        target_encoder_feature_maps: tp.List[torch.Tensor],
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
            self.decoder_blocks,
            target_encoder_feature_maps,
        ):
            x = block(x, feature_map)

        return torch.tanh(self.output_conv(x))

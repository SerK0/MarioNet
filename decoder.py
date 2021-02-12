import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MarioNetModule, pairwise, warp_image


class WarpAlignmentBlock(nn.Module):
    """
    'To adapt pose-normalized feature maps to the pose of the driver, we generate an estimated
      flow map of the driver fu using 1x1 convolution that takes u as the input. Alignment by
      T(Sj; fu) follows. Then, the result is concatenated to u and fed into the following
      residual upsampling block.'
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=2, kernel_size=1)

    def forward(self, x, feature_map):
        optical_flow = torch.tanh(self.conv(x))
        return warp_image(feature_map, optical_flow)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, feature_map_channels, out_channels):
        super().__init__()
        self.warp_alignment = WarpAlignmentBlock(in_channels)
        self.conv = nn.Conv2d(
            in_channels + feature_map_channels, out_channels, kernel_size=1
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x, feature_map):
        warp_aligned_feature_map = self.warp_alignment(x, feature_map)
        x = torch.cat([x, warp_aligned_feature_map], dim=1)
        x = F.relu(self.conv(x))
        return self.upsample(x)


class Decoder(MarioNetModule):
    """
    Decoder. Quote from the paper:
    'The decoder Q(zxy, Si (i=1...K)) consists of four warp-alignment blocks followed by residual
      upsampling blocks. Note that the last upsampling block is followed by an additional
      convolution layer and a hyperbolic tangent activation function.'
    """

    def __init__(self, config):
        super().__init__(config)

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

    def forward(self, blender_output, target_encoder_feature_maps):
        x = blender_output
        for block, feature_map in zip(
            self.decoder_blocks, reversed(target_encoder_feature_maps)
        ):
            x = block(x, feature_map)

        return torch.tanh(self.output_conv(x))

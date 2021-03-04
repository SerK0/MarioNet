import torch
import torch.nn as nn

from .common.blocks import ResBlockDown
from .common.conv_merger import ConvMerger
from .common.utils import pairwise
from ..config import Config

import typing as tp


class Discriminator(nn.Module):
    """
    MarioNET Discriminator
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(Discriminator, self).__init__()
        self.config = config.model.Discriminator

        self.conv_merger = ConvMerger(
            self.config.image_channels,
            self.config.landmarks_channels,
            self.config.channels[0],
        )

        self.blocks = nn.ModuleList(
            [
                ResBlockDown(in_channels, out_channels)
                for in_channels, out_channels in pairwise(self.config.channels)
            ]
        )

        self.output_conv = nn.Conv2d(
            in_channels=self.config.channels[-1],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, image: torch.Tensor, landmarks: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Discriminator forward pass

        :param torch.Tensor image: Generator (or Real) image
        :param torch.Tensor landmarks: Target image landmarks
        :return:
            - PatchGAN like output of Discriminator: torch.Tensor
            - Intermediate features of Discriminator: list[torch.Tensor]
        """

        landmarks = torch.squeeze(landmarks, dim=1)

        x = self.conv_merger(image, landmarks)

        intermediate_features = []
        for block in self.blocks:
            x = block(x)
            intermediate_features.append(x)

        return torch.sigmoid(self.output_conv(x)), intermediate_features

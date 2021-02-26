import torch
import torch.nn as nn

from .common.blocks import ResBlockDown
from .config.config import Config

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

        self.config = config["model"][self.__class__.__name__]

        modules = [
            ResBlockDown(self.config.channels[0], self.config.channels[1]),
            ResBlockDown(self.config.channels[1], self.config.channels[2]),
            ResBlockDown(self.config.channels[2], self.config.channels[3]),
            ResBlockDown(self.config.channels[3], self.config.channels[4]),
        ]

        modules += [nn.Conv2d(self.config.channels[-1], 1, 3, 1)]

        self.model = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Discriminator forward pass

        :param x: Generator (or Real) image projected to 64 dim space
        :return:
            - PatchGAN like output of Discriminator: torch.Tensor
            - Intermediate features of Discriminator: list[torch.Tensor]
        """
        intermediate_features = []

        for module in self.model:
            x = module(x)
            intermediate_features.append(x)

        return torch.sigmoid(x), intermediate_features

import typing as tp

import torch


class GeneratorHingeLoss:
    """
    Generator part of hinge GAN loss.
    """

    def __init__(
        self, discriminator: tp.Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        :param tp.Callable[[torch.Tensor], torch.Tensor] discriminator: discriminator
        :returns: None
        """
        self.discriminator = discriminator

    def __call__(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor output: generator output
        :returns: hinge loss
        :rtype: torch.Tensor
        """
        output_realness, _ = self.discriminator(output_tensor)
        output_realness = output_realness[:, 0]
        return -torch.mean(output_realness, dim=0)

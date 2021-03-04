import torch


class GeneratorHingeLoss:
    """
    Generator part of hinge GAN loss.
    """

    def __call__(self, output_realness: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor output_realness: discriminator output
        :returns: hinge loss
        :rtype: torch.Tensor
        """
        return -torch.mean(output_realness)

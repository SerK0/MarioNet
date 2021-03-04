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
        output_realness = output_realness[:, 0]
        return -torch.mean(output_realness, dim=0)

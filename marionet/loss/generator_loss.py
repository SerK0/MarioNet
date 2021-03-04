import torch

from marionet.model import Discriminator

from .perceptual_loss import (
    PerceptualLossVGG19,
    PerceptualLossVGG_VD_16,
    PerceptualLoss,
)
from .feature_map_loss import FeatureMapLoss
from .generator_hinge_loss import GeneratorHingeLoss


class GeneratorLoss:
    """
    Overall generator loss.
    Consists of generator part of GAN hinge loss, perceptual loss with VGG19 and VGG_VD_16
    and feature map difference loss for discriminator feature maps.
    """

    def __init__(
        self,
        discriminator: Discriminator,
        lambda_p: float = 0.01,
        lambda_pf: float = 0.01,
        lambda_fm: float = 10.0,
    ):
        """
        :param tp.Callable[[torch.Tensor], torch.Tensor] discriminator: discriminator
        :param float lambda_p: coefficient for perceptual loss with VGG19
        :param float lambda_fp: coefficient for perceptual loss with VGG_VD_16
        :param float lambda_fm: coefficient for discriminator feature map loss
        :returns: None
        """
        self.discriminator = discriminator

        self.gan_loss = GeneratorHingeLoss()

        self.vgg19_loss = PerceptualLoss(PerceptualLossVGG19())
        self.lambda_p = lambda_p

        self.vgg_vd_16_loss = PerceptualLoss(PerceptualLossVGG_VD_16())
        self.lambda_pf = lambda_pf

        self.feature_map_loss = FeatureMapLoss()
        self.lambda_fm = lambda_fm

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, target_landmarks: torch.Tensor
    ) -> torch.Tensor:
        """
        :param torch.Tensor output: generator output
        :param torch.Tensor target: target image
        :returns: overall generator loss
        :rtype: torch.Tensor
        """
        output_realness, output_feature_maps = self.discriminator(
            output, target_landmarks
        )
        _, target_feature_maps = self.discriminator(target, target_landmarks)
        return (
            self.gan_loss(output_realness)
            + (self.lambda_p * self.vgg19_loss(output, target))
            + (self.lambda_pf * self.vgg_vd_16_loss(output, target))
            + (
                self.lambda_fm
                * self.feature_map_loss(output_feature_maps, target_feature_maps)
            )
        )

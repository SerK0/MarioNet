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
        self,
        reenacted_image: torch.Tensor,
        driver_image: torch.Tensor,
        driver_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Overall generator loss. It is being computed from reenacted image
        (i.e. generator output) and ground-truth reenacted image and its
        landmarks (to ease discriminator's job).

        'Since the paired target and the driver images from different
         identities cannot be acquired without explicit annotation,
         we trained our model using the target and the driver image
         extracted from the same video.'

        Due to this fact, actual driver image is being used as the
        ground-truth image, hence the names of parameters.

        :param torch.Tensor reenacted_image: reenacted image
        :param torch.Tensor driver_image: driver image
        :param torch.Tensor driver_landmarks: driver landmarks
        :returns: overall generator loss
        :rtype: torch.Tensor
        """
        reenacted_realness, reenacted_feature_maps = self.discriminator(
            reenacted_image, driver_landmarks
        )
        _, driver_feature_maps = self.discriminator(driver_image, driver_landmarks)
        return (
            self.gan_loss(reenacted_realness)
            + (self.lambda_p * self.vgg19_loss(reenacted_image, driver_image))
            + (self.lambda_pf * self.vgg_vd_16_loss(reenacted_image, driver_image))
            + (
                self.lambda_fm
                * self.feature_map_loss(reenacted_feature_maps, driver_feature_maps)
            )
        )

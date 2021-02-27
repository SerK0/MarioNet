import torch
import torch.nn as nn

from .config import Config
from .marionet_modules import TargetEncoder, Decoder, DriverEncoder, Blender


class MarioNet(nn.Module):
    """
    MarioNet main model.

    Consists of all the building blocks defined in ./marionet_modules.py.
    Does everything except for the preprocessing as it can (and should)
    be done separately to speed up training process.
    """

    def __init__(self, config: Config) -> None:
        """
        :param Config config: config
        :returns: None
        """
        super(MarioNet, self).__init__()
        self.config = config.model.MarioNet

        self.tensor_conv = nn.Conv2d(
            in_channels=self.config.image_channels + self.config.landmark_channels,
            out_channels=self.config.tensor_channels,
            kernel_size=3,
            padding=1,
        )
        self.target_encoder = TargetEncoder(config)
        self.driver_encoder = DriverEncoder(config)
        self.blender = Blender(config)
        self.decoder = Decoder(config)

    def forward(
        self,
        target_image: torch.Tensor,
        target_landmarks: torch.Tensor,
        driver_image: torch.Tensor,
        driver_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        MarioNet forward pass.

        :param torch.Tensor driver: driver image, shape [B, C, W, H]
        :param torch.Tensor driver_landmarks: driver landmarks, shape [B, L, W, H]
        :param torch.Tensor target: target images, shape [B, N, C, W, H]
        :param torch.Tensor target_landmarks: target landmarks, shape [B, N, C, W, H]
        :returns: reenacted target image
        :rtype: torch.Tensor

        Here B - batch size, C - image channels (i.e. self.config.in_channels), L - landmarks channels,
        N - number of target images for few-shot reenactment, W/H - image width/height.
        """
        driver_tensor = torch.cat([driver_image, driver_landmarks], dim=1)
        driver_tensor = self.tensor_conv(driver_tensor)
        zx = self.driver_encoder(driver_tensor)

        batch_size, num_targets, image_channels, width, height = target_image.shape
        target_image = target_image.view(-1, image_channels, width, height)
        _, _, landmark_channels, _, _ = target_landmarks.shape
        target_landmarks = target_landmarks.view(-1, landmark_channels, width, height)
        target_tensor = torch.cat([target_image, target_landmarks], dim=1)
        target_tensor = self.tensor_conv(target_tensor)
        s, zy = self.target_encoder(target_tensor)

        s = [
            torch.mean(
                feature_map.view(
                    batch_size,
                    num_targets,
                    feature_map.size(1),
                    feature_map.size(2),
                    feature_map.size(3),
                ),
                dim=1,
            )
            for feature_map in s
        ]
        zy = zy.view(batch_size, num_targets, zy.size(1), zy.size(2), zy.size(3))

        zxy = self.blender(zx, zy)
        return self.decoder(zxy, s)

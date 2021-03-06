import torch
import torch.nn as nn


class ConvMerger(nn.Module):
    """
    Merges image and landmark and projects them into intermediate tensor representation.
    """

    def __init__(
        self, image_channels: int, landmarks_channels: int, tensor_channels: int
    ) -> None:
        """
        :param int image_channels: image channels
        :param int landmarks_channels: landmarks channels
        :param int tensor_channels: output tensor channels
        :returns: None
        """
        super(ConvMerger, self).__init__()
        self.conv_projection = nn.Conv2d(
            in_channels=image_channels + landmarks_channels,
            out_channels=tensor_channels,
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

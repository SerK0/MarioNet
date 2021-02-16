import torch
import torch.nn as nn

from .utils import warp_image


class WarpAlignmentBlock(nn.Module):
    """
    'To adapt pose-normalized feature maps to the pose of the driver, we generate an estimated
      flow map of the driver fu using 1x1 convolution that takes u as the input. Alignment by
      T(Sj; fu) follows. Then, the result is concatenated to u and fed into the following
      residual upsampling block.'
    """

    def __init__(self, in_channels: int) -> None:
        """
        :param int in_channels: input channels
        :returns: None
        """
        super(WarpAlignmentBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=2, kernel_size=1)

    def forward(self, x: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :param torch.Tensor feature_map: feature map input tensor
        :rtype: torch.Tensor
        """
        optical_flow = torch.tanh(self.conv(x))
        return warp_image(feature_map, optical_flow)

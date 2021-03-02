import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp_alignment import WarpAlignmentBlock
from .positional_encoding import PositionalEncoding


"""
Main building blocks for the MarioNet model.

ResBlockDown, ResBlockUp and UNetResBlockUp are BigGAN blocks.
(see more at https://arxiv.org/abs/1809.11096)

Decoder Block is the MarioNet's Decoder block based on ResBlockUp.

SelfAttentionBlock is used in MarioNet's Blender.
"""


class ResBlockDown(nn.Module):
    """
    BigGAN downsampling block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        :param int in_channels: input channels
        :param int out_channels: output channels
        :returns: None
        """
        super(ResBlockDown, self).__init__()

        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2),
        )

        self.conv_downsample = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :rtype: torch.Tensor
        """
        return self.residual_connection(x) + self.conv_downsample(x)


class ResBlockUp(nn.Module):
    """
    BigGAN upsampling block.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        :param int in_channels: input channels
        :param int out_channels: output channels
        :returns: None
        """
        super(ResBlockUp, self).__init__()

        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

        self.conv_upsample = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :rtype: torch.Tensor
        """
        return self.residual_connection(x) + self.conv_upsample(x)


class UNetResBlockUp(nn.Module):
    """
    U-Net (https://arxiv.org/abs/1505.04597) adaptation of BigGAN upsampling block.
    """

    def __init__(
        self, in_channels: int, skip_connection_channels: int, out_channels: int
    ) -> None:
        """
        :param int in_channels: input channels
        :param int skip_connection_channels: skip-connection input channels
        :param int out_channels: output channels
        :returns: None
        """
        super(UNetResBlockUp, self).__init__()

        self.residual_connection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

        self.upsample = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_connection_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :param torch.Tensor skip_connection: skip connection
        :rtype: torch.Tensor
        """
        upsampled = self.upsample(x)
        upsampled = torch.cat([upsampled, skip_connection], dim=1)
        return self.residual_connection(x) + self.conv(upsampled)


class DecoderBlock(nn.Module):
    """
    Decoder basic building block. Consists of WarpAlignmentBlock and ResBlockUp.
    """

    def __init__(
        self, in_channels: int, feature_map_channels: int, out_channels: int
    ) -> None:
        """
        :param int in_channels: input channels
        :param int feature_map_channels: feature map input channels
        :param int out_channels: output channels
        :returns: None
        """
        super(DecoderBlock, self).__init__()
        self.warp_alignment = WarpAlignmentBlock(in_channels)
        self.res_upsample_block = ResBlockUp(
            in_channels + feature_map_channels, out_channels
        )

    def forward(self, x: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor x: input tensor
        :param torch.Tensor feature_map: feature map input tensor
        :rtype: torch.Tensor
        """
        warp_aligned_feature_map = self.warp_alignment(x, feature_map)
        x = torch.cat([x, warp_aligned_feature_map], dim=1)
        return self.res_upsample_block(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, driver_feature_dim, target_feature_dim, attention_feature_dim):
        super(SelfAttentionBlock, self).__init__()

        self.q_proj = nn.Linear(driver_feature_dim, attention_feature_dim)
        self.px_proj = nn.Linear(driver_feature_dim, attention_feature_dim)

        self.k_proj = nn.Linear(target_feature_dim, attention_feature_dim)
        self.py_proj = nn.Linear(target_feature_dim, attention_feature_dim)

        self.v_proj = nn.Linear(target_feature_dim, driver_feature_dim)

        self.attention_feature_size = attention_feature_dim

    def forward(self, zx: torch.Tensor, zy: torch.Tensor) -> torch.Tensor:
        """
        :param zx: driver feature map tensor --- size: [B x cx x H x W]
        :param zy: target feature map tensor --- size: [B x K x cy x H x W]
        :return:
            self 'attentioned' feature map
        """
        batch_size, cx, hx, wx = zx.size()
        Px = torch.cat(
            [
                PositionalEncoding.get_matrix((hx, wx, cx)).unsqueeze(0)
                for _ in range(batch_size)
            ],
            dim=0,
        )
        q = self.q_proj(zx.permute(0, 2, 3, 1)) + self.px_proj(Px)

        batch_size, K, cy, h, w = zy.size()

        Py = torch.cat(
            [
                PositionalEncoding.get_matrix((h, w, cy)).unsqueeze(0)
                for _ in range(batch_size * K)
            ],
            dim=0,
        )
        Py = Py.view(batch_size, K, h, w, cy)

        k = self.k_proj(zy.permute(0, 1, 3, 4, 2)) + self.py_proj(Py)

        v = self.v_proj(zy.permute(0, 1, 3, 4, 2))

        q_flatten = q.view(batch_size, -1, self.attention_feature_size)
        k_flatten = k.view(batch_size, -1, self.attention_feature_size)

        attn_value = torch.bmm(q_flatten, k_flatten.permute(0, 2, 1)) / torch.sqrt(
            torch.tensor(self.attention_feature_size, dtype=torch.float32)
        )

        softmax_attentioned = F.softmax(attn_value.view(batch_size, -1), dim=0).view(
            *attn_value.size()
        )
        output_t = torch.bmm(softmax_attentioned, v.view(batch_size, -1, cx))

        return output_t.view(batch_size, cx, hx, wx)

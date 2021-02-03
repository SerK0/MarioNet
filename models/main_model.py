import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        stride = 1 if not downsample else 2

        if self.downsample:
            self.identity_sparse = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.identity_sparse = nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):

        interm_res = self.act1(self.bn1(self.conv1(x)))
        final_res = self.bn2(self.conv2(interm_res))

        return self.act(final_res + self.identity_sparse(x))


class DriverEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(DriverEncoder, self).__init__()

        self.block1 = BasicBlock(feature_dim, feature_dim * 2, downsample=True)
        self.block2 = BasicBlock(feature_dim * 2, feature_dim * 4, downsample=True)
        self.block3 = BasicBlock(feature_dim * 4, feature_dim * 8, downsample=True)
        self.block4 = BasicBlock(feature_dim * 8, feature_dim * 16, downsample=True)

    def forward(self, rx):
        x = self.block1(rx)
        x = self.block2(x)
        x = self.block3(x)
        zx = self.block4(x)

        return zx

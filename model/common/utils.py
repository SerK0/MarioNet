import itertools
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise(iterable: tp.Iterable[tp.Any]) -> tp.Iterable[tuple[tp.Any, tp.Any]]:
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    Code taken from https://docs.python.org/3/library/itertools.html

    :param tp.Iterable[tp.Any] iterable: iterable
    :returns: pairs iterable
    :rtype: tp.Iterable[tuple[tp.Any, tp.Any]]
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def warp_image(image: torch.Tensor, optical_flow: torch.Tensor) -> torch.Tensor:
    """
    Warps image WRT optical flow.

    :param torch.Tensor image: image to warp, shape [B, C, W, H]
    :param torch.Tensor optical_flow: optical flow, shape [B, 2, W, H]
    :returns: warped image, shape [B, C, W, H]
    :rtype: torch.Tensor

    Here B - batch size, C - channels, W - width, H - height.
    """
    optical_flow = optical_flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, optical_flow)

import itertools

import torch.nn as nn
import torch.nn.functional as F


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    Code taken from https://docs.python.org/3/library/itertools.html
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def warp_image(image, optical_flow):
    optical_flow = optical_flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, optical_flow)

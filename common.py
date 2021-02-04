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
    """
    Warp image according to optical flow map.
    Heavily influenced by https://github.com/AliaksandrSiarohin/monkey-net/blob/master/modules/generator.py#L51
    """
    _, _, flow_h, flow_w = optical_flow.size()
    _, _, image_h, image_w = image.size()
    # TODO(binpord): MarioNETte authors use average pooling instead of nearest interpolation
    #   as opposed to the referenced paper.
    optical_flow = F.interpolate(optical_flow, size=(image_h, image_w), mode="nearest")
    optical_flow = optical_flow.permute(0, 2, 3, 1)
    return F.grid_sample(image, optical_flow)


class MarioNetModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["model"][self.__class__.__name__]

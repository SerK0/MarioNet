import itertools

import torch.nn as nn


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    Code taken from https://docs.python.org/3/library/itertools.html
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class MarioNetModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config["model"][self.__class__.__name__]

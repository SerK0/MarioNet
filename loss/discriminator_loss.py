
import torch
import torch.nn as nn


class DiscriminatorLossPatches(nn.Module):
    def __init__(self, base_loss=nn.BCELoss):
        super(DiscriminatorLossPatches, self).__init__()
        self.base_loss = base_loss()

    def __call__(self, features_tensor, target_type):
        if target_type == 'real':
            return self.base_loss(features_tensor, torch.ones(features_tensor.size(), device=features_tensor.device))
        elif target_type == 'fake':
            return self.base_loss(features_tensor, torch.zeros(features_tensor.size(), device=features_tensor.device))
        else:
            raise ValueError("Incorrect target_type: {}".format(target_type))

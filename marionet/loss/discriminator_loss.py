import torch
import torch.nn as nn


class DiscriminatorHingeLoss(nn.Module):
    """
    HingeLoss functor for Discriminator
    """

    def __init__(self):
        super(DiscriminatorHingeLoss, self).__init__()

    def __call__(self, real_discriminator_features, fake_discriminator_features):

        ones_tensor = torch.ones_like(real_discriminator_features)

        real_part = torch.max(
            torch.tensor([0, torch.mean(ones_tensor - real_discriminator_features)])
        )
        fake_part = torch.max(
            torch.tensor([0, torch.mean(ones_tensor - fake_discriminator_features)])
        )

        return real_part + fake_part


class DiscriminatorLossPatches(nn.Module):
    """
    BCELoss functor (default) for Discriminator
    """

    def __init__(self, base_loss=None):
        super(DiscriminatorLossPatches, self).__init__()
        if base_loss is None:
            self.base_loss = nn.BCELoss()
        else:
            self.base_loss = base_loss

    def __call__(self, features_tensor: torch.Tensor, is_real: bool):
        if is_real:
            return self.base_loss(
                features_tensor,
                torch.ones(features_tensor.size(), device=features_tensor.device),
            )
        else:
            return self.base_loss(
                features_tensor,
                torch.zeros(features_tensor.size(), device=features_tensor.device),
            )

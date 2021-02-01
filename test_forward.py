import torch

from target_encoder import TargetEncoder


def test_target_encoder():
    batch_size = 2
    image_channels = 3
    image_dim = 224
    landmark_channels = 2
    image = torch.rand(batch_size, image_channels, image_dim, image_dim)
    landmark = torch.rand(batch_size, landmark_channels, image_dim, image_dim)
    TargetEncoder(image_channels, landmark_channels)(image, landmark)

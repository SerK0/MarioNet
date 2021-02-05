import torch

from config import Config
from target_encoder import TargetEncoder


def test_target_encoder():
    batch_size = 2
    image_channels = 3
    image_dim = 224
    landmark_channels = 2
    image = torch.rand(batch_size, image_channels, image_dim, image_dim)
    landmark = torch.rand(batch_size, landmark_channels, image_dim, image_dim)
    config = Config(
        {
            "model": {
                "TargetEncoder": {
                    "image_channels": image_channels,
                    "landmark_channels": landmark_channels,
                    "downsampling_channels": [
                        64,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                    ],
                    "upsampling_channels": [
                        2048,
                        1024,
                        512,
                        256,
                        128,
                    ],
                }
            }
        }
    )
    TargetEncoder(config)(image, landmark)

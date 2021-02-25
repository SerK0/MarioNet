import pytest
import torch
import yaml

from model.config import Config
from model.marionet_modules import TargetEncoder, Decoder


@pytest.fixture
def config():
    return Config(
        yaml.load(
            """
        model:
            TargetEncoder:
                image_channels: 3
                landmark_channels: 2
                downsampling_channels:
                    - 64
                    - 128
                    - 256
                    - 512
                    - 512
                    - 512
                upsampling_channels:
                    - 512
                    - 512
                    - 256
                    - 128
                    - 64
            Decoder:
                channels:
                    - 512
                    - 512
                    - 256
                    - 128
                    - 64
                output_channels: 3
        """
        )
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def image_dim():
    return 224


def test_target_encoder(batch_size, image_dim, config):
    image_channels = 3
    landmark_channels = 2
    image = torch.rand(batch_size, image_channels, image_dim, image_dim)
    landmark = torch.rand(batch_size, landmark_channels, image_dim, image_dim)
    TargetEncoder(config)(image, landmark)


def test_decoder(batch_size, config):
    lowest_resolution = 7  # to match 224 image dim
    blender_output = torch.rand(
        batch_size,
        config.model.Decoder.channels[-1],
        lowest_resolution,
        lowest_resolution,
    )
    target_encoder_feature_maps = []
    resolution = lowest_resolution
    for channels in config.model.TargetEncoder.downsampling_channels[1:-1:-1]:
        target_encoder_feature_maps.append(
            torch.rand(batch_size, channels, resolution, resolution)
        )
        resolution *= 2

    target_encoder_feature_maps = list(reversed(target_encoder_feature_maps))
    Decoder(config)(blender_output, target_encoder_feature_maps)


def test_driver_encoder():
    pass


def test_blender():
    pass

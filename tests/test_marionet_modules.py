import pytest
import torch
import yaml

from pathlib import Path

from marionet.config import Config
from marionet.model.marionet_modules import TargetEncoder, Decoder


@pytest.fixture
def config_path():
    project_dir = Path(__file__).parent.parent
    return project_dir / "config.yaml"


@pytest.fixture
def config(config_path):
    with open(config_path, "r") as f:
        return Config(yaml.load(f.read()))


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def image_dim():
    return 224


@pytest.fixture
def tensor_channels():
    return 64


@pytest.fixture
def target_tensor(batch_size, tensor_channels, image_dim):
    return torch.rand(batch_size, tensor_channels, image_dim, image_dim)


def test_target_encoder(config, target_tensor):
    TargetEncoder(config)(target_tensor)


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

    Decoder(config)(blender_output, target_encoder_feature_maps)


def test_driver_encoder():
    pass


def test_blender():
    pass


def test_discriminator():
    pass

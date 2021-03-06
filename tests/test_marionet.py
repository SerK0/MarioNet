import pytest
import torch
import yaml

from pathlib import Path

from marionet.config import Config
from marionet.model.marionet import MarioNet


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
    return 256


@pytest.fixture
def image_channels():
    return 3


@pytest.fixture
def landmark_channels():
    return 3


@pytest.fixture
def num_targets():
    return 2


@pytest.fixture
def target_image(batch_size, num_targets, image_channels, image_dim):
    return torch.rand(batch_size, num_targets, image_channels, image_dim, image_dim)


@pytest.fixture
def target_landmarks(batch_size, num_targets, landmark_channels, image_dim):
    return torch.rand(batch_size, num_targets, landmark_channels, image_dim, image_dim)


@pytest.fixture
def driver_landmarks(batch_size, landmark_channels, image_dim):
    return torch.rand(batch_size, landmark_channels, image_dim, image_dim)


def test_marionet(
    config,
    target_image,
    target_landmarks,
    driver_landmarks,
):
    MarioNet(config)(target_image, target_landmarks, driver_landmarks)

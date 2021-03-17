import torch

from pathlib import Path

from marionet.model import Discriminator
from marionet.config import Config
from marionet.loss import GeneratorLoss


def test_generator_loss():
    project_root = Path(__file__).parent.parent
    config = Config.from_file(project_root / "config.yaml")
    criterion = GeneratorLoss(Discriminator(config))
    batch_size = 2
    reenacted_image = torch.rand(batch_size, 3, 224, 224)
    driver_image = torch.rand(batch_size, 3, 224, 224)
    driver_landmarks = torch.rand(batch_size, 3, 224, 224)
    criterion(reenacted_image, driver_image, driver_landmarks)

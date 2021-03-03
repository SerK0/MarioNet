import torch

from pathlib import Path

from marionet.model import MarioNet, Discriminator
from marionet.config import Config
from marionet.loss import GeneratorLoss


def test_generator_loss():
    project_root = Path(__file__).parent.parent
    config = Config.from_file(project_root / "config.yaml")
    criterion = GeneratorLoss(MarioNet(config), Discriminator(config))
    batch_size = 2
    output = torch.rand(batch_size, 3, 224, 224)
    target = torch.rand(batch_size, 3, 224, 224)
    target_landmarks = torch.rand(batch_size, 3, 224, 224)
    criterion(output, target, target_landmarks)

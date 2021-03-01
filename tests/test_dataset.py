from pathlib import Path
from torch.utils.data import DataLoader

from model.dataset import MarioNetDataset
from model.config import Config


def test_dataloader():
    cfg = Config.from_file(Path(__file__).parent.parent / "model/config/config.yaml")

    marionet_dataset = MarioNetDataset(
        cfg.dataset.folder,
        cfg.dataset.faces_structure,
        cfg.dataset.identity_structure,
        cfg.dataset.video_structure,
        cfg.dataset.n_target_images,
        cfg.dataset.image_size,
    )

    item = 1
    n_channels = 3
    assert list(marionet_dataset[item]["source_image"].size()) == [
        n_channels,
        cfg.dataset.image_size,
        cfg.dataset.image_size,
    ]
    assert list(marionet_dataset[item]["source_landmarks"].size()) == [
        n_channels,
        cfg.dataset.image_size,
        cfg.dataset.image_size,
    ]
    assert list(marionet_dataset[item]["target_images"].size()) == [
        cfg.dataset.n_target_image,
        n_channels,
        cfg.dataset.image_size,
        cfg.dataset.image_size,
    ]
    assert list(marionet_dataset[item]["target_landmarks"].size()) == [
        cfg.dataset.n_target_image,
        n_channels,
        cfg.dataset.image_size,
        cfg.dataset.image_size,
    ]
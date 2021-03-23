import os
import random
import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from marionet.config import Config
from marionet.dataset.dataset import MarioNetDataset
from marionet.loss import DiscriminatorHingeLoss, GeneratorLoss
from marionet.model.discriminator import Discriminator
from marionet.model.marionet import MarioNet
from utils import Trainer


def check_dataloader(marionet_dataset_dataloader):
    batch = next(iter(marionet_dataset_dataloader))
    print(batch["source_image"].size())
    print(batch["source_landmarks"].size())
    print(batch["target_images"].size())
    print(batch["target_landmarks"].size())


def main(cfg: Config):

    np.random.seed(cfg.training.random_seed)
    random.seed(cfg.training.random_seed)
    torch.manual_seed(cfg.training.random_seed)

    identities = os.listdir(
        os.path.join(cfg.dataset.folder, cfg.dataset.identity_structure)
    )
    random.shuffle(identities)

    train_identities = identities[: -cfg.training.number_indentities_in_test]
    test_identities = identities[-cfg.training.number_indentities_in_test :]

    marionet_dataset_train = MarioNetDataset(
        cfg.dataset.folder,
        cfg.dataset.faces_structure,
        train_identities,
        cfg.dataset.video_structure,
        cfg.dataset.n_target_images,
        cfg.dataset.image_size,
    )

    marionet_dataset_test = MarioNetDataset(
        cfg.dataset.folder,
        cfg.dataset.faces_structure,
        test_identities,
        cfg.dataset.video_structure,
        cfg.dataset.n_target_images,
        cfg.dataset.image_size,
    )

    train_dataloader = DataLoader(
        marionet_dataset_train,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )

    test_dataloader = DataLoader(
        marionet_dataset_test,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )

    generator = MarioNet(cfg)
    discriminator = Discriminator(cfg)

    criterion_generator = GeneratorLoss(discriminator)
    criterion_discriminator = DiscriminatorHingeLoss()

    optimizer_generator = Adam(generator.parameters(), lr=cfg.training.generator.lr)
    optimizer_discriminator = Adam(
        discriminator.parameters(), lr=cfg.training.discriminator.lr
    )

    print(
        f"train_size_identities = {len(marionet_dataset_train)}, test_size_identities = {len(marionet_dataset_test)}"
    )
    Trainer(cfg).training(
        generator=generator,
        discriminator=discriminator,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion_generator=criterion_generator,
        criterion_dicriminator=criterion_discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
    )


if __name__ == "__main__":
    cfg = Config.from_file(Path(__file__).parent / "config.yaml")
    main(cfg)

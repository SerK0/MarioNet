import os
import torch
import wandb
import argparse

from pathlib import Path
from random import shuffle

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


def train_model(cfg: Config, device: str) -> None:

    identities = os.listdir(
        os.path.join(cfg.dataset.folder, cfg.dataset.identity_structure)
    )
    shuffle(identities)

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

    generator = MarioNet(cfg).to(device)
    discriminator = Discriminator(cfg).to(device)

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
    default_config_p = str(Path(__file__).parent / "config.yaml")

    parser = argparse.ArgumentParser("MarioNet training parameters")
    parser.add_argument('--config', type=str, default=default_config_p)
    parser.add_argument('--wandb-name', type=str, default='marionet_training')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = Config.from_file(args.config)

    if config.training.wandb_logging:
        wandb.init(name=args.wandb_name, project="MarioNet", config={
            "Architecture": "MarioNett",
            "batch_size": config.training.batch_size
        })

    if args.device[:4] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_model(config, device)

    if config.training.wandb_logging:
        wandb.finish()

    print("Training_finished")

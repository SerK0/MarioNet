import typing as tp

from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam

from marionet.config import Config
from marionet.dataset.dataset import MarioNetDataset
from marionet.model.discriminator import Discriminator
from marionet.model.marionet import MarioNet
from marionet.loss import GeneratorLoss, DiscriminatorHingeLoss
from utils import Trainer


def check_dataloader(marionet_dataset_dataloader):
    batch = next(iter(marionet_dataset_dataloader))
    print(batch["source_image"].size())
    print(batch["source_landmarks"].size())
    print(batch["target_images"].size())
    print(batch["target_landmarks"].size())


def main(cfg: Config):

    marionet_dataset = MarioNetDataset(
        cfg.dataset.folder,
        cfg.dataset.faces_structure,
        cfg.dataset.identity_structure,
        cfg.dataset.video_structure,
        cfg.dataset.n_target_images,
        cfg.dataset.image_size,
    )

    marionet_dataset_dataloader = DataLoader(
        marionet_dataset,
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

    Trainer(cfg).training(
        generator=generator,
        discriminator=discriminator,
        train_dataloader=marionet_dataset_dataloader,
        criterion_generator=criterion_generator,
        criterion_dicriminator=criterion_discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
    )


if __name__ == "__main__":
    cfg = Config.from_file(Path(__file__).parent / "config.yaml")
    main(cfg)
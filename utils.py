import torch
import typing as tp

from marionet.config import Config
from marionet.dataset.dataset import MarioNetDataset
from marionet.model.discriminator import Discriminator
from marionet.model.marionet import MarioNet
from marionet.loss import GeneratorLoss, DiscriminatorHingeLoss


class Trainer:
    """
    Class for MarioNet training
    """

    def __init__(self, cfg: Config, num_epoch: int = 100):
        """
        params Config cfg: config file with training parameters
        params int max_epoch: maximum number of epoch to train
        """
        self.cfg = cfg.training
        self.num_epoch = num_epoch

    def training(
        self,
        generator: MarioNet,
        discriminator: Discriminator,
        train_dataloader: MarioNetDataset,
        criterion_generator: GeneratorLoss,
        criterion_dicriminator: DiscriminatorHingeLoss,
        optimizer_generator: torch.optim.Adam,
        optimizer_discriminator: torch.optim.Adam,
    ) -> None:
        """
        Training pipeline of MarioNet model

        params MarioNet generator: generator part of network
        params Discriminator discriminator: dicriminator part of network
        params MarioNetDataset train_dataloader: train dataloader of images for MarioNet training
        params GeneratorLoss criterion_generator: generator loss
        params DiscriminatorHingeLoss criterion_dicriminator: dicriminator loss
        params torch.optim.Adam optimizer_generator: generator optimizator
        params torch.optim.Adam optimizer_discriminator: dicriminator optimizator
        """

        for epoch in range(self.num_epoch):
            print(f"Epoch {epoch}")
            for num_batch, batch in enumerate(train_dataloader):

                discriminator_loss = self.discriminator_step(
                    generator,
                    discriminator,
                    batch,
                    criterion_dicriminator,
                    optimizer_discriminator,
                )

                generator_loss = self.generator_step(
                    generator,
                    discriminator,
                    batch,
                    criterion_generator,
                    optimizer_generator,
                )

                print(
                    f"Num_batch {num_batch}, generator_loss {generator_loss}, discriminator_loss {discriminator_loss}"
                )

    def generator_step(
        self,
        generator: MarioNetDataset,
        discriminator: Discriminator,
        batch: tp.Dict[str, torch.Tensor],
        criterion_generator: GeneratorLoss,
        optimizer_generator: torch.optim.Adam,
    ) -> float:
        """
        Generator step

        params MarioNet generator: generator part of network
        params Discriminator discriminator: dicriminator part of network
        params tp.Dict[str, torch.Tensor] batch: batch of data consisting of driver/target images/landmarks
        params GeneratorLoss criterion_generator: generator loss
        params torch.optim.Adam optimizer_generator: generator optimizator
        return: batch loss for generator
        """
        generator.train()
        discriminator.eval()

        optimizer_generator.zero_grad()

        reenacted_images = generator(
            target_image=batch["target_images"],
            target_landmarks=batch["target_landmarks"],
            driver_landmarks=batch["driver_landmarks"],
        )

        loss = criterion_generator(
            reenacted_image=reenacted_images,
            driver_image=batch["driver_image"],
            driver_landmarks=batch["driver_landmarks"],
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(), self.cfg.generator.clipping
        )
        optimizer_generator.step()

        return loss.item()

    def discriminator_step(
        self,
        generator: MarioNetDataset,
        discriminator: Discriminator,
        batch: tp.Dict[str, torch.Tensor],
        criterion_dicriminator: DiscriminatorHingeLoss,
        optimizer_discriminator: torch.optim.Adam,
    ) -> float:
        """
        Discriminator step

        params MarioNet generator: generator part of network
        params Discriminator discriminator: dicriminator part of network
        params tp.Dict[str, torch.Tensor] batch: batch of data consisting of driver/target images/landmarks
        params DiscriminatorHingeLoss criterion_dicriminator: dicriminator loss
        params torch.optim.Adam optimizer_discriminator: dicriminator optimizator
        return: batch loss for dicriminator
        """
        generator.eval()
        discriminator.train()

        reenacted_images = generator(
            target_image=batch["target_images"],
            target_landmarks=batch["target_landmarks"],
            driver_landmarks=batch["driver_landmarks"],
        ).detach()

        optimizer_discriminator.zero_grad()

        features_tensor_driver_images = discriminator(
            image=batch["driver_image"], landmarks=batch["driver_landmarks"]
        )[0]
        features_tensor_reenacted_images = discriminator(
            image=reenacted_images, landmarks=batch["driver_landmarks"]
        )[0]

        loss = criterion_dicriminator(
            real_discriminator_features=features_tensor_driver_images,
            fake_discriminator_features=features_tensor_reenacted_images,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), self.cfg.discriminator.clipping
        )
        optimizer_discriminator.step()

        return loss.item()
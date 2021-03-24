import typing as tp

import wandb
import torch
import numpy as np
from torchvision.utils import save_image
from pathlib import Path
from shutil import rmtree
from tqdm.auto import tqdm

from marionet.config import Config
from marionet.dataset.dataset import MarioNetDataset
from marionet.loss import DiscriminatorHingeLoss, GeneratorLoss
from marionet.model.discriminator import Discriminator
from marionet.model.marionet import MarioNet


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class Trainer:
    """
    Class for MarioNet training
    """

    def __init__(self, cfg: Config):
        """
        params Config cfg: config file with training parameters
        params int max_epoch: maximum number of epoch to train
        """
        self.cfg = cfg.training
        self.save_path = Path(cfg.training.log_dir)
        self.ckpt_save_dir = self.save_path / cfg.training.checkpoint_dir
        self.img_save_dir = self.save_path / cfg.training.image_log_dir
        self.wandb_logging = cfg.training.wandb_logging

        if not self.save_path.exists():
            self.save_path.mkdir()

        if self.ckpt_save_dir.exists():
            rmtree(self.ckpt_save_dir)

        self.ckpt_save_dir.mkdir()

        if self.img_save_dir.parent.exists():
            rmtree(self.img_save_dir.parent)

        self.img_save_dir.parent.mkdir(parents=True)

        self.ckpt_save_dir = str(self.ckpt_save_dir)
        self.img_save_dir = str(self.img_save_dir)

    def training(
        self,
        generator: MarioNet,
        discriminator: Discriminator,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
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
        params MarioNetDataset test_dataloader: test dataloader of images for testing Marionet module
        params GeneratorLoss criterion_generator: generator loss
        params DiscriminatorHingeLoss criterion_dicriminator: dicriminator loss
        params torch.optim.Adam optimizer_generator: generator optimizator
        params torch.optim.Adam optimizer_discriminator: dicriminator optimizator
        """

        best_generator_loss = 1e8

        for epoch in range(self.cfg.num_epoch):
            print(f"Epoch {epoch}")
            pbar = tqdm(train_dataloader, leave=False, desc='Starting train')
            generator_loss_history = []
            for batch_idx, batch in enumerate(pbar):

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
                generator_loss_history.append(generator_loss)

                new_description = "Epoch {}, Generator_loss: {:.5f}, discriminator_loss {:.5f}"
                pbar.set_description(new_description.format(epoch, generator_loss, discriminator_loss))

            if best_generator_loss > np.mean(generator_loss_history):
                best_generator_loss = np.mean(generator_loss_history)
                ckpt = {
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'epoch': epoch
                }

                torch.save(ckpt, self.ckpt_save_dir + '/ckpt.pth')

            self.generate_samples(
                generator, test_dataloader, index=f"{epoch}"
            )

    def generator_step(
        self,
        generator: MarioNet,
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

        gan_loss, vgg19_perceptual_loss, vgg_vd_16_perceptual_loss, feature_map_loss = criterion_generator(
            reenacted_image=reenacted_images,
            driver_image=batch["driver_image"],
            driver_landmarks=batch["driver_landmarks"],
        )
        if self.wandb_logging:
            wandb.log({
                'generator/gan_loss': gan_loss.item(),
                'generator/vgg19_perceptual_loss': vgg19_perceptual_loss.item(),
                'generator/vgg_vd_16_perceptual_loss': vgg_vd_16_perceptual_loss.item(),
                'generator/feature_map_loss': feature_map_loss.item()
            })

        loss = gan_loss + vgg19_perceptual_loss + vgg_vd_16_perceptual_loss + feature_map_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(), self.cfg.generator.clipping
        )
        optimizer_generator.step()

        return loss.item()

    def discriminator_step(
        self,
        generator: MarioNet,
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

        if self.wandb_logging:
            wandb.log({
                'discriminator/general': loss.item()
            })

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), self.cfg.discriminator.clipping
        )
        optimizer_discriminator.step()

        return loss

    @torch.no_grad()
    def generate_samples(
        self, generator: MarioNet, test_dataloader: MarioNetDataset, index: str = "0"
    ) -> None:
        """
        :param MarioNet generator: generator network
        :param MarioNetDataset test_dataloader: test dataloader
        :param int index: index of saving results
        :rtype: bool
        """

        generator.eval()
        for num_batch, batch in enumerate(test_dataloader):
            samples = [
                batch["driver_image"][0],
                batch["driver_landmarks"][0],
            ]

            for target_image in batch["target_images"][0]:
                samples.append(target_image)

            reenacted_images = generator(
                target_image=batch["target_images"],
                target_landmarks=batch["target_landmarks"],
                driver_landmarks=batch["driver_landmarks"],
            )

            samples.append(reenacted_images[0])

            generator_results = torch.cat(samples, dim=2)

            if self.wandb_logging:
                wandb.log({
                    'images/epoch_{}'.format(index): wandb.Image(generator_results.detach().permute(1, 2, 0).numpy())
                })

            save_image(
                generator_results,
                self.img_save_dir.format(index),
                nrow=1,
                padding=0,
                normalize=True,
            )

            break

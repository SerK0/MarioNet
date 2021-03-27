import typing as tp

import torch
import wandb
from imageio import imsave
from torchvision.utils import save_image

from marionet.config import Config
from marionet.dataset.dataset import MarioNetDataset
from marionet.loss import DiscriminatorHingeLoss, GeneratorLoss
from marionet.model.discriminator import Discriminator
from marionet.model.marionet import MarioNet


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def move_batch_to_device(
    batch: tp.Dict[str, torch.Tensor], device: str
) -> tp.Dict[str, torch.Tensor]:
    result_batch = {}

    for key, images in batch.items():
        result_batch[key] = images.to(device)

    return result_batch


class Trainer:
    """
    Class for MarioNet training
    """

    def __init__(self, cfg: Config, device: str):
        """
        params Config cfg: config file with training parameters
        params str device: device
        """
        self.cfg = cfg.training
        self.device = device
        self.wandb = wandb.init(
            project=cfg.training.wandb.project, name=cfg.training.wandb.name
        )

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

        print(
            f"train_size_identities = {len(train_dataloader)}, test_size_identities = {len(test_dataloader)}"
        )
        print(f"test identities ->>> {test_dataloader.dataset.identity_structure}")

        for epoch in range(self.cfg.num_epoch):
            print(f"Epoch {epoch}")
            for num_batch, batch in enumerate(train_dataloader):

                batch = move_batch_to_device(batch, self.device)

                discriminator_loss = self.discriminator_step(
                    generator,
                    discriminator,
                    batch,
                    criterion_dicriminator,
                    optimizer_discriminator,
                )

                if (num_batch + 1) % self.cfg.generator.step == 0:
                    (
                        gan_loss,
                        vgg19_perceptual_loss,
                        vgg_vd_16_perceptual_loss,
                        feature_map_loss,
                        overall_generator_loss,
                    ) = self.generator_step(
                        generator,
                        discriminator,
                        batch,
                        criterion_generator,
                        optimizer_generator,
                    )

                if (num_batch + 1) % self.cfg.logging.log_step == 0:
                    wandb.log(
                        {
                            "generator/overall loss": overall_generator_loss,
                            "generator/gan_loss": gan_loss,
                            "generator/vgg19_perceptual_loss": vgg19_perceptual_loss,
                            "generator/vgg_vd_16_perceptual_loss": vgg_vd_16_perceptual_loss,
                            "generator/feature_map_loss": feature_map_loss,
                            "discriminator/overall_loss": discriminator_loss,
                        }
                    )

                    print(
                        f"Num_batch {num_batch}, overall_generator_loss {overall_generator_loss}, discriminator_loss {discriminator_loss}"
                    )

            if (epoch + 1) % self.cfg.samples.sample_step == 0:
                input_images, reenacted_image = self.generate_samples(
                    generator, test_dataloader, index=f"{epoch}_{num_batch}"
                )

                input_images = input_images.permute(1, 2, 0).cpu().numpy()
                reenacted_image = reenacted_image.permute(1, 2, 0).cpu().numpy()

                wandb.log(
                    {
                        "images/epoch_{}".format(epoch): [
                            wandb.Image(input_images, caption="Input Images"),
                        ]
                    }
                )
            if (epoch + 1) % self.cfg.model_saving.step == 0:
                torch.save(
                    generator.state_dict(),
                    self.cfg.model_saving.path.format(f"generator_{epoch + 1}"),
                )
                torch.save(
                    discriminator.state_dict(),
                    self.cfg.model_saving.path.format(f"discriminator_{epoch + 1}"),
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

        (
            gan_loss,
            vgg19_perceptual_loss,
            vgg_vd_16_perceptual_loss,
            feature_map_loss,
        ) = criterion_generator(
            reenacted_image=reenacted_images,
            driver_image=batch["driver_image"],
            driver_landmarks=batch["driver_landmarks"],
        )

        loss = (
            gan_loss
            + vgg19_perceptual_loss
            + vgg_vd_16_perceptual_loss
            + feature_map_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            generator.parameters(), self.cfg.generator.clipping
        )
        optimizer_generator.step()

        return (
            gan_loss.item(),
            vgg19_perceptual_loss.item(),
            vgg_vd_16_perceptual_loss.item(),
            feature_map_loss.item(),
            loss.item(),
        )

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
            image=batch["driver_image"],
            landmarks=batch["driver_landmarks"],
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

    @torch.no_grad()
    def generate_samples(
        self,
        generator: MarioNet,
        test_dataloader: MarioNetDataset,
        index: str = "0",
        save_local: bool = False,
    ) -> bool:
        """
        :param MarioNet generator: generator network
        :param MarioNetDataset test_dataloader: test dataloader
        :param int index: index of saving results
        :param bool save_local: whether to save images locally
        :rtype: bool
        """

        generator.eval()
        for num_batch, batch in enumerate(test_dataloader):
            batch = move_batch_to_device(batch, self.device)

            samples = [
                batch["driver_image"][0],
                batch["driver_landmarks"][0],
            ]

            for target_image in batch["target_images"][0]:
                samples.append(target_image)

            reenacted_images = generator(
                target_image=batch["target_images"][0].unsqueeze(0),
                target_landmarks=batch["target_landmarks"][0].unsqueeze(0),
                driver_landmarks=batch["driver_landmarks"][0].unsqueeze(0),
            ).detach()

            samples.append(reenacted_images[0])

            generator_results = torch.cat(samples, dim=2)

            if save_local:

                save_image(
                    generator_results,
                    self.cfg.samples.saving_path.format(index),
                    nrow=1,
                    padding=0,
                    normalize=True,
                )

                save_image(
                    denorm(reenacted_images[0]),
                    self.cfg.samples.saving_path.format("generator_image_" + index),
                    nrow=1,
                    padding=0,
                    normalize=True,
                )

            break
        return generator_results, samples[-1]

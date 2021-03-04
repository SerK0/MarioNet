import torch


class Trainer:
    def __init__(self, max_epoch: int = 100):
        self.max_epoch = max_epoch

    def training(
        self,
        generator,
        discriminator,
        train_dataloader,
        criterion_generator,
        criterion_dicriminator,
        optimizer_generator,
        optimizer_discriminator,
    ):

        for epoch in range(self.max_epoch):
            for batch in train_dataloader:
                fake_images = self.generator_step(
                    generator,
                    discriminator,
                    batch,
                    criterion_generator,
                    optimizer_generator,
                )
                self.discriminator_step(
                    generator,
                    discriminator,
                    batch,
                    fake_images,
                    criterion_dicriminator,
                    optimizer_discriminator,
                )

    def generator_step(
        self, generator, discriminator, batch, criterion_generator, optimizer_generator
    ):
        generator.train()
        discriminator.eval()

        optimizer_generator.zero_grad()

        fake_images = generator.forward(
            target_image=batch["target_images"],
            target_landmarks=batch["target_landmarks"],
            driver_image=batch["source_image"],
            driver_landmarks=batch["source_landmarks"],
        )

        loss = criterion_generator(
            output=fake_images,
            target=batch["source_image"],
            target_landmarks=batch["source_landmarks"],
        )  # Пока непонятно, что подставлять в target нужно уточнить и переделать

        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
        generator.step()

        return fake_images

    def discriminator_step(
        self,
        generator,
        discriminator,
        batch,
        criterion_dicriminator,
        optimizer_discriminator,
    ):
        generator.eval()
        discriminator.train()

        optimizer_discriminator.zero_grad()

        loss = criterion_dicriminator()
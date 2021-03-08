import torch


class Trainer:
    def __init__(self, cfg, max_epoch: int = 100):
        self.cfg = cfg
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
            print({"Epoch {}".format(epoch)})
            for num, batch in enumerate(train_dataloader):

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
                    {
                        "Num_batch {0}, generator_loss {1}, discriminator_loss {2}".format(
                            num, generator_loss, discriminator_loss
                        )
                    }
                )

    def generator_step(
        self, generator, discriminator, batch, criterion_generator, optimizer_generator
    ):
        generator.train()
        discriminator.eval()

        optimizer_generator.zero_grad()

        reenacted_images = generator.forward(
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
            generator.parameters(), self.cfg.training.generator.lr
        )
        optimizer_generator.step()

        return loss.item()

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

        reenacted_images = generator.forward(
            target_image=batch["target_images"],
            target_landmarks=batch["target_landmarks"],
            driver_landmarks=batch["driver_landmarks"],
        ).detach()

        optimizer_discriminator.zero_grad()

        features_tensor_driver_images = discriminator.forward(
            image=batch["driver_image"], landmarks=batch["driver_landmarks"]
        )[0]
        features_tensor_reenacted_images = discriminator.forward(
            image=reenacted_images, landmarks=batch["driver_landmarks"]
        )[0]

        loss = criterion_dicriminator(
            real_discriminator_features=features_tensor_driver_images,
            fake_discriminator_features=features_tensor_reenacted_images,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(), self.cfg.training.discriminator.lr
        )
        optimizer_discriminator.step()

        return loss.item()
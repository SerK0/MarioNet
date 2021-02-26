from pathlib import Path

cfg = Config.from_file(Path(__file__).parent / "config/config.yaml")

marionet_dataset = MarioNetDataset(
    cfg.dataloader.folder,
    cfg.dataloader.faces_structure,
    cfg.dataloader.identity_structure,
    cfg.dataloader.video_structure,
    cfg.dataloader.n_target_image,
    cfg.dataloader.image_size,
)

dl = DataLoader(marionet_dataset, batch_size=4)

for b in dl:
    break

print(b["target_images"].size())

# for i, image in enumerate(b["target_images"][0]):
#     print(image.size())
#     io.imsave("/home/serk0/i{}.jpg".format(i), image.detach().numpy().T)

# for i, image in enumerate(b["target_landmarks"][0]):
#     print(image.size())
#     io.imsave("/home/serk0/l{}.jpg".format(i), image.detach().numpy().T)

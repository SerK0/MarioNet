import os
import numpy as np
import re

import torch
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io

from .config import Config

np.random.seed(1)


class MarioNetDataset(Dataset):
    def __init__(
        self,
        folder: str,
        faces_structure: str,
        identity_structure: str,
        video_structue: str,
        n_target_image=4,
        image_size=128,
    ):
        self.folder = folder
        self.faces = faces_structure
        self.identity_structure = os.listdir(os.path.join(folder, identity_structure))
        self.video_structue = video_structue
        self.n_target_image = n_target_image
        self.image_size = image_size

        transformations = torch.nn.Sequential(
            transforms.Resize((image_size,), interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
        )

        self.transforms = torch.jit.script(transformations)

    def __len__(self):
        return len(self.identity_structure)

    def __getitem__(self, index):
        source_identity = self.identity_structure[index]
        target_identity = np.random.choice(self.identity_structure, size=1)[0]
        target_identity_subfolders = os.listdir(
            os.path.join(self.folder, self.video_structue.format(target_identity))
        )

        path_to_source_faces = glob(
            os.path.join(self.folder, self.faces.format(source_identity, "*"))
        )

        target_identity_subfolder_name = np.random.choice(
            target_identity_subfolders, size=1
        )[0]

        path_to_target_faces = glob(
            os.path.join(
                self.folder,
                self.faces.format(target_identity, target_identity_subfolder_name),
            ),
        )

        source_face_path = np.random.choice(path_to_source_faces, size=1)[0]
        target_face_path = np.random.choice(
            path_to_target_faces, size=self.n_target_image
        )

        source_image = io.imread(source_face_path)
        source_landmarks = io.imread(re.sub("Faces", "Landmarks", source_face_path))

        target_images = []
        target_landmarks = []

        for target_face in target_face_path:
            target_image = io.imread(target_face)
            target_image_landmarks = io.imread(
                re.sub("Faces", "Landmarks", target_face)
            )

            target_images.append(self.__resize_image(target_image.T))
            target_landmarks.append(self.__resize_image(target_image_landmarks.T))

        result = {
            "source_image": self.__resize_image(source_image.T),
            "source_landmarks": self.__resize_image(source_landmarks.T),
            "target_images": torch.stack(target_images),
            "target_landmarks": torch.stack(target_landmarks),
        }

        return result

    def __resize_image(self, image):
        image = torch.tensor(image)
        return self.transforms.forward(image)


if __name__ == "__main__":
    import sys

    print(sys.path)
    from pathlib import Path

    cfg = Config.from_file(Path(__file__).parent / "config/config.yaml")

    marionet_dataset = MarioNetDataset(
        cfg.dataloader.folder,
        cfg.dataloader.faces_structure,
        cfg.dataloader.identity_structure,
        cfg.dataloader.video_structue,
        cfg.dataloader.n_target_image,
        cfg.dataloader.image_size,
    )

    dl = DataLoader(marionet_dataset, batch_size=4)

    for b in dl:
        break

    print(b["target_images"].size())

    for i, image in enumerate(b["target_images"][0]):
        print(image.size())
        io.imsave("/home/serk0/i{}.jpg".format(i), image.detach().numpy().T)

    for i, image in enumerate(b["target_landmarks"][0]):
        print(image.size())
        io.imsave("/home/serk0/l{}.jpg".format(i), image.detach().numpy().T)

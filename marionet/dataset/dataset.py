import os
import re
import typing as tp
from glob import glob

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class MarioNetDataset(Dataset):
    """
    Image/Landmark dataset for training MarioNet
    """

    def __init__(
        self,
        folder: str,
        faces_structure: str,
        identity_structure: tp.List,
        video_structure: str,
        n_target_images: int = 4,
        image_size: int = 128,
    ) -> None:
        """
        :param folder: Path to root of dataset.
        :param faces_structure: Subfolder structure in Faces subfolder of self.folder path.
        :param list identity_structure: List of pather to identies
        :param video_structure: Path to videos of particular identity
        :param n_target_images: Number of target images to sample
        :param image_size: Resize image to image_size x image_size
        :param bool return_test_dataset: wether to split to train/test dataset
        :returns: None
        """
        self.folder = folder
        self.faces = faces_structure
        self.identity_structure = identity_structure
        self.video_structure = video_structure
        self.n_target_images = n_target_images
        self.image_size = image_size

        self.transformations = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.identity_structure)

    def __getitem__(self, index: int) -> tp.Dict[str, torch.Tensor]:

        identity = np.random.choice(self.identity_structure, size=1)[0]
        identity_subfolders = os.listdir(
            os.path.join(self.folder, self.video_structure.format(identity))
        )

        identity_subfolder_name = np.random.choice(identity_subfolders, size=1)[0]

        path_to_faces = glob(
            os.path.join(
                self.folder,
                self.faces.format(identity, identity_subfolder_name),
            ),
        )

        if len(path_to_faces) < self.n_target_images + 1:
            return self.__getitem__(index)

        face_pathes = np.random.choice(
            path_to_faces,
            size=self.n_target_images + 1,
        )

        driver_face_path, target_face_path = face_pathes[0], face_pathes[1:]

        driver_image = Image.open(driver_face_path)
        driver_landmarks = Image.open(re.sub("Faces", "Landmarks", driver_face_path))

        target_images = []
        target_landmarks = []

        for target_face in target_face_path:
            target_image = Image.open(target_face)
            target_image_landmarks = Image.open(
                re.sub("Faces", "Landmarks", target_face)
            )

            target_images.append(self.__resize_image(target_image))
            target_landmarks.append(self.__resize_image(target_image_landmarks))

        result = {
            "driver_image": self.__resize_image(driver_image),
            "driver_landmarks": self.__resize_image(driver_landmarks),
            "target_images": torch.stack(target_images),
            "target_landmarks": torch.stack(target_landmarks),
        }

        return result

    def __resize_image(self, image: np.array) -> torch.Tensor:
        """
        :param image: Image in numpy format, H x W x C.
        """
        return self.transformations(image)
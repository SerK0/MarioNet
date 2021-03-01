import os
import re
from glob import glob

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class MarioNetDataset(Dataset):
    def __init__(
        self,
        folder: str,
        faces_structure: str,
        identity_structure: str,
        video_structure: str,
        n_target_images: int = 4,
        image_size: int = 128,
    ):
        '''
            :nparam folder: Path to root of dataset.
            :nparam faces_structure: Subfolder structure in Faces subfolder of self.folder path.
            :nparam identity_structure: Path to identies
            :nparam video_structure: Path to videos of particular identity
            :nparam n_target_images: Number of target images to sample
            :nparam image_size: Resize image to image_size x image_size
        '''
        self.folder = folder
        self.faces = faces_structure
        self.identity_structure = os.listdir(os.path.join(folder, identity_structure))
        self.video_structure = video_structure
        self.n_target_images = n_target_images
        self.image_size = image_size

        transformations = torch.nn.Sequential(
            transforms.Resize((image_size,), interpolation=Image.BILINEAR),
            transforms.CenterCrop(image_size),
        )

        self.transforms = torch.jit.script(transformations)

    def __len__(self):
        return len(self.identity_structure)

    def __getitem__(self, index: int) -> dict:
        source_identity = self.identity_structure[index]
        target_identity = np.random.choice(self.identity_structure, size=1)[0]
        target_identity_subfolders = os.listdir(
            os.path.join(self.folder, self.video_structure.format(target_identity))
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
            path_to_target_faces, size=self.n_target_images
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

    def __resize_image(self, image: np.array) -> torch.Tensor:
        image = torch.tensor(image)
        return self.transforms(image)
import os
import re
from glob import glob
from tqdm import tqdm

from preprocessor import LandmarkExtractor
from skimage import io

from config.config import Config


def extract_landmarks(config):

    path_to_images = glob(os.path.join(config.path_to_voxceleb, config.subfolders))

    landmark_extractor = LandmarkExtractor(
        landmark_type=config.landmark_type, device=config.device
    )

    for path_to_image in tqdm(path_to_images):
        saving_path = re.sub("/Faces", "/Landmarks", path_to_image)
        saving_folder = "/".join(saving_path.split("/")[:-1])
        os.makedirs(saving_folder, exist_ok=True)
        image_with_landmarks = landmark_extractor(path_to_image)
        io.imsave(saving_path, image_with_landmarks)


if __name__ == "__main__":
    config = Config.from_file("config/config.yaml")
    extract_landmarks(config)

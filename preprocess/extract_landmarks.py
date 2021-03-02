import os
import re
from glob import glob
from tqdm import tqdm

from pathlib import Path
from skimage import io

from model.config import Config
from .preprocess import LandmarkExtractor


def extract_landmarks():
    """
    Extracting landmark for images from subfolders
    """
    cfg = Config.from_file(Path(__file__).parent.parent / "model/config/config.yaml")

    path_to_images = glob(
        os.path.join(cfg.preprocess.folder, cfg.preprocess.dataset_structure)
    )
    landmark_extractor = LandmarkExtractor(
        landmark_type=cfg.preprocess.landmark_type, device=cfg.preprocess.device
    )

    for path_to_image in tqdm(path_to_images):
        saving_path = re.sub("Faces", "Landmarks", path_to_image)
        saving_folder = "/".join(saving_path.split("/")[:-1])
        os.makedirs(saving_folder, exist_ok=True)
        image_with_landmarks = landmark_extractor(path_to_image)
        io.imsave(saving_path, image_with_landmarks)
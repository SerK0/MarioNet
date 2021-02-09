import os
from glob import glob
from tqdm import tqdm

from preprocessor import LandmarkExtractor
from skimage import io

if __name__ == '__main__':
    path_to_images = glob('/home/serk0/Business/new_live/VoxCeleb1/Faces/*/*/*/*')
    landmark_extractor = LandmarkExtractor(landmark_type='3D', device='cpu')

    for path_to_image in tqdm(path_to_images):
        saving_path = path_to_image[:39] + '/Landmarks' + path_to_image[45:]
        saving_folder = '/'.join(saving_path.split('/')[:-1])
        os.makedirs(saving_folder, exist_ok=True)
        image_with_landmarks = landmark_extractor(path_to_image)
        io.imsave(saving_path, image_with_landmarks)
import collections
import numpy as np

import face_alignment as fa
import cv2
from skimage import io


class LandmarkExtractor:
    """
    Class for landmark extraction from images with faces
    """
    def __init__(self, landmark_type: str, device: str, extract_faces: bool = False)->None:
        """
        :param str landmark_type: value -> '2D' or '3D'
        :param str device: value -> 'cpu' or 'cuda'
        :returns: None
        """

        assert landmark_type == "2D" or landmark_type == "3D"

        self.landmark_type = (
            fa.LandmarksType._2D if landmark_type == "2D" else fa.LandmarksType._3D
        )
        self.face_alignment = fa.FaceAlignment(self.landmark_type, device=device)
        self.extract_faces = extract_faces
        self.detected_faces = None
        self.pred_type = collections.namedtuple("prediction_type", ["slice", "color"])
        self.pred_types = {
            "face": self.pred_type(slice(0, 17), (0, 128, 0)),
            "eyebrow1": self.pred_type(slice(17, 22), (255, 255, 0)),
            "eyebrow2": self.pred_type(slice(22, 27), (255, 255, 0)),
            "nose": self.pred_type(slice(27, 31), (0, 0, 255)),
            "nostril": self.pred_type(slice(31, 36), (0, 0, 255)),
            "eye1": self.pred_type(slice(36, 42), (255, 0, 0)),
            "eye2": self.pred_type(slice(42, 48), (255, 0, 0)),
            "lips": self.pred_type(slice(48, 60), (0, 255, 255)),
            "teeth": self.pred_type(slice(60, 68), (0, 255, 255)),
        }

    def __call__(self, input_path: str) -> np.array:
        """
        :param str input_path: path to image
        :rtype: np.array
        """

        input_image = io.imread(input_path)

        if self.extract_faces:
            self.detected_faces = np.array([0, 0, input_image.shape[0], input_image[1]])

        predicted_landmarks = self.face_alignment.get_landmarks_from_image(
            input_image, detected_faces=self.detected_faces
        )[-1]

        image_with_landmarks = self.__generate_image_with_cv2(
            input_image, predicted_landmarks
        )

        return image_with_landmarks

    def __generate_image_with_cv2(
        self, input_image: np.array, 
        predicted_landmarks: np.array
    ) -> np.array:
        """
        :param input_image: np.array H x W x C
        :param predicted_landmarks: np.array 68 x 3 for 3D
        :rtype: np.array
        """

        output_image = np.zeros_like(input_image)

        for name, pred_type in zip(self.pred_types, self.pred_types.values()):
            points = predicted_landmarks[:, :2][pred_type.slice].astype(int)

            for point1, point2 in zip(points, points[1:]):
                cv2.line(output_image, tuple(point1), tuple(point2), pred_type.color, 2)

            if name in ["eye1", "eye2", "lips", "teeth"]:
                cv2.line(
                    output_image,
                    tuple(points[0]),
                    tuple(points[-1]),
                    pred_type.color,
                    2,
                )

        return output_image
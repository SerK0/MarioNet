import collections
import face_alignment as fa
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


class LandmarkExtractor():
    def __init__(self, landmark_type, device):
        '''
        :nparam landmark_type: value -> '2D' or '3D'
        :nparam device: value -> 'cpu' or 'cuda'
        '''

        assert landmark_type == '2D' or landmark_type == '3D'

        self.landmark_type = fa.LandmarksType._2D if landmark_type == '2D' else fa.LandmarksType._3D
        self.face_alignment = fa.FaceAlignment(
            self.landmark_type, device=device)

        self.plot_style = dict(marker='o',
                               markersize=0,
                               linestyle='-',
                               lw=1)

        self.pred_type = collections.namedtuple(
            'prediction_type', ['slice', 'color'])

        self.pred_types = {'face': self.pred_type(slice(0, 17), (0, 0.5, 0)),
                           'eyebrow1': self.pred_type(slice(17, 22), (1.0, 1.0, 0)),
                           'eyebrow2': self.pred_type(slice(22, 27), (1.0, 1.0, 0)),
                           'nose': self.pred_type(slice(27, 31), (0, 0, 1)),
                           'nostril': self.pred_type(slice(31, 36), (0, 0, 1)),
                           'eye1': self.pred_type(slice(36, 42), (1, 0, 0)),
                           'eye2': self.pred_type(slice(42, 48), (1, 0, 0)),
                           'lips': self.pred_type(slice(48, 60), (0, 1, 1)),
                           'teeth': self.pred_type(slice(60, 68), (0, 1, 1))
                           }

    def __call__(self, input_path):
        '''
        :param input_path:
        '''

        input_image = io.imread(input_path)

        predicted_landmarks = self.face_alignment.get_landmarks_from_image(
            input_image)[-1]

        image_with_landmarks = self.__generate_image_with_landmarks(
            input_image, predicted_landmarks)

        return image_with_landmarks

    def __generate_image_with_landmarks(self, input_image, predicted_landmarks):

        empty_image = np.zeros_like(input_image)

        fig = plt.gcf()
        DPI = fig.get_dpi()
        fig.set_size_inches(
            empty_image.shape[1]/float(DPI), empty_image.shape[0]/float(DPI))

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(empty_image, extent=[
                   0, empty_image.shape[1], empty_image.shape[0], 0])

        for pred_type in self.pred_types.values():
            plt.plot(predicted_landmarks[pred_type.slice, 0],
                     predicted_landmarks[pred_type.slice, 1],
                     color=pred_type.color, **self.plot_style)

        fig.canvas.draw()

        output_image = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        output_image = output_image.reshape(empty_image.shape)

        return output_image


if __name__ == '__main__':
    from skimage import io

    landmark_extractor = LandmarkExtractor(landmark_type='3D', device='cpu')
    input_path = '../MarioNet/data/test.jpg'
    landmarks = landmark_extractor(input_path)
    io.imsave('../MarioNet/data/test_landmarks.jpg', landmarks)

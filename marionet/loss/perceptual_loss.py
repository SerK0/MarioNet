import urllib.request
import shutil
import typing as tp

import torch
import torch.nn as nn
import torchvision

from pathlib import Path

from .feature_map_loss import FeatureMapLoss


class PerceptualLossVGG19(nn.Module):
    """
    Heavily influenced by pytorch example:
    https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    """

    def __init__(self, device) -> None:
        """
        Perceptual loss VGG19 init

        :returns: None
        """
        super(PerceptualLossVGG19, self).__init__()

        vgg_features = torchvision.models.vgg19(pretrained=True).features

        self.relu1_1 = nn.Sequential()
        for i in range(2):
            self.relu1_1.add_module(str(i), vgg_features[i])

        self.relu2_1 = nn.Sequential()
        for i in range(2, 7):
            self.relu2_1.add_module(str(i), vgg_features[i])

        self.relu3_1 = nn.Sequential()
        for i in range(7, 12):
            self.relu3_1.add_module(str(i), vgg_features[i])

        self.relu4_1 = nn.Sequential()
        for i in range(12, 21):
            self.relu4_1.add_module(str(i), vgg_features[i])

        self.relu5_1 = nn.Sequential()
        for i in range(21, 29):
            self.relu5_1.add_module(str(i), vgg_features[i])

        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def forward(
        self, x: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param torch.Tensor x: input image
        :returns: tuple of 5 torch.Tensors - relu1_1, relu2_1, relu3_1, relu4_1 and relu5_1
        """
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu4_1 = self.relu4_1(relu3_1)
        relu5_1 = self.relu5_1(relu4_1)
        return relu1_1, relu2_1, relu3_1, relu4_1, relu5_1


class PerceptualLossVGG_VD_16(nn.Module):
    """
    Based on original author's code available at
    https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_vd_face_fer_dag.py
    https://www.robots.ox.ac.uk/~albanie/pytorch-models.html
    """

    def __init__(self, device) -> None:
        """
        Perceptual loss VGG VD 16 init

        :returns: None
        """
        super(PerceptualLossVGG_VD_16, self).__init__()
        self.meta = {
            "mean": [129.186279296875, 104.76238250732422, 93.59396362304688],
            "std": [1, 1, 1],
            "imageSize": [224, 224, 3],
        }
        self.conv1_1 = nn.Conv2d(
            3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(
            64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv2_1 = nn.Conv2d(
            64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(
            128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv3_1 = nn.Conv2d(
            128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(
            256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv4_1 = nn.Conv2d(
            256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(
            kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False
        )
        self.conv5_1 = nn.Conv2d(
            512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)
        )
        self.relu5_1 = nn.ReLU()

        self.load_checkpoint()

        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def load_checkpoint(self) -> None:
        """
        Checkpoint loading function.

        Checks for the ./vgg_vd.pth file and downloads it if not available.
        After that loads model via self.load_state_dict.

        :returns: None
        """
        checkpoint_path = Path(__file__).parent / "vgg_vd.pth"
        if not checkpoint_path.exists():
            print("Downloading checkpoint file")
            checkpoint_url = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_vd_face_sfew_dag.pth"
            with urllib.request.urlopen(checkpoint_url) as response, open(
                checkpoint_path, "wb"
            ) as checkpoint_file:
                shutil.copyfileobj(response, checkpoint_file)

            print("Download completed")

        # load checkpoint
        # there will be 10 unexpected keys for classification layers, that I deleted
        #   to save GPU memory
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(checkpoint_path), strict=False
        )

        assert not missing_keys
        assert len(unexpected_keys) == 10

    def forward(
        self, data: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param torch.Tensor data: input image
        :returns: tuple of 5 torch.Tensors - relu1_1, relu2_1, relu3_1, relu4_1 and relu5_1
        """
        x1 = self.conv1_1(data)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        return x2, x7, x12, x18, x26


class PerceptualLoss:
    """
    Preceptual loss class.

    Takes reenacted and driver images and runs specified CNN on them.
    After that compares outputed feature maps with L1 loss.
    """

    def __init__(
        self,
        cnn: nn.Module,
        criterion: tp.Optional[
            tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> None:
        """
        :param nn.Module cnn: CNN to parse images
        :returns: None
        """
        self.cnn = cnn
        self.criterion = FeatureMapLoss(criterion)

    def __call__(
        self, reenacted_image: torch.Tensor, driver_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs CNN and compares feature maps.

        :param torch.Tensor reenacted_image: reenacted image (i.e. generator output)
        :param torch.Tensor driver_image: driver image (i.e. ground-truth reenacted image)
        :returns: perceptual loss
        :rtype: torch.Tensor
        """
        reenacted_feature_maps = self.cnn(reenacted_image)
        driver_feature_maps = self.cnn(driver_image)
        return self.criterion(reenacted_feature_maps, driver_feature_maps)

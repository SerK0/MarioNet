import torch
import pytest

from loss.perceptual_loss import (
    PerceptualLossVGG19,
    PerceptualLossVGG_VD_16,
    PerceptualLoss,
)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def input_image(batch_size):
    return torch.rand(batch_size, 3, 224, 224)


@pytest.fixture
def target_image(batch_size):
    return torch.rand(batch_size, 3, 224, 224)


def test_vgg19(input_image):
    model = PerceptualLossVGG19()
    assert all(not param.requires_grad for param in model.parameters())
    model(input_image)


def test_vgg_vd_16(input_image):
    model = PerceptualLossVGG_VD_16()
    assert all(not param.requires_grad for param in model.parameters())
    model(input_image)


def test_perceptual_loss(input_image, target_image):
    cnn = PerceptualLossVGG19()
    criterion = PerceptualLoss(cnn)
    criterion(input_image, target_image)

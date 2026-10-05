import pytest
from caipinference.models.grayscale_resnet50_toy_classifier import gray_classifier
from caipinference.models.rgb_resnet50_colon_classifier import rgb_classifier
from PIL import Image


@pytest.fixture
def image():
    return Image.new(mode="RGB", size=rgb_classifier.metadata.input_size)


@pytest.fixture
def rgb_classifier_results(image):
    return rgb_classifier.inference(image=image, tissue="Colon")


@pytest.fixture
def gray_classifier_results(image):
    return gray_classifier.inference(image=image, tissue="Colon")


def test_softmax_normalization_rgb_classifier(rgb_classifier_results):
    assert pytest.approx(1.0) == sum(rgb_classifier_results)


def test_softmax_normalization_gray_classifier(gray_classifier_results):
    assert pytest.approx(1.0) == sum(gray_classifier_results)

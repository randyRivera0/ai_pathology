"""TensorFlow adapter for the Experiment 9 binary colorectal classifier."""

import os

from dotenv import load_dotenv

from caipinference.models.resnet50_colon_classifier import ResNet50ColonClassifier
from caipinference.models.classifier import ModelMetadata

load_dotenv()

MODEL_PATH_ENV = "AI_PATHOLOGY_RESNET50_MODEL_PATH"
EXPECTED_MODEL_SHA256 = "6228d60d3c067d01a18c4ca118ba58af5292ecb0749efc525ab2352054fd64ae"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Benign colon tissue", "Colon adenocarcinoma")
MODEL_METADATA = ModelMetadata(
    model_id="resnet50-binary-colorectal-final",
    display_name="ResNet50 binary colorectal classifier",
    version="exp-9",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)

model_path = os.getenv(MODEL_PATH_ENV)

rgb_classifier = ResNet50ColonClassifier(
    model_path=model_path,
    sha256=EXPECTED_MODEL_SHA256,
    model_metadata=MODEL_METADATA,
)

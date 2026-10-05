"""TensorFlow adapter for the Experiment 10 grayscale toy classifier."""

import os
from pathlib import Path

from caipinference.models.classifier import ModelMetadata
from caipinference.models.resnet50_colon_classifier import ResNet50ColonClassifier
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH_ENV = "AI_PATHOLOGY_GRAYSCALE_MODEL_PATH"
EXPECTED_MODEL_SHA256 = "213493463227b3b4c8e7b78006e68578866ea76b5f9a93952a0220272d80c62d"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Benign colon tissue", "Colon adenocarcinoma")
MODEL_METADATA = ModelMetadata(
    model_id="resnet50-binary-colorectal-grayscale-toy",
    display_name="Experimental grayscale ResNet50 toy classifier",
    version="exp-10",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)

model_path = os.getenv(MODEL_PATH_ENV, str(Path("checkpoints/grayscale_resnet50_toy.keras")))

gray_classifier = ResNet50ColonClassifier(
    model_path=model_path,
    sha256=EXPECTED_MODEL_SHA256,
    model_metadata=MODEL_METADATA,
)

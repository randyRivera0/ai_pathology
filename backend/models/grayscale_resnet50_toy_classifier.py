"""TensorFlow adapter for the Experiment 10 grayscale toy classifier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from PIL import Image

from backend.models.protocol import ModelArtifactError, ModelMetadata


MODEL_PATH_ENV = "AI_PATHOLOGY_GRAYSCALE_MODEL_PATH"
EXPECTED_MODEL_SHA256 = (
    "213493463227b3b4c8e7b78006e68578866ea76b5f9a93952a0220272d80c62d"
)
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Benign colon tissue", "Colon adenocarcinoma")
MODEL_METADATA = ModelMetadata(
    model_id="resnet50-binary-colorectal-grayscale-toy",
    display_name="Experimental grayscale ResNet50 toy classifier",
    version="exp-10",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)


class GrayscaleResNet50ToyClassifier:
    """Load and run the Experiment 10 grayscale toy artifact."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Configure lazy loading of the verified Experiment 10 artifact."""

        project_root = Path(__file__).resolve().parents[2]
        configured_path = os.getenv(MODEL_PATH_ENV)
        artifact_name = "grayscale_resnet50_toy.keras"
        artifact_candidates = tuple(
            (project_root / "experiments" / "exp-10").rglob(artifact_name)
        )
        default_path = (
            artifact_candidates[0]
            if len(artifact_candidates) == 1
            else project_root / "experiments" / "exp-10" / artifact_name
        )
        self.model_path = Path(model_path or configured_path or default_path).resolve()
        self._model: tf.keras.Model | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def metadata(self) -> ModelMetadata:
        """Return the Experiment 10 model contract."""

        return MODEL_METADATA

    def predict(self, image: Image.Image) -> tuple[float, ...]:
        """Return ordered benign and adenocarcinoma probabilities."""

        model = self._load_model()
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        image_batch = np.asarray(rgb_image, dtype=np.float32)[None, ...]

        with self._predict_lock:
            output = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float64)

        if output.shape != (1, 1):
            raise ModelArtifactError(f"Unexpected prediction shape: {output.shape}")

        adenocarcinoma_probability = float(output[0, 0])
        if not np.isfinite(adenocarcinoma_probability):
            raise ModelArtifactError("The grayscale model returned a non-finite probability.")
        if not 0.0 <= adenocarcinoma_probability <= 1.0:
            raise ModelArtifactError("The grayscale model returned an invalid probability.")

        return (1.0 - adenocarcinoma_probability, adenocarcinoma_probability)

    def _load_model(self) -> tf.keras.Model:
        """Load, verify, and cache the Experiment 10 checkpoint."""

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise ModelArtifactError(
                    f"Grayscale artifact not found. Set {MODEL_PATH_ENV} or place "
                    "grayscale_resnet50_toy.keras under experiments/exp-10."
                )
            if self._sha256(self.model_path) != EXPECTED_MODEL_SHA256:
                raise ModelArtifactError(
                    "Grayscale model SHA-256 does not match the verified artifact."
                )

            model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
                },
                compile=False,
            )
            if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
                raise ModelArtifactError(
                    f"Unexpected grayscale model input shape: {model.input_shape}"
                )
            if tuple(model.output_shape) != (None, 1):
                raise ModelArtifactError(
                    f"Unexpected grayscale model output shape: {model.output_shape}"
                )

            self._model = model
            return model

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calculate an artifact checksum without loading it fully into memory."""

        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

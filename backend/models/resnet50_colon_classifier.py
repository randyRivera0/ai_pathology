"""TensorFlow adapter for the Experiment 3 binary colon classifier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from PIL import Image

from backend.models.protocol import ModelArtifactError, ModelMetadata


MODEL_PATH_ENV = "AI_PATHOLOGY_RESNET50_MODEL_PATH"
EXPECTED_MODEL_SHA256 = (
    "e48e3afcf2521e731fa68ed4f8585cb3c19d864f5d563aae74246cd88a4d86d1"
)
IMAGE_SIZE = (224, 224)
IMAGENET_BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype=np.float32)
CLASS_NAMES = ("Benign colon tissue", "Colon adenocarcinoma")
MODEL_METADATA = ModelMetadata(
    model_id="resnet50-binary-colon-group-aware",
    display_name="ResNet50 binary colon classifier",
    version="exp-3",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)


class ResNet50ColonClassifier:
    """Load and run the group-aware Experiment 3 ResNet50 artifact."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Configure lazy loading without affecting the Experiment 1 adapter."""

        project_root = Path(__file__).resolve().parents[2]
        configured_path = os.getenv(MODEL_PATH_ENV)
        default_path = (
            project_root
            / "notebooks"
            / "final"
            / "resnet50_binary_best.keras"
        )
        self.model_path = Path(model_path or configured_path or default_path).resolve()
        self._model: tf.keras.Model | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def metadata(self) -> ModelMetadata:
        """Return the verified Experiment 3 model contract."""

        return MODEL_METADATA

    def predict(self, image: Image.Image) -> tuple[float, ...]:
        """Return ordered benign and adenocarcinoma probabilities."""

        model = self._load_model()
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        image_array = np.asarray(rgb_image, dtype=np.float32)
        image_batch = self._preprocess(image_array)[None, ...]

        with self._predict_lock:
            output = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float64)

        if output.shape != (1, 1):
            raise ModelArtifactError(f"Unexpected prediction shape: {output.shape}")

        adenocarcinoma_probability = float(output[0, 0])
        if not np.isfinite(adenocarcinoma_probability):
            raise ModelArtifactError("The model returned a non-finite probability.")
        if not 0.0 <= adenocarcinoma_probability <= 1.0:
            raise ModelArtifactError("The model returned an invalid sigmoid probability.")

        return (1.0 - adenocarcinoma_probability, adenocarcinoma_probability)

    def _load_model(self) -> tf.keras.Model:
        """Load, verify, and cache the Experiment 3 checkpoint."""

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise ModelArtifactError(
                    f"Model artifact not found. Set {MODEL_PATH_ENV} or place the "
                    "final checkpoint at "
                    "notebooks/final/resnet50_binary_best.keras."
                )
            if self._sha256(self.model_path) != EXPECTED_MODEL_SHA256:
                raise ModelArtifactError(
                    "Model SHA-256 does not match the verified final artifact."
                )

            model = tf.keras.models.load_model(self.model_path, compile=False)
            if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
                raise ModelArtifactError(
                    f"Unexpected model input shape: {model.input_shape}"
                )
            if tuple(model.output_shape) != (None, 1):
                raise ModelArtifactError(
                    f"Unexpected model output shape: {model.output_shape}"
                )

            self._model = model
            return model

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Apply the explicit Caffe-style preprocessing used during training."""

        bgr_image = np.asarray(image, dtype=np.float32)[..., ::-1]
        return bgr_image - IMAGENET_BGR_MEANS

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calculate an artifact checksum without loading it fully into memory."""

        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

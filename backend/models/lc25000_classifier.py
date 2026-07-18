"""TensorFlow adapter for the Experiment 1 LC25000 classifier."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from backend.models.protocol import ModelArtifactError, ModelMetadata


MODEL_PATH_ENV = "AI_PATHOLOGY_MODEL_PATH"
EXPECTED_MODEL_SHA256 = (
    "b746aef5199588d1d68c2c42717543171cba30ba377dca6aa740fdc09a81e97f"
)
IMAGE_SIZE = (224, 224)
CLASS_NAMES = (
    "Colon adenocarcinoma",
    "Benign colon tissue",
    "Lung adenocarcinoma",
    "Benign lung tissue",
    "Lung squamous cell carcinoma",
)
MODEL_METADATA = ModelMetadata(
    model_id="efficientnet-b7-exp-1",
    display_name="EfficientNetB7",
    version="exp-1",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)


class LC25000Classifier:
    """Load and run the five-class EfficientNetB7 experiment artifact."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Configure the adapter without loading the expensive model immediately."""

        project_root = Path(__file__).resolve().parents[2]
        configured_path = os.getenv(MODEL_PATH_ENV)
        self.model_path = Path(
            model_path or configured_path or project_root / "models" / "model.h5"
        ).resolve()
        self._model: tf.keras.Model | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def metadata(self) -> ModelMetadata:
        """Return the verified Experiment 1 model contract."""

        return MODEL_METADATA

    def predict(self, image: Image.Image) -> tuple[float, ...]:
        """Return ordered softmax scores for one decoded RGB-compatible image."""

        model = self._load_model()
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        image_array = np.asarray(rgb_image, dtype=np.float32)
        image_batch = preprocess_input(np.expand_dims(image_array, axis=0))
        with self._predict_lock:
            output = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float64)

        if output.shape != (1, len(self.metadata.class_names)):
            raise ModelArtifactError(f"Unexpected prediction shape: {output.shape}")

        scores = output[0]
        if not np.isfinite(scores).all():
            raise ModelArtifactError("The model returned non-finite prediction scores.")
        if np.any(scores < 0) or not np.isclose(scores.sum(), 1.0, atol=1e-5):
            raise ModelArtifactError("The model returned invalid softmax probabilities.")

        return tuple(float(score) for score in scores)

    def _load_model(self) -> tf.keras.Model:
        """Load and verify the model once, then reuse it for later predictions."""

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise ModelArtifactError(
                    f"Model artifact not found. Set {MODEL_PATH_ENV} or place the "
                    "Experiment 1 H5 at models/model.h5."
                )

            if self._sha256(self.model_path) != EXPECTED_MODEL_SHA256:
                raise ModelArtifactError(
                    "Model SHA-256 does not match the verified Experiment 1 artifact."
                )

            model = tf.keras.models.load_model(self.model_path, compile=False)
            if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3):
                raise ModelArtifactError(
                    f"Unexpected model input shape: {model.input_shape}"
                )
            if model.output_shape[-1] != len(self.metadata.class_names):
                raise ModelArtifactError(
                    f"Unexpected model output shape: {model.output_shape}"
                )

            self._model = model
            return model

    @staticmethod
    def _sha256(path: Path) -> str:
        """Calculate an artifact checksum without loading the full file into memory."""

        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

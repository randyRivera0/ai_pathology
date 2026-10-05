"""TensorFlow adapter for the Experiment 9 and Experiment 10 binary colorectal classifiers."""

import hashlib
from pathlib import Path
from threading import Lock

import keras
import numpy as np
from caipinference.models.classifier import Classifier, ModelArtifactError, ModelMetadata
from PIL import Image


class ResNet50ColonClassifier(Classifier):
    """Load and run the final Experiment 9 ResNet50 artifact."""

    def __init__(self, model_path: str, model_metadata: ModelMetadata, sha256: str) -> None:
        """Configure lazy loading of the verified Experiment 9 artifact."""

        super().__init__(metadata=model_metadata)
        self.model_path = Path(model_path).resolve()
        self.sha256 = sha256
        self._model: keras.Model | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def inference(self, image: Image.Image, tissue: str) -> tuple[float, ...]:
        """Return ordered benign and adenocarcinoma probabilities."""

        model = self._load_model()
        rgb_image = image.convert("RGB").resize(self.metadata.input_size, Image.Resampling.NEAREST)
        image_batch = np.asarray(rgb_image, dtype=np.float32)[None, ...]

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

    def _load_model(self) -> keras.Model:
        """Load, verify, and cache the Experiment 9 checkpoint."""

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise ModelArtifactError(
                    "Model artifact not found. Set MODEL_PATH_ENV or place the "
                    "final checkpoint under experiments/exp-9 as "
                    "resnet50_final.keras."
                )
            if self._sha256(self.model_path) != self.sha256:
                raise ModelArtifactError(
                    "Model SHA-256 does not match the verified final artifact."
                )

            model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    "preprocess_input": keras.applications.resnet50.preprocess_input,
                },
                compile=False,
            )
            if tuple(model.input_shape[1:]) != (*self.metadata.input_size, 3):
                raise ModelArtifactError(f"Unexpected model input shape: {model.input_shape}")
            if tuple(model.output_shape) != (None, 1):
                raise ModelArtifactError(f"Unexpected model output shape: {model.output_shape}")

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

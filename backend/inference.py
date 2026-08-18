"""Framework-independent image inference service for the frontend."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from backend.models.protocol import ImageClassifier


MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InvalidImageError(ValueError):
    """Report invalid or unsupported uploaded image content."""


@dataclass(frozen=True)
class ClassScore:
    """Associate one display label with its model score."""

    class_name: str
    score: float


@dataclass(frozen=True)
class PredictionResult:
    """Expose model output without leaking TensorFlow objects to the UI."""

    predicted_class: str
    confidence: float
    scores: tuple[ClassScore, ...]


class InferenceService:
    """Decode uploaded bytes and orchestrate one-model inference."""

    def __init__(self, classifier: ImageClassifier | None = None) -> None:
        """Create a service around the supplied or default classifier adapter."""

        if classifier is None:
            from backend.models.resnet50_colon_classifier import (
                ResNet50ColonClassifier,
            )

            classifier = ResNet50ColonClassifier()
        self.classifier = classifier

    def predict(self, image_bytes: bytes) -> PredictionResult:
        """Validate uploaded bytes and return ordered model scores."""

        image = self._decode_image(image_bytes)
        raw_scores = self.classifier.predict(image)
        scores = tuple(
            ClassScore(class_name=class_name, score=score)
            for class_name, score in zip(
                self.classifier.metadata.class_names,
                raw_scores,
                strict=True,
            )
        )
        winner = max(scores, key=lambda item: item.score)
        return PredictionResult(
            predicted_class=winner.class_name,
            confidence=winner.score,
            scores=scores,
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> Image.Image:
        """Decode one bounded PNG or JPEG payload and detach it from its stream."""

        if not image_bytes:
            raise InvalidImageError("The selected image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise InvalidImageError("The selected image exceeds the 10 MB limit.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise InvalidImageError("Use a valid PNG or JPEG image.")
                image.load()
                return image.copy()
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise InvalidImageError(
                "The selected image is corrupt or unreadable."
            ) from error

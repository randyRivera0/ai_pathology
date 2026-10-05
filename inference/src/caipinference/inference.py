"""Framework-independent image inference service for the frontend."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from caipinference.models.classifier import Classifier
from caipinference.schemas.prediction_result import ClassScore, PredictionResult

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InvalidImageError(ValueError):
    """Report invalid or unsupported uploaded image content."""


class InferenceService:
    """Decode uploaded bytes and orchestrate one-model inference."""

    def __init__(self, classifier: Classifier) -> None:
        """Create a service around the supplied or default classifier adapter."""

        self.classifier = classifier

    def inference(self, image_bytes: bytes, tissue: str) -> PredictionResult:
        """Validate uploaded bytes and return ordered model scores."""

        image = self._decode_image(image_bytes)
        raw_scores = self.classifier.inference(image, tissue)
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
            metadata=self.classifier.metadata,
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
            raise InvalidImageError("The selected image is corrupt or unreadable.") from error

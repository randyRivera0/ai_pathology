from pydantic import BaseModel
from caipinference.models.classifier import ModelMetadata


class ClassScore(BaseModel):
    """Associate one display label with its model score."""

    class_name: str
    score: float


class PredictionResult(BaseModel):
    """Expose model output without leaking TensorFlow objects to the UI."""

    predicted_class: str
    confidence: float
    scores: tuple[ClassScore, ...]
    metadata: ModelMetadata

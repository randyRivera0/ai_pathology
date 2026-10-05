"""Framework-neutral contracts shared by inference model adapters."""

from abc import ABC, abstractmethod

from PIL import Image
from pydantic import BaseModel


class ModelArtifactError(RuntimeError):
    """Report a missing, incompatible, or unexpected model artifact."""


class ModelMetadata(BaseModel):
    """Describe one model without exposing its inference framework."""

    model_id: str
    display_name: str
    version: str
    class_names: tuple[str, ...]
    input_size: tuple[int, int]


class Classifier(ABC):
    """Define the behavior required by the application inference service."""

    def __init__(self, metadata: ModelMetadata):
        self._metadata = metadata

    @property
    def metadata(self) -> ModelMetadata:
        "Return model metadata."
        return self._metadata

    @abstractmethod
    def inference(self, image: Image.Image, tissue: str) -> tuple[float, ...]:
        """Return ordered scores for one decoded image."""

"""Framework-neutral contracts shared by inference model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class ModelArtifactError(RuntimeError):
    """Report a missing, incompatible, or unexpected model artifact."""


@dataclass(frozen=True)
class ModelMetadata:
    """Describe one model without exposing its inference framework."""

    model_id: str
    display_name: str
    version: str
    class_names: tuple[str, ...]
    input_size: tuple[int, int]


class ImageClassifier(Protocol):
    """Define the behavior required by the application inference service."""

    @property
    def metadata(self) -> ModelMetadata:
        """Return stable metadata and ordered output labels for this model."""

        ...

    def predict(self, image: Image.Image) -> tuple[float, ...]:
        """Return ordered scores for one decoded image."""

        ...

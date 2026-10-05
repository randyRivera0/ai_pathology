from caipinference.models.classifier import Classifier, ModelMetadata
from PIL import Image

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Benign colon tissue", "Colon adenocarcinoma")
MODEL_METADATA = ModelMetadata(
    model_id="mock-classifier",
    display_name="Mock classifier",
    version="",
    class_names=CLASS_NAMES,
    input_size=IMAGE_SIZE,
)


class MockClassifier(Classifier):
    def __init__(self, model_metadata: ModelMetadata) -> None:
        self.model_metadata = model_metadata

    def inference(self, image: Image.Image, tissue: str) -> tuple[float, ...]:
        adenocarcinoma_probability = 0.9
        return (1.0 - adenocarcinoma_probability, adenocarcinoma_probability)

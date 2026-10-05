import io

import pytest
from caipinference.app import app
from caipinference.models.rgb_resnet50_colon_classifier import rgb_classifier
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def image_bytes() -> bytes:
    image = Image.new(mode="RGB", size=rgb_classifier.metadata.input_size)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


def test_api(image_bytes) -> None:
    client = TestClient(app)
    response = client.post(
        url="/predict/rgb",
        data={"tissue": "Colon"},
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    scores = [item["score"] for item in response.json()["scores"]]
    assert sum(scores) == pytest.approx(1.0)

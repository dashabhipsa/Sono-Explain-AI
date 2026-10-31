from fastapi.testclient import TestClient
from PIL import Image
import io

from app.backend.main import app
import app.backend.main as main_module


class FakePredictor:
    checkpoint_epoch = 18

    def predict(self, image):
        import numpy as np
        return {
            "predicted_class": "benign",
            "class_probabilities": {
                "benign": 0.8,
                "malignant": 0.1,
                "normal": 0.1,
            },
            "segmentation_mask": np.zeros((224, 224), dtype=np.uint8),
            "model_version": "E3",
            "checkpoint_epoch": 18,
        }


class FakeService:
    def __init__(self):
        self.predictor = FakePredictor()

    def predict(self, image):
        import base64
        import io
        import numpy as np
        from PIL import Image
        mask = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        return {
            "predicted_class": "benign",
            "class_probabilities": {"benign": 0.8, "malignant": 0.1, "normal": 0.1},
            "segmentation_mask_png_base64": base64.b64encode(buffer.getvalue()).decode(),
            "segmentation_mask_width": 224,
            "segmentation_mask_height": 224,
            "segmentation_threshold": 0.5,
            "model_version": "E3",
            "checkpoint_epoch": 18,
        }


def test_predict_endpoint_contract(monkeypatch):
    monkeypatch.setattr(main_module, "ModelService", lambda *args, **kwargs: FakeService())
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    buffer.seek(0)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("sample.png", buffer.getvalue(), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "benign"
    assert set(body["class_probabilities"]) == {"benign", "malignant", "normal"}
    assert body["segmentation_mask_width"] == 224


def test_predict_rejects_invalid_file(monkeypatch):
    monkeypatch.setattr(main_module, "ModelService", lambda *args, **kwargs: FakeService())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("not-image.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 400

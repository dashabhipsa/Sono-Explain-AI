from PIL import Image
from src.inference.predictor import E3Predictor

def test_checkpoint_inference():
    predictor = E3Predictor("checkpoints/E3/best_model_e3.pth")
    result = predictor.predict(Image.new("RGB", (320, 240), "gray"))
    assert result["segmentation_mask"].shape == (224, 224)
    assert abs(sum(result["class_probabilities"].values()) - 1) < 1e-5
    assert result["predicted_class"] in predictor.classes

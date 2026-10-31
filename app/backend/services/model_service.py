from pathlib import Path
import io, base64
import numpy as np
from PIL import Image
from src.inference.predictor import E3Predictor


class ModelService:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"E3 checkpoint not found: {path}")
        self.predictor = E3Predictor(str(path), device=device)

    def analyze(self, image: Image.Image, explain=True, estimate_uncertainty=True) -> dict:
        result = self.predictor.analyze(image, explain, estimate_uncertainty)
        mask = np.asarray(result.pop("segmentation_mask"), dtype=np.uint8) * 255
        mask_buffer = io.BytesIO()
        Image.fromarray(mask, mode="L").save(mask_buffer, format="PNG")
        result["segmentation_mask_png_base64"] = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
        result["segmentation_mask_width"] = int(mask.shape[1])
        result["segmentation_mask_height"] = int(mask.shape[0])
        if result.get("gradcam") is not None:
            cam = np.clip(result.pop("gradcam") * 255, 0, 255).astype(np.uint8)
            cam_buffer = io.BytesIO()
            Image.fromarray(cam, mode="L").save(cam_buffer, format="PNG")
            result["gradcam_png_base64"] = base64.b64encode(cam_buffer.getvalue()).decode("ascii")
        else:
            result.pop("gradcam", None)
        result["segmentation_threshold"] = 0.5
        result["disclaimer"] = "Research use only. Not intended for diagnosis or clinical decision-making."
        return result

    def predict(self, image: Image.Image) -> dict:
        return self.analyze(image, explain=False, estimate_uncertainty=False)

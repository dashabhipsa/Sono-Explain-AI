from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint_epoch: int | None = None


class PredictionResponse(BaseModel):
    predicted_class: str
    class_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float] = {}
    mc_mean_probabilities: dict[str, float] = {}
    calibrated_confidence: float | None = None
    temperature: float | None = None
    predictive_entropy: float | None = None
    mean_predictive_variance: float | None = None
    segmentation_mask_png_base64: str
    segmentation_mask_width: int
    segmentation_mask_height: int
    gradcam_png_base64: str | None = None
    segmentation_threshold: float = Field(default=0.5)
    model_version: str
    checkpoint_epoch: int | None = None
    disclaimer: str = "Research use only. Not intended for diagnosis or clinical decision-making."

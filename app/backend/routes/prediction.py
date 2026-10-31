from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.backend.schemas.prediction import PredictionResponse
from app.backend.services.model_service import ModelService

router = APIRouter(prefix="/api/v1", tags=["inference"])
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    explain: bool = Form(True),
    estimate_uncertainty: bool = Form(True),
):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Supported formats: JPEG, PNG, BMP, WEBP.")

    try:
        service = router.model_service
        result = (
            service.analyze(image, explain=explain, estimate_uncertainty=estimate_uncertainty)
            if hasattr(service, "analyze") else service.predict(image)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    return result


def configure_model_service(service: ModelService):
    router.model_service = service

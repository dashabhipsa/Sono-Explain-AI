import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routes.prediction import router as prediction_router
from app.backend.routes.prediction import configure_model_service
from app.backend.services.model_service import ModelService

CHECKPOINT_PATH = os.getenv(
    "SONOEXPLAIN_CHECKPOINT",
    "checkpoints/E3/best_model_e3.pth",
)
DEVICE = os.getenv("SONOEXPLAIN_DEVICE", "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_model_service(ModelService(CHECKPOINT_PATH, device=DEVICE))
    yield


app = FastAPI(
    title="SonoExplain AI API",
    description=(
        "Research API for E3 breast ultrasound classification and lesion "
        "segmentation. Not for clinical diagnosis."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("SONOEXPLAIN_CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(prediction_router)


@app.get("/health")
def health():
    service = getattr(prediction_router, "model_service", None)
    if service is None:
        return {"status": "starting", "model": "E3"}
    return {
        "status": "ready",
        "model": "MultiScaleBUSI",
        "checkpoint_epoch": service.predictor.checkpoint_epoch,
    }

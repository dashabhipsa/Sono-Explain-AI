# SonoExplain AI

End-to-end research prototype for breast ultrasound image classification and lesion segmentation using the trained E3 MultiScaleBUSI model. Includes Grad-CAM, Monte Carlo Dropout estimates, temperature-scaled probabilities, FastAPI, Streamlit and downloadable JSON/PDF summaries.

**Research use only. Not validated for diagnosis or clinical decision-making.**

## Requirements

Python 3.10–3.12 is recommended. Create an environment and install:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The trained checkpoint is included at `checkpoints/E3/best_model_e3.pth`. Set `SONOEXPLAIN_CHECKPOINT` to override it. CPU is the default; use `SONOEXPLAIN_DEVICE=cuda` on a compatible CUDA installation.

## Run locally

Terminal 1:

```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
streamlit run app/frontend/Home.py
```

Open `http://localhost:8501` for the application and `http://localhost:8000/docs` for the API documentation.

## Run with Docker Compose

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## API

`POST /api/v1/predict` accepts multipart `file`, optional `explain` (default true) and `estimate_uncertainty` (default true). Supported image formats: JPEG, PNG, BMP and WEBP. Maximum upload size: 10 MB.

Response includes raw and temperature-scaled class probabilities, MC-dropout mean probabilities, predictive entropy, mean predictive variance, predicted segmentation mask and optional Grad-CAM PNG, all with model/checkpoint metadata. Temperature is read from the checkpoint's recorded test-metrics field. For scientific evaluation, calibration should be fitted on validation data only; the stored checkpoint field is retained as supplied and must be audited before public reporting.

Preprocessing follows the training pipeline: RGB conversion, aspect-ratio-preserving padding to 224×224, bilinear interpolation and ImageNet normalization. The mask and Grad-CAM are in padded 224×224 coordinates.

## Tests

```bash
pytest -q
```

Tests cover real checkpoint inference, API contract and invalid uploads. Grad-CAM and MC-dropout should also be validated against representative real images and the original notebook before any research claims are made.


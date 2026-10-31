# SonoExplain AI — Streamlit direct-inference patch

This patch removes the frontend's HTTP dependency on the separate FastAPI server.
It reuses the existing `ModelService` and `E3Predictor` from the project.

## Apply

1. Extract this ZIP into the root of your existing `SonoExplain-AI` project.
2. Allow it to overwrite:
   - `app/frontend/Home.py`
   - `src/inference/predictor.py`
   - `requirements.txt`
3. Keep the existing `app/backend/services/model_service.py`, model architecture,
   checkpoint, and all other project files unchanged.
4. Commit and push the changes to GitHub.

## Streamlit Community Cloud

- Repository: your existing private `SonoExplain-AI` repository
- Branch: `main`
- Main file: `app/frontend/Home.py`
- Python: select Python 3.11 if offered.

The app loads `checkpoints/E3/best_model_e3.pth` by default. If needed, set
`SONOEXPLAIN_CHECKPOINT` in Streamlit Secrets to that repository-relative path.

No `SONOEXPLAIN_API_URL` secret is needed after this patch.

## Important

This patch is syntax-checked, but it has not been run against your full project,
checkpoint, or Streamlit Community Cloud. First deployment may still reveal
memory, dependency, or import issues. Test with de-identified sample images only.

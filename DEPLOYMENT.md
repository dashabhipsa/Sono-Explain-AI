# Public deployment guide

SonoExplain AI can be made publicly accessible with a shareable URL. The current project has two processes: a FastAPI inference API and a Streamlit interface. A straightforward first deployment is:

- **FastAPI backend:** Render Web Service
- **Streamlit frontend:** Streamlit Community Cloud
- **Source control:** GitHub

This creates a public URL for the frontend and a separate public API URL. The frontend calls the API using the `SONOEXPLAIN_API_URL` environment variable.

## Before publishing

1. Use only de-identified, non-sensitive example images. Do not invite people to upload identifiable patient images or medical records.
2. Add request rate limiting, upload throttling and basic abuse protection before opening the endpoint widely.
3. Confirm the checkpoint can load within the hosting provider's memory and disk limits. A successful local CPU run does not guarantee the same result on a small cloud instance.
4. Review the dataset/model license and confirm you have permission to publish the trained checkpoint and code.
5. Keep the research-only disclaimer visible. This is not a diagnostic or treatment tool.

## 1. Push the project to GitHub

Create a repository and upload the project contents. Include the checkpoint only if you have permission to distribute it. If you keep the repository private, confirm your chosen hosting plan supports private repositories.

## 2. Deploy the FastAPI backend on Render

Create a new **Web Service** connected to the GitHub repository.

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Set these environment variables:

- `SONOEXPLAIN_CHECKPOINT=checkpoints/E3/best_model_e3.pth`
- `SONOEXPLAIN_DEVICE=cpu`
- `SONOEXPLAIN_CORS_ORIGINS=*` (only if browser-based cross-origin calls are introduced; the current Streamlit server calls the API server-to-server)

After deployment, copy the backend URL, for example `https://your-api-name.onrender.com`. Test `https://your-api-name.onrender.com/health` and `/docs`.

A free/low-cost service may sleep, have resource limits or cold-start latency. Use an always-on plan for a dependable public demo and check the current provider pricing/limits before choosing a plan.

## 3. Deploy the Streamlit frontend

1. Open Streamlit Community Cloud and sign in with GitHub.
2. Choose **Create app** and select the repository, branch `main`, and entrypoint `app/frontend/Home.py`.
3. In Advanced settings / Secrets, add:

```toml
SONOEXPLAIN_API_URL = "https://your-api-name.onrender.com"
```

4. Deploy and test the generated `https://...streamlit.app` URL.

If the app is deployed from a public GitHub repository, the app is normally public. Check the app's visibility settings before sharing.

## 4. Custom domain and Google discovery

A public Streamlit URL can be shared directly. To use a memorable custom domain, configure a domain with a hosting provider that supports custom domains and set up DNS as instructed by that provider. Search-engine indexing is not guaranteed just because a URL is public. For better discoverability, create a public project landing page with a descriptive title, summary, metadata, links to the app and source repository, and optionally a sitemap; then submit the site in Google Search Console. Do not promise that Google will index it or that it will rank for a particular search.

## 5. Post-deployment checks

- Open the public frontend URL in a private/incognito browser window.
- Check the service status indicator.
- Run a test using a de-identified sample image you are allowed to share.
- Confirm classification, mask, Grad-CAM and uncertainty panels render.
- Test report downloads.
- Check backend logs and memory during inference.
- Verify that uploads are not persisted or logged unintentionally.

## Notes

- The frontend hides internal checkpoint/epoch metadata from end users.
- The API currently returns model metadata for debugging. Do not expose internal details in public marketing material.
- Public availability is not the same as clinical readiness, regulatory clearance, or validation.

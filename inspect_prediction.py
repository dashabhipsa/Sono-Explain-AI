import base64
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://127.0.0.1:8000/api/v1/predict"

IMAGE_PATH = r"C:\Users\Dell\Desktop\malignant (91).png"

with open(IMAGE_PATH, "rb") as f:
    response = requests.post(
        API_URL,
        files={"file": ("ultrasound.png", f, "image/png")},
        data={
            "explain": "true",
            "estimate_uncertainty": "true"
        }
    )

response.raise_for_status()
result = response.json()

for key, filename in [
    ("segmentation_mask_png_base64", "predicted_mask.png"),
    ("gradcam_png_base64", "gradcam.png")
]:
    image_bytes = base64.b64decode(result[key])

    image = Image.open(BytesIO(image_bytes))
    image.save(filename)

    print(f"Saved {filename}: {image.size}")

print("Prediction:", result["predicted_class"])
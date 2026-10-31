from PIL import Image, ImageOps
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def preprocess(image: Image.Image, size: int = 224):
    image = ImageOps.pad(image.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)
    return transforms.Normalize(MEAN, STD)(transforms.ToTensor()(image))

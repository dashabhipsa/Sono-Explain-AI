from pathlib import Path
import gc
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from src.models.multiscale_busi import MultiScaleBUSI


class E3Predictor:
    def __init__(self, checkpoint_path, device="cpu", mc_samples=30):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = checkpoint.get("model_config", checkpoint.get("config", {}))
        self.classes = self.config.get("classes", ["benign", "malignant", "normal"])
        self.image_size = int(self.config.get("image_size", 224))
        self.mean = self.config.get("normalization_mean", [0.485, 0.456, 0.406])
        self.std = self.config.get("normalization_std", [0.229, 0.224, 0.225])
        self.temperature = float(checkpoint.get("test_metrics", {}).get("temperature", 1.0))
        self.mc_samples = int(mc_samples)
        self.model = MultiScaleBUSI(num_classes=len(self.classes), pretrained=False, image_size=self.image_size)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.checkpoint_epoch = checkpoint.get("epoch")
        del checkpoint
        gc.collect()
        self.model.to(self.device).eval()

    def preprocess(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = ImageOps.pad(image.convert("RGB"), (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, self.mean, self.std).unsqueeze(0)

    def _gradcam(self, x, class_index):
        activations, gradients = [], []
        layer = self.model.encoder[8]
        def capture_activation(_module, _inputs, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))
        forward_handle = layer.register_forward_hook(capture_activation)
        backward_handle = None
        try:
            self.model.zero_grad(set_to_none=True)
            logits, _, _ = self.model(x)
            logits[0, class_index].backward()
            weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0, 0]
            cam -= cam.min()
            cam /= cam.max().clamp_min(1e-8)
            return cam.detach().cpu().numpy().astype(np.float32)
        finally:
            forward_handle.remove()
            if backward_handle is not None:
                backward_handle.remove()

    def _mc_dropout(self, x):
        dropout_layers = [m for m in self.model.modules() if isinstance(m, nn.Dropout)]
        for layer in dropout_layers:
            layer.train()
        samples = []
        try:
            with torch.no_grad():
                for _ in range(self.mc_samples):
                    logits, _, _ = self.model(x)
                    samples.append(torch.softmax(logits.float() / self.temperature, dim=1))
        finally:
            for layer in dropout_layers:
                layer.eval()
            self.model.eval()
        stacked = torch.stack(samples)
        mean_probs = stacked.mean(dim=0)[0]
        entropy = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum()
        variance = stacked.var(dim=0, unbiased=False).mean()
        return mean_probs.cpu().tolist(), float(entropy.cpu()), float(variance.cpu())

    def analyze(self, image, explain=True, estimate_uncertainty=True):
        x = self.preprocess(image).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, mask_logits, _ = self.model(x)
            raw_probs = torch.softmax(logits.float(), dim=1)[0]
            calibrated_probs = torch.softmax(logits.float() / self.temperature, dim=1)[0]
            mask = (torch.sigmoid(mask_logits.float())[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)
        idx = int(raw_probs.argmax())
        cam = self._gradcam(x, idx) if explain else None
        if estimate_uncertainty:
            mc_probs, entropy, variance = self._mc_dropout(x)
        else:
            mc_probs = calibrated_probs.cpu().tolist()
            p = calibrated_probs.clamp_min(1e-8)
            entropy = float((-(p * p.log()).sum()).cpu())
            variance = 0.0
        return {
            "predicted_class": self.classes[idx],
            "class_probabilities": dict(zip(self.classes, raw_probs.cpu().tolist())),
            "calibrated_probabilities": dict(zip(self.classes, calibrated_probs.cpu().tolist())),
            "mc_mean_probabilities": dict(zip(self.classes, mc_probs)),
            "calibrated_confidence": float(calibrated_probs.max().cpu()),
            "temperature": self.temperature,
            "predictive_entropy": entropy,
            "mean_predictive_variance": variance,
            "segmentation_mask": mask,
            "gradcam": cam,
            "model_version": "E3-MultiScaleBUSI",
            "checkpoint_epoch": self.checkpoint_epoch,
        }

    def predict(self, image):
        return self.analyze(image, explain=False, estimate_uncertainty=False)

"""Shared inference helpers: strict checkpoint loading and batched prediction.

load_variant() rebuilds the architecture from the label maps/config saved in the checkpoint and loads the weights with
strict=True, so a checkpoint that does not match the expected architecture raises instead of predicting from random
or partially loaded weights.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from model import FruitHealthNet  # noqa: E402

HEADS = ("crop", "health", "cause", "condition", "severity")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_variant(variant, which="best", dev=None, run=None):
    """run = checkpoint folder name under models/ (default: the variant itself, e.g. 'image' or 'image_seed1')."""
    path = C.MODELS / (run or variant) / f"{which}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No {which}.pt for variant '{variant}' at {path}. Train it first (src/train.py).")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, maps = ck["config"], ck["label_maps"]
    if cfg["variant"] != variant:
        raise ValueError(f"Checkpoint {path} is for variant {cfg['variant']}, not {variant}")
    if cfg["n_conditions"] != len(maps["condition"]):
        raise ValueError("Checkpoint condition count does not match its label map")
    model = FruitHealthNet(variant, len(maps["condition"]), pretrained=False)
    model.load_state_dict(ck["model"], strict=True)  # raises on any missing/unexpected/mismatched weight
    return model.to(dev or get_device()).eval(), maps, cfg


class GradCAM:
    """Grad-CAM on the last conv block of EfficientNet-B0 (model.features[-1], 1280 x 7 x 7) for the PREDICTED class of a
    head. Create it only after any no-grad forward passes you need; it registers a forward hook on the model."""

    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        model.features[-1].register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        self.acts = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, image, env, head):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            out = self.model(image, env)
            logits = out[head]
            idx = logits.argmax(1)
            logits.gather(1, idx[:, None]).sum().backward()
            w = self.grads.mean((2, 3), keepdim=True)
            cam = F.relu((w * self.acts).sum(1))
            cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach(), idx


def overlay(img224, cam224):
    """Blend an RGB uint8 224x224 image with a [0,1] CAM drawn with the JET colormap."""
    heat = cv2.applyColorMap((cam224 * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (0.55 * img224 + 0.45 * heat).astype(np.uint8)


@torch.no_grad()
def predict_loader(model, loader, dev):
    """Run a DataLoader (batches with 'image' and 'env') and return concatenated float32 logits per head (CPU)."""
    amp = dev.type == "cuda"
    outs = {k: [] for k in HEADS}
    for b in loader:
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            o = model(b["image"].to(dev), b["env"].to(dev))
        for k in HEADS:
            outs[k].append(o[k].float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}

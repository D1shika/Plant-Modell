"""Multi-task model, label maps, masked losses.

Variants: image | env | fusion | fusion_shuffled_env.  (fusion_shuffled_env uses the fusion architecture; the
shuffling of training environment rows is done in train.py, so the architecture is identical to `fusion`.)
Feature dims: image 1280, env 32, fusion 1312 (1280 + 32). Heads are sized to the variant's own feature vector.
Heads: crop(8), health(2), cause(2: Biotic/Abiotic), condition(all crop-condition classes incl. Crop___healthy),
severity (cumulative ordinal: 2 logits = P(sev > Mild), P(sev > Moderate)).
Healthy-condition convention: `Crop___healthy` is a normal class of the condition head.
Cause/severity are masked to unhealthy samples (cause) and valid-heuristic unhealthy samples (severity).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CROPS = ["Tomato", "Apple", "Grape", "Strawberry", "Peach", "Cherry", "Orange", "Pepper_bell"]
HEALTH = ["Healthy", "Unhealthy"]
CAUSE = ["Biotic", "Abiotic"]
SEVERITY = ["Mild", "Moderate", "Severe"]
VARIANTS = ("image", "env", "fusion", "fusion_shuffled_env")
IMG_DIM, ENV_DIM = 1280, 32
IGNORE = -1  # label value meaning "not defined for this sample"


def load_backbone(pretrained=True):
    torch.hub.set_dir(str(Path(__file__).resolve().parent.parent / "models" / "torch_cache"))
    w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.efficientnet_b0(weights=w)
    return m.features, m.avgpool  # features -> (B,1280,h,w); Grad-CAM uses the last conv block of `features`


class FruitHealthNet(nn.Module):
    def __init__(self, variant, n_conditions, pretrained=True):
        super().__init__()
        assert variant in VARIANTS
        self.variant, self.n_conditions = variant, n_conditions
        self.use_img = variant in ("image", "fusion", "fusion_shuffled_env")
        self.use_env = variant in ("env", "fusion", "fusion_shuffled_env")
        if self.use_img:
            self.features, self.pool = load_backbone(pretrained)
        if self.use_env:
            self.env = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        d = (IMG_DIM if self.use_img else 0) + (ENV_DIM if self.use_env else 0)
        self.feat_dim = d
        self.crop = nn.Linear(d, len(CROPS))
        self.health = nn.Linear(d, 2)
        self.cause = nn.Linear(d, 2)
        self.condition = nn.Linear(d, n_conditions)
        self.severity = nn.Linear(d, 2)

    def forward(self, image=None, env=None):
        parts = []
        if self.use_img:
            parts.append(self.pool(self.features(image)).flatten(1))
        if self.use_env:
            parts.append(self.env(env))
        h = torch.cat(parts, 1)
        return {"crop": self.crop(h), "health": self.health(h), "cause": self.cause(h),
                "condition": self.condition(h), "severity": self.severity(h)}

    def param_groups(self, lr_head, lr_backbone, wd):
        bb = list(self.features.parameters()) if self.use_img else []
        ids = {id(p) for p in bb}
        rest = [p for p in self.parameters() if id(p) not in ids]
        g = [{"params": rest, "lr": lr_head, "weight_decay": wd}]
        if bb:
            g.append({"params": bb, "lr": lr_backbone, "weight_decay": wd})
        return g


def masked_ce(logits, target, smoothing=0.1, weight=None):
    """Cross-entropy over samples whose target != IGNORE. Returns (loss, n_valid); loss is 0 (with grad graph) if none."""
    m = target != IGNORE
    n = int(m.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[m], target[m], weight=weight, label_smoothing=smoothing), n


def ordinal_targets(sev):
    """sev in {0,1,2} or IGNORE -> (B,2) float targets [sev>0, sev>1] and validity mask."""
    m = sev != IGNORE
    t = torch.stack([(sev > 0).float(), (sev > 1).float()], 1)
    return t, m


def ordinal_loss(logits, sev):
    t, m = ordinal_targets(sev)
    n = int(m.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[m], t[m]), n


def decode_severity(logits):
    """Consistent decoding: class 0 unless P(>Mild)>0.5; class 2 only if also P(>Moderate)>0.5."""
    p = torch.sigmoid(logits)
    gt1 = (p[:, 0] > 0.5).long()
    return gt1 * (1 + (p[:, 1] > 0.5).long())


def total_loss(out, batch, cond_weight=None, smoothing=0.1):
    parts, counts = {}, {}
    for k in ("crop", "health", "cause"):
        parts[k], counts[k] = masked_ce(out[k], batch[k], smoothing)
    parts["condition"], counts["condition"] = masked_ce(out["condition"], batch["condition"], smoothing, cond_weight)
    parts["severity"], counts["severity"] = ordinal_loss(out["severity"], batch["severity"])
    return sum(parts.values()), parts, counts

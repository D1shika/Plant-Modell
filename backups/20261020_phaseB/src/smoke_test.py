"""Phase E smoke test: forward/backward, masked losses, ordinal loss, checkpoint round-trip, CPU/GPU."""
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BATCH_SIZE, LR_BACKBONE, LR_HEAD, MODELS, SEED, WEIGHT_DECAY  # noqa: E402
from dataset import LeafDataset, build_table, save_label_maps  # noqa: E402
from log_util import log  # noqa: E402
from model import (IGNORE, VARIANTS, FruitHealthNet, decode_severity, masked_ce, ordinal_loss,  # noqa: E402
                   total_loss)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, detail)


torch.manual_seed(SEED)
# ---- masked loss
lg = torch.randn(4, 3, requires_grad=True)
l0, n0 = masked_ce(lg, torch.full((4,), IGNORE))
check("masked CE: all-ignored batch -> loss 0, n=0, finite, backward ok", l0.item() == 0 and n0 == 0)
l0.backward()
t = torch.tensor([0, IGNORE, 2, IGNORE])
l1, n1 = masked_ce(lg, t, smoothing=0.0)
ref = torch.nn.functional.cross_entropy(lg[[0, 2]], t[[0, 2]])
check("masked CE equals CE over valid rows only", n1 == 2 and torch.allclose(l1, ref))
# ---- ordinal loss
sev = torch.tensor([0, 1, 2, IGNORE])
good = torch.tensor([[-10., -10.], [10., -10.], [10., 10.], [5., 5.]])
lo, no = ordinal_loss(good, sev)
check("ordinal loss ~0 for confident correct logits, ignores masked row", no == 3 and lo.item() < 1e-3, f"loss={lo.item():.6f}")
bad = torch.tensor([[10., 10.], [-10., -10.], [-10., -10.], [0., 0.]])
check("ordinal loss large for confident wrong logits", ordinal_loss(bad, sev)[0].item() > 5)
dec = decode_severity(torch.tensor([[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]]))
check("ordinal decode [Mild,Moderate,Severe,Mild(inconsistent->Mild)]", dec.tolist() == [0, 1, 2, 0], str(dec.tolist()))

# ---- data
df, maps = build_table()
save_label_maps(maps)
stats = json.load(open(MODELS / "env_stats.json"))
n_cond = len(maps["condition"])
tr = df[df["split"] == "train"].sample(16, random_state=SEED)
ds = LeafDataset(tr, maps, stats, train=True)
batch = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)))
check("batch shapes", batch["image"].shape == (8, 3, 224, 224) and batch["env"].shape == (8, 4))
check("env normalised with train stats (finite)", torch.isfinite(batch["env"]).all().item())
healthy = batch["health"] == 0
check("cause masked for healthy samples", (batch["cause"][healthy] == IGNORE).all().item() if healthy.any() else True,
      f"healthy in batch={int(healthy.sum())}")

# ---- variants on CPU
for v in VARIANTS:
    m = FruitHealthNet(v, n_cond, pretrained=True)
    exp = {"image": 1280, "env": 32}.get(v, 1312)
    check(f"{v}: feature dim {exp}", m.feat_dim == exp)
    opt = torch.optim.AdamW(m.param_groups(LR_HEAD, LR_BACKBONE, WEIGHT_DECAY))
    out = m(batch["image"][:4], batch["env"][:4])
    b4 = {k: batch[k][:4] for k in ("crop", "health", "cause", "condition", "severity")}
    loss, parts, cnt = total_loss(out, b4)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    g_ok = all(p.grad is not None for p in m.parameters() if p.requires_grad)
    check(f"{v}: CPU forward/backward/step, finite loss, all grads present", torch.isfinite(loss).item() and g_ok,
          f"loss={loss.item():.3f} n={cnt}")

# ---- checkpoint round-trip (fusion)
m = FruitHealthNet("fusion", n_cond, pretrained=True).eval()
(MODELS / "_smoke_test").mkdir(exist_ok=True)  # untrained test checkpoint: kept away from the real models
ck = MODELS / "_smoke_test" / "smoke_ckpt.pt"
torch.save({"model": m.state_dict(), "variant": "fusion", "n_conditions": n_cond, "label_maps": maps}, ck)
m2 = FruitHealthNet("fusion", n_cond, pretrained=False).eval()
m2.load_state_dict(torch.load(ck, map_location="cpu")["model"])
with torch.no_grad():
    a, b = m(batch["image"][:2], batch["env"][:2]), m2(batch["image"][:2], batch["env"][:2])
check("checkpoint save/reload gives identical outputs", all(torch.allclose(a[k], b[k]) for k in a))

# ---- GPU (AMP) step at the configured batch size
if torch.cuda.is_available():
    dev = torch.device("cuda")
    m = FruitHealthNet("fusion", n_cond, pretrained=True).to(dev).train()
    opt = torch.optim.AdamW(m.param_groups(LR_HEAD, LR_BACKBONE, WEIGHT_DECAY))
    scaler = torch.amp.GradScaler("cuda")
    bt = {k: v.to(dev) for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            out = m(bt["image"], bt["env"])
        loss, _, _ = total_loss({k: v.float() for k, v in out.items()}, bt)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    check(f"GPU AMP fusion step batch={len(bt['image'])}", torch.isfinite(loss).item(),
          f"peak VRAM={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, {(time.time() - t0) / 3:.2f}s/step incl. warmup")
else:
    print("CUDA not available -> GPU test skipped (CPU fallback is what ran)")

ok = all(r for _, r in results)
print(f"\n{sum(r for _, r in results)}/{len(results)} checks passed")
log(f"smoke_test: {sum(r for _, r in results)}/{len(results)} passed; failed={[n for n, r in results if not r]}")
sys.exit(0 if ok else 1)

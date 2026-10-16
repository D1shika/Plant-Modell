"""Train one variant (image | env | fusion | fusion_shuffled_env). Resumable: checkpoints every epoch.

  python src/train.py --all                            # whole pipeline: skip finished prep stages, train 4 variants, evaluate
  python src/train.py --variant fusion                 # full run (resumes from models/fusion/last.pt if present)
  python src/train.py --variant fusion --max_steps 40  # timing test only: no checkpoints, nothing marked complete

Outputs in models/<variant>/: last.pt (resumable), best.pt (lowest val loss), history.csv, config.json, DONE.json.
Learning curves: outputs/figures/curves_<variant>.png.
Validation uses real environment values for every variant; for fusion_shuffled_env only the TRAINING environment
rows are permuted across training samples (seed 42), which is the control for 'does the env branch help'.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from dataset import LeafDataset, build_table, save_label_maps  # noqa: E402
from log_util import log  # noqa: E402
from model import VARIANTS, FruitHealthNet, total_loss  # noqa: E402


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def cond_weights(df_train, maps):
    """Inverse-frequency weights (mean 1) if any condition is >3:1 smaller than the largest; else None."""
    cnt = df_train["condition"].value_counts().reindex(maps["condition"]).fillna(0).values.astype(float)
    ratio = cnt.max() / np.maximum(cnt, 1)
    if ratio.max() <= 3.0:
        return None, float(ratio.max())
    w = cnt.max() / np.maximum(cnt, 1)
    return torch.tensor(w / w.mean(), dtype=torch.float32), float(ratio.max())


@torch.no_grad()
def evaluate(model, loader, dev, cw, amp):
    model.eval()
    tot, n_b = 0.0, 0
    hit = {k: [0, 0] for k in ("crop", "health", "cause", "condition")}
    for b in loader:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out = model(b["image"], b["env"])
        out = {k: v.float() for k, v in out.items()}
        loss, _, _ = total_loss(out, b, cw.to(dev) if cw is not None else None, C.LABEL_SMOOTHING)
        tot += loss.item()
        n_b += 1
        for k in hit:
            m = b[k] != -1
            hit[k][0] += int((out[k].argmax(1)[m] == b[k][m]).sum())
            hit[k][1] += int(m.sum())
    acc = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in hit.items()}
    return tot / max(n_b, 1), acc, {k: v[1] for k, v in hit.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, help="train one variant (required unless --all)")
    ap.add_argument("--all", action="store_true",
                    help="reproducible pipeline: resume/skip finished preparation stages, train all 4 variants, evaluate")
    ap.add_argument("--skip-eval", action="store_true", help="with --all: stop after training")
    ap.add_argument("--smoke", action="store_true", help="with --all: also run the smoke test first")
    ap.add_argument("--seed", type=int, default=C.SEED,
                    help="training seed (the data split stays fixed at 42); a non-default seed saves to "
                         "models/<variant>_seed<N>/ so the original run is never touched")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--max_steps", type=int, default=0, help="timing test: stop after N train steps, save nothing")
    a = ap.parse_args()
    if a.all:
        from pipeline import run_pipeline
        run_pipeline(skip_eval=a.skip_eval, smoke=a.smoke)
        return
    if a.variant is None:
        ap.error("--variant is required unless --all is given")
    timing = a.max_steps > 0
    v = a.variant
    run = v if a.seed == C.SEED else f"{v}_seed{a.seed}"
    out_dir = C.MODELS / run
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE.json").exists() and not timing:
        print(f"{run}: DONE.json exists -> training already complete; nothing to do")
        return
    seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and C.MIXED_PRECISION
    df, maps = build_table()
    save_label_maps(maps)
    stats = json.load(open(C.MODELS / "env_stats.json"))
    tr_df = df[df["split"] == "train"].reset_index(drop=True)
    va_df = df[df["split"] == "val"].reset_index(drop=True)
    use_img = v != "env"
    env_override = None
    if v == "fusion_shuffled_env":  # permute training env rows across training samples
        perm = np.random.default_rng(a.seed).permutation(len(tr_df))
        mu, sd = np.array(stats["mean"]), np.array(stats["std"])
        env_n = ((tr_df[maps["env_features"]].values - mu) / sd)[perm]
        env_override = env_n
    tr_ds = LeafDataset(tr_df, maps, stats, train=True, env_override=env_override, load_image=use_img)
    va_ds = LeafDataset(va_df, maps, stats, train=False, load_image=use_img)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    g = torch.Generator()
    tr_dl = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, drop_last=True, generator=g, **kw)
    va_dl = DataLoader(va_ds, batch_size=max(a.batch, 32), shuffle=False, **kw)
    cw, ratio = cond_weights(tr_df, maps)
    cfg = {"variant": v, "epochs": a.epochs, "batch": a.batch, "workers": a.workers, "img_size": C.IMG_SIZE, "amp": amp,
           "device": str(dev), "lr_head": C.LR_HEAD, "lr_backbone": C.LR_BACKBONE, "wd": C.WEIGHT_DECAY,
           "label_smoothing": C.LABEL_SMOOTHING, "grad_clip": C.GRAD_CLIP, "patience": C.EARLY_STOP_PATIENCE,
           "seed": a.seed, "run": run, "split_seed": C.SEED, "n_train": len(tr_df), "n_val": len(va_df), "condition_class_weights": cw is not None,
           "max_condition_imbalance_ratio": round(ratio, 2), "n_conditions": len(maps["condition"])}
    if not timing:
        json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)

    model = FruitHealthNet(v, len(maps["condition"]), pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.param_groups(C.LR_HEAD, C.LR_BACKBONE, C.WEIGHT_DECAY))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, bad, hist = 0, float("inf"), 0, []
    last = out_dir / "last.pt"
    if last.exists() and not timing:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["config"]["variant"] != v or ck["label_maps"]["condition"] != maps["condition"]:
            sys.exit("ABORT: last.pt does not match this variant / label maps; not resuming (file preserved)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        set_rng_state(ck["rng"])
        log(f"train {run}: RESUMED from epoch {start} (best val {best:.4f}, patience counter {bad})")
        print(f"resumed from epoch {start}")
    log(f"train {run}: start cfg={json.dumps(cfg)} class_weights_ratio={ratio:.2f}")

    for ep in range(start, a.epochs):
        t0 = time.time()
        model.train()
        g.manual_seed(a.seed + ep)
        run_loss, nb = 0.0, 0
        for i, b in enumerate(tr_dl):
            b = {k: x.to(dev, non_blocking=True) for k, x in b.items()}
            try:
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    out = model(b["image"], b["env"])
                loss, _, _ = total_loss({k: x.float() for k, x in out.items()}, b,
                                        cw.to(dev) if cw is not None else None, C.LABEL_SMOOTHING)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                log(f"train {run}: CUDA OOM at epoch {ep} step {i} batch {a.batch}; stop and report (no silent change)")
                sys.exit("CUDA out of memory: re-run with a smaller --batch (the change will be logged)")
            run_loss += loss.item()
            nb += 1
            if timing and nb == 10:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                t_start = time.time()
            if timing and nb == a.max_steps:
                torch.cuda.synchronize() if dev.type == "cuda" else None
                per = (time.time() - t_start) / (a.max_steps - 10)
                steps = len(tr_dl)
                print(f"TIMING {v}: {per:.3f}s/step at batch {a.batch}, {steps} steps/epoch -> "
                      f"~{per * steps / 60:.1f} min/epoch train (+val), peak VRAM "
                      f"{torch.cuda.max_memory_allocated() / 2**20 if dev.type == 'cuda' else 0:.0f} MiB")
                return
            if i % 200 == 0:
                print(f"  ep{ep} step {i}/{len(tr_dl)} loss {run_loss / nb:.4f}", flush=True)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        vl, acc, ns = evaluate(model, va_dl, dev, cw, amp)
        dur = time.time() - t0
        row = {"epoch": ep, "train_loss": run_loss / nb, "val_loss": vl, "val_health_acc": acc["health"],
               "val_cause_acc": acc["cause"], "val_condition_acc": acc["condition"], "val_crop_acc": acc["crop"],
               "n_cause": ns["cause"], "lr": lr, "seconds": dur}
        hist.append(row)
        improved = vl < best
        if improved:
            best, bad = vl, 0
        else:
            bad += 1
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad, "history": hist, "config": cfg,
              "label_maps": maps, "rng": rng_state()}
        torch.save(ck, out_dir / "last.tmp")
        (out_dir / "last.tmp").replace(last)
        if improved:
            torch.save({"model": model.state_dict(), "config": cfg, "label_maps": maps, "epoch": ep, "val_loss": vl},
                       out_dir / "best.tmp")
            (out_dir / "best.tmp").replace(out_dir / "best.pt")
        pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
        msg = (f"train {run} epoch {ep}: train_loss {run_loss / nb:.4f} val_loss {vl:.4f} health_acc {acc['health']:.4f} "
               f"cause_acc {acc['cause']:.4f}(n={ns['cause']}) cond_acc {acc['condition']:.4f} lr {lr:.2e} "
               f"{dur / 60:.1f} min checkpoint=last{'+best' if improved else ''} patience {bad}/{C.EARLY_STOP_PATIENCE}")
        print(msg, flush=True)
        log(msg)
        if bad >= C.EARLY_STOP_PATIENCE:
            log(f"train {run}: early stopping at epoch {ep}")
            print("early stopping")
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h["epoch"], h["train_loss"], label="train")
    ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set(title=f"{run}: loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(h["epoch"], h["val_health_acc"], label="health")
    ax[1].plot(h["epoch"], h["val_condition_acc"], label="condition")
    ax[1].set(title="validation accuracy", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    (C.OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(C.OUTPUTS / "figures" / f"curves_{run}.png", dpi=120)
    json.dump({"variant": v, "run": run, "seed": a.seed, "epochs_run": len(hist), "best_val_loss": best, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out_dir / "DONE.json", "w"), indent=2)
    log(f"train {run}: COMPLETE epochs_run={len(hist)} best_val_loss={best:.4f}")


if __name__ == "__main__":
    main()

"""Build the image index (path,crop,condition,health,cause,source,acquisition,group_id).

Usage:  python src/prepare_data.py --pilot 40      # small subset -> data/processed/index_pilot.csv
        python src/prepare_data.py                 # full        -> data/processed/index.csv

Conventions: condition = "<Crop>___<Condition>", healthy leaves use "<Crop>___healthy" (kept in the condition head).
Disease/pest -> cause Biotic; healthy -> cause None. No abiotic data is available (user decision, option b).
PlantVillage group_id = original leaf from leaf-map.json; PlantDoc group_id is left empty here
(assigned later by perceptual-hash grouping in split_data.py).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, DATA_RAW, ROOT, SEED, MAX_IMAGES_PER_CONDITION  # noqa: E402
from log_util import log  # noqa: E402

MIN_SIDE = 32
IMG_EXT = {".jpg", ".jpeg", ".png"}

# ---- PlantVillage folder -> (crop, condition-name). Healthy = "healthy".
PV_MAP = {
    "Apple___Apple_scab": ("Apple", "Apple_scab"), "Apple___Black_rot": ("Apple", "Black_rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar_apple_rust"), "Apple___healthy": ("Apple", "healthy"),
    "Cherry_(including_sour)___Powdery_mildew": ("Cherry", "Powdery_mildew"),
    "Cherry_(including_sour)___healthy": ("Cherry", "healthy"),
    "Grape___Black_rot": ("Grape", "Black_rot"), "Grape___Esca_(Black_Measles)": ("Grape", "Esca"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Grape", "Leaf_blight"), "Grape___healthy": ("Grape", "healthy"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Orange", "Haunglongbing"),
    "Peach___Bacterial_spot": ("Peach", "Bacterial_spot"), "Peach___healthy": ("Peach", "healthy"),
    "Pepper,_bell___Bacterial_spot": ("Pepper_bell", "Bacterial_spot"),
    "Pepper,_bell___healthy": ("Pepper_bell", "healthy"),
    "Strawberry___Leaf_scorch": ("Strawberry", "Leaf_scorch"), "Strawberry___healthy": ("Strawberry", "healthy"),
    "Tomato___Bacterial_spot": ("Tomato", "Bacterial_spot"), "Tomato___Early_blight": ("Tomato", "Early_blight"),
    "Tomato___Late_blight": ("Tomato", "Late_blight"), "Tomato___Leaf_Mold": ("Tomato", "Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato___Target_Spot": ("Tomato", "Target_Spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic_virus"), "Tomato___healthy": ("Tomato", "healthy"),
}

# ---- PlantDoc folder -> (crop, condition-name). Anything not listed is EXCLUDED (see PD_EXCLUDED reasons).
PD_MAP = {
    "Apple Scab Leaf": ("Apple", "Apple_scab"), "Apple rust leaf": ("Apple", "Cedar_apple_rust"),
    "Apple leaf": ("Apple", "healthy"),
    "Bell_pepper leaf": ("Pepper_bell", "healthy"),
    "Cherry leaf": ("Cherry", "healthy"), "Peach leaf": ("Peach", "healthy"),
    "Strawberry leaf": ("Strawberry", "healthy"),
    "grape leaf": ("Grape", "healthy"), "grape leaf black rot": ("Grape", "Black_rot"),
    "Tomato leaf": ("Tomato", "healthy"), "Tomato Early blight leaf": ("Tomato", "Early_blight"),
    "Tomato Septoria leaf spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato leaf bacterial spot": ("Tomato", "Bacterial_spot"), "Tomato leaf late blight": ("Tomato", "Late_blight"),
    "Tomato leaf mosaic virus": ("Tomato", "Mosaic_virus"),
    "Tomato leaf yellow virus": ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato mold leaf": ("Tomato", "Leaf_Mold"),
    "Tomato two spotted spider mites leaf": ("Tomato", "Spider_mites"),
}
PD_EXCLUDED = {
    "Bell_pepper leaf spot": "ambiguous: generic 'leaf spot' not verified as Bacterial_spot",
    "Blueberry leaf": "crop not supported", "Corn Gray leaf spot": "crop not supported",
    "Corn leaf blight": "crop not supported", "Corn rust leaf": "crop not supported",
    "Potato leaf early blight": "crop not supported", "Potato leaf late blight": "crop not supported",
    "Raspberry leaf": "crop not supported", "Soyabean leaf": "crop not supported",
    "Squash Powdery mildew leaf": "crop not supported",
}


def pv_leaf_key(fname: str) -> str:
    """'<uuid>___<Orig name>.JPG' -> lower-cased '<orig name>' (leaf-map key)."""
    stem = Path(fname).stem
    return stem.split("___", 1)[1].lower() if "___" in stem else stem.lower()


def discover():
    rows, dropped = [], Counter()
    lm = json.load(open(DATA_RAW / "plantvillage_dl" / "leaf_grouping" / "leaf-map.json", encoding="utf-8"))
    pv_root = DATA_RAW / "plantvillage"
    for folder, (crop, cond) in PV_MAP.items():
        for p in sorted((pv_root / folder).iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            key = pv_leaf_key(p.name)
            hit = lm.get(key)
            gid, leaf_note = "", "no_leaf_map_entry"
            if hit:
                # leaf-map labels the PlantVillage Apple___Black_rot images "Apple_Frogeye Spot"
                # (frogeye leaf spot = leaf symptom of black rot); accepted as an alias.
                names = {folder} | ({"Apple_Frogeye Spot"} if folder == "Apple___Black_rot" else set())
                cls_ids = [h.split(":::") for h in hit if h.split(":::")[0] in names]
                if cls_ids and folder == "Apple___Black_rot":
                    gid, leaf_note = f"PV:{folder}:{cls_ids[0][1]}", "ok_alias_frogeye"
                    cls_ids = []
                if cls_ids:
                    gid, leaf_note = f"PV:{folder}:{cls_ids[0][1]}", "ok"
                elif not gid:
                    leaf_note = "leaf_map_class_mismatch"
            if not gid:  # fallback: original-image name only (NOT leaf-level); phash grouping refines it later
                gid = f"PVK:{folder}:{key}"
            rows.append(dict(path=p, crop=crop, condition=f"{crop}___{cond}", source="PlantVillage",
                             acquisition="lab", group_id=gid, _note=leaf_note, _key=key, _folder=folder))
    pd_root = DATA_RAW / "plantdoc"
    for split in ("train", "test"):
        for cls in sorted(d.name for d in (pd_root / split).iterdir() if d.is_dir()):
            if cls not in PD_MAP:
                dropped[("PlantDoc", cls, PD_EXCLUDED.get(cls, "UNMAPPED"))] += sum(
                    1 for f in (pd_root / split / cls).iterdir() if f.suffix.lower() in IMG_EXT)
                continue
            crop, cond = PD_MAP[cls]
            for p in sorted((pd_root / split / cls).iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    rows.append(dict(path=p, crop=crop, condition=f"{crop}___{cond}", source="PlantDoc",
                                     acquisition="field", group_id="", _note="phash_pending", _key=p.name,
                                     _folder=f"{split}/{cls}"))
    return rows, dropped


def label(row):
    healthy = row["condition"].endswith("___healthy")
    row["health"] = "Healthy" if healthy else "Unhealthy"
    row["cause"] = "None" if healthy else "Biotic"
    return row


def valid(p: Path):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
            im.convert("RGB")
        return (True, "") if min(w, h) >= MIN_SIDE else (False, "smaller_than_32px")
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="max images per (condition, source) for a pilot subset")
    a = ap.parse_args()
    rng = random.Random(SEED)
    rows, dropped = discover()
    rows = [label(r) for r in rows]
    print(f"discovered images (mapped classes only): {len(rows)}")
    print("\nPROPOSED PLANTVILLAGE MAPPING"); [print(f"  {k:55s} -> {v[0]}___{v[1]}") for k, v in PV_MAP.items()]
    print("\nPROPOSED PLANTDOC MAPPING"); [print(f"  {k:40s} -> {v[0]}___{v[1]}") for k, v in PD_MAP.items()]
    print("\nPLANTDOC EXCLUDED"); [print(f"  {k:30s} {v}") for k, v in PD_EXCLUDED.items()]
    print("\nleaf-map lookup status (PlantVillage target images):", dict(Counter(
        r["_note"] for r in rows if r["source"] == "PlantVillage")))
    print("\nTEN EXAMPLES of PlantVillage group-id extraction (filename -> key -> group_id):")
    pv = sorted((r for r in rows if r["source"] == "PlantVillage"), key=lambda r: str(r["path"]))
    for r in random.Random(SEED).sample(pv, 10):
        print(f"  {Path(r['path']).name!r} -> {r['_key']!r} -> {r['group_id']}  [{r['_note']}]")

    # sampling: pilot cap per (condition, source); full run caps per condition, keeping all field images first.
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["source"]) if a.pilot else r["condition"]].append(r)
    chosen = []
    for k, items in sorted(by.items()):
        items.sort(key=lambda r: str(r["path"]))
        cap = a.pilot or MAX_IMAGES_PER_CONDITION
        field = [r for r in items if r["acquisition"] == "field"]
        lab = [r for r in items if r["acquisition"] == "lab"]
        rng.shuffle(lab)
        chosen += field[:cap] + lab[:max(0, cap - len(field[:cap]))]

    kept, drop_reasons = [], Counter()
    for r in chosen:
        ok, why = valid(r["path"])
        if ok:
            kept.append(r)
        else:
            drop_reasons[why] += 1
    print(f"\nsampled {len(chosen)}  valid {len(kept)}  dropped {len(chosen) - len(kept)} {dict(drop_reasons)}")
    out = DATA_PROCESSED / ("index_pilot.csv" if a.pilot else "index.csv")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "crop", "condition", "health", "cause", "source", "acquisition", "group_id"])
        for r in sorted(kept, key=lambda r: str(r["path"])):
            w.writerow([Path(r["path"]).relative_to(ROOT).as_posix(), r["crop"], r["condition"], r["health"],
                        r["cause"], r["source"], r["acquisition"], r["group_id"]])
    print("wrote", out)
    print("\nCrop x cause x acquisition:")
    print(dict(sorted(Counter((r["crop"], r["cause"], r["acquisition"]) for r in kept).items())))
    print("\nCrops with no abiotic examples: ALL (no abiotic data obtained)")
    log(f"prepare_data {'pilot' if a.pilot else 'full'}: discovered={len(rows)} sampled={len(chosen)} valid={len(kept)} dropped={dict(drop_reasons)} out={out.name}")


if __name__ == "__main__":
    main()

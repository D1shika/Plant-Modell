"""Build the 256x256 JPEG (q=95) image cache used for fast training. Resumable: skips existing files."""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, ROOT  # noqa: E402
from dataset import CACHE_DIR, cache_path, load_rgb_256  # noqa: E402
from log_util import log  # noqa: E402


def work(rel):
    cp = cache_path(rel)
    if cp.exists() and cp.stat().st_size > 0:
        return 0
    load_rgb_256(ROOT / rel).save(cp, "JPEG", quality=95)
    return 1


if __name__ == "__main__":
    rels = list(pd.read_csv(DATA_PROCESSED / "splits.csv")["path"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(6) as pool:
        new = sum(pool.imap(work, rels, chunksize=64))
    n = sum(1 for r in rels if cache_path(r).exists())
    size = sum(p.stat().st_size for p in CACHE_DIR.glob("*.jpg")) / 2**20
    print(f"cache: {n}/{len(rels)} present, {new} newly written, {size:.0f} MB")
    log(f"build_cache: {n}/{len(rels)} present ({new} new), {size:.0f} MB in data/processed/cache256")
    sys.exit(0 if n == len(rels) else 1)

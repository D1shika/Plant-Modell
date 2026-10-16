"""Decode every cached image fully to detect truncated/corrupt files. Prints bad files; exit 1 if any."""
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CACHE_DIR  # noqa: E402
from log_util import log  # noqa: E402


def chk(p):
    try:
        with Image.open(p) as im:
            im.load()
            return str(p), im.size == (256, 256) and im.mode == "RGB"
    except Exception:  # noqa: BLE001
        return str(p), False


if __name__ == "__main__":
    files = sorted(CACHE_DIR.glob("*.jpg"))
    with Pool(6) as pool:
        res = pool.map(chk, files, chunksize=64)
    bad = [p for p, ok in res if not ok]
    print(f"checked {len(files)} cache files, bad={len(bad)} {bad[:5]}")
    log(f"verify_cache: checked={len(files)} bad={len(bad)}")
    sys.exit(1 if bad else 0)

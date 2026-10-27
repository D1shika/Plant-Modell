"""Compare current SHA-256 hashes of checkpoints, label maps and processed data with logs/repair/hashes_before_phaseB.txt.
Read-only. Standard library only.  Run:  .venv\\Scripts\\python.exe -I tests\\check_hashes.py
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
before = {}
for line in (ROOT / "logs" / "repair" / "hashes_before_phaseB.txt").read_text().splitlines():
    h, p = line.split(maxsplit=1)
    before[p.lstrip("*")] = h
changed, missing = [], []
for p, h in sorted(before.items()):
    f = ROOT / p
    if not f.exists():
        missing.append(p)
        continue
    d = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    print(("SAME    " if d.hexdigest() == h else "CHANGED ") + p)
    if d.hexdigest() != h:
        changed.append(p)
print(f"\n{len(before)} files checked: {len(before) - len(changed) - len(missing)} identical, "
      f"{len(changed)} changed, {len(missing)} missing")
sys.exit(1 if changed or missing else 0)

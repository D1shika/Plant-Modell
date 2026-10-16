"""Extract PlantDoc files from a shallow git clone whose checkout failed on Windows
(filenames contain '?'). Reads blobs with git and writes them under sanitized names.
Writes a manifest of original -> sanitized path. Safe to rerun (skips files that exist)."""
import csv
import re
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
dest.mkdir(parents=True, exist_ok=True)

out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "HEAD"],
                     capture_output=True, check=True).stdout
entries = [e for e in out.split(b"\0") if e]
seen, rows, written, skipped = {}, [], 0, 0
for e in entries:
    meta, raw_path = e.split(b"\t", 1)
    mode, typ, sha = meta.decode().split()
    if typ != "blob":
        continue
    orig = raw_path.decode("utf-8")
    parts = [re.sub(r'[<>:"|?*\\\x00-\x1f]', "_", p).rstrip(" .") for p in orig.split("/")]
    if any(p in ("", "..") for p in parts):
        print("SKIP unsafe path:", orig)
        continue
    last = parts[-1]
    if len(last) > 80:  # avoid Windows MAX_PATH; keep extension + blob hash for uniqueness
        ext = Path(last).suffix[:8]
        parts[-1] = last[:50] + "_" + sha[:10] + ext
    safe = "/".join(parts)
    if safe.lower() in seen:  # case-insensitive collision (NTFS): disambiguate with blob sha
        p = Path(safe)
        safe = p.with_name(p.stem + "_" + sha[:8] + p.suffix).as_posix()
    seen[safe.lower()] = orig
    target = dest.joinpath(*safe.split("/"))
    if dest not in target.resolve().parents:
        print("SKIP outside dest:", orig)
        continue
    rows.append((orig, safe, sha))
    if target.exists():
        skipped += 1
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", sha],
                          capture_output=True, check=True).stdout
    # verify content against the git blob id before writing
    import hashlib
    h = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert h == sha, f"hash mismatch for {orig}"
    target.write_bytes(data)
    written += 1

with open(dest / "_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["original_path", "sanitized_path", "blob_sha"])
    w.writerows(rows)
print(f"files={len(rows)} written={written} skipped_existing={skipped} renamed={sum(o!=s for o,s,_ in rows)}")

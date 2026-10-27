"""Run every Phase B test script and summarise.  Run:  .venv\\Scripts\\python.exe -I tests\\run_tests.py

test_venv_inference.py is skipped (reported as SKIPPED, not passed) when torch is not importable.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scripts = ["check_hashes.py", "test_reconcile.py", "test_saved_predictions.py", "test_venv_inference.py"]
has_torch = importlib.util.find_spec("torch") is not None
summary = []
for s in scripts:
    if s == "test_venv_inference.py" and not has_torch:
        summary.append((s, "SKIPPED (torch not available in this interpreter)"))
        continue
    print(f"\n######## {s}", flush=True)
    rc = subprocess.run([sys.executable, "-I", str(HERE / s)], cwd=HERE.parent).returncode
    summary.append((s, "PASSED" if rc == 0 else f"FAILED (exit {rc})"))
print("\n######## summary")
for s, r in summary:
    print(f"{s:28s} {r}")
sys.exit(0 if all(r == "PASSED" or r.startswith("SKIPPED") for _, r in summary) else 1)

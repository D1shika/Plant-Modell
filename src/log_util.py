"""Single timestamped, flush-on-write logging helper. Never log secrets."""
from datetime import datetime
from config import LOG_FILE


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
        f.flush()

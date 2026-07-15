import time
import urllib.parse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class RateLimiter:
    """Per-domain rate limiter to avoid being blocked."""

    DOMAIN_DELAYS = {
        "kaggle.com": 3.0,
        "huggingface.co": 2.0,
        "roboflow.com": 2.0,
        "github.com": 1.5,
        "zenodo.org": 2.0,
        "ieee.org": 4.0,
        "earthdata.nasa.gov": 3.0,
        "earthexplorer.usgs.gov": 4.0,
        "paperswithcode.com": 2.0,
        "arxiv.org": 3.5,
        "default": 1.5,
        "ddg": 2.0,
    }

    def __init__(self):
        self._last_request: dict[str, float] = defaultdict(float)

    def wait(self, url_or_key: str) -> None:
        if url_or_key.startswith("http"):
            domain = urllib.parse.urlparse(url_or_key).netloc.replace("www.", "")
        else:
            domain = url_or_key

        delay = self.DOMAIN_DELAYS.get(domain, self.DOMAIN_DELAYS["default"])
        elapsed = time.time() - self._last_request[domain]
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request[domain] = time.time()


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"search_{timestamp}.log"

    logger = logging.getLogger("datasets_explorer")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger

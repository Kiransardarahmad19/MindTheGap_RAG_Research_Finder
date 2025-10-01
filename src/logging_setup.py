import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_dir: str, level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "app.log"

    logger = logging.getLogger()
    logger.setLevel(level.upper())

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level.upper())
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Rotating file
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level.upper())
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Silence very noisy libs (tweak as needed)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.handlers = [ch, fh]

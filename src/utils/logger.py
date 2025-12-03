import logging
from pathlib import Path

def get_logger(name: str, log_file: str = "logs/app.log") -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

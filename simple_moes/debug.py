import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(
        name: str = "SimpleMoEs",
        level: str = "INFO",
        log_file: str | None = None,
        fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Logger already set up
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=20*1024*1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
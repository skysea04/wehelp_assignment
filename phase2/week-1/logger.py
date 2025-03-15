import logging
import os
from datetime import datetime

from constants import BASE_DIR

# Ensure logs directory exists
logs_dir = BASE_DIR / "logs"
os.makedirs(logs_dir, exist_ok=True)


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    in_file: bool = True,
    in_console: bool = True,
):
    """
    Sets up a logger that writes to both console and file.

    Args:
        name (str): Name of the logger
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO)
        in_file (bool): Whether to log to a file
        in_console (bool): Whether to log to console

    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    if in_console:
        c_handler = logging.StreamHandler()
        c_handler.setLevel(log_level)
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

    # File handler - create a new log file with timestamp
    if in_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(logs_dir, f"{name}_{timestamp}.log")
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(log_level)
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


file_log = setup_logger(name="file_log", in_console=False)
stream_log = setup_logger(name="stream_log", in_file=False)

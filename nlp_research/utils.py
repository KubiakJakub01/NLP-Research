"""Utility functions for the project."""
import logging

import coloredlogs

# Set up logging
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
file_handler.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(
    coloredlogs.ColoredFormatter("%(asctime)s %(levelname)s %(message)s")
)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_handler)


def LOG_DEBUG(*args, **kwargs):
    """Log an debug message."""
    logger.debug(*args, **kwargs)


def LOG_INFO(*args, **kwargs):
    """Log an info message."""
    logger.info(*args, **kwargs)


def LOG_WARNING(*args, **kwargs):
    """Log a warning message."""
    logger.warning(*args, **kwargs)


def LOG_ERROR(*args, **kwargs):
    """Log an error message."""
    logger.error(*args, **kwargs)

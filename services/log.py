import logging
from logging import Logger

logger = Logger(__name__)

logging.basicConfig(level=logging.CRITICAL)

# DEBUG - INFO - WARNING - ERROR - CRITICAL

logger.debug("Starting tests...")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")


def debug_print(*args, **kwargs):
    """Conditional debug printing based on environment variable"""
    logger.debug(*args, **kwargs)


def info_print(*args, **kwargs):
    """Conditional info printing based on environment variable"""
    logger.info(*args, **kwargs)


def warning_print(*args, **kwargs):
    """Conditional warning printing based on environment variable"""
    logger.warning(*args, **kwargs)

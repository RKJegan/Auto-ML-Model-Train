"""Utility decorators."""
from __future__ import annotations

import functools
import time

from .logger import get_logger

logger = get_logger()


def log_step(func):
    """Decorator that logs the start and end of a function call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting: {func.__qualname__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Completed: {func.__qualname__} in {elapsed:.2f}s")
        return result
    return wrapper

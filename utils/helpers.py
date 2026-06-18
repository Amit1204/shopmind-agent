"""Misc utility functions shared across the project."""
import time
import functools
from utils.logger import get_logger

logger = get_logger(__name__)


def timeit(func):
    """Decorator: logs execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def confidence_response(result, confidence: float, source_count: int = 0, evidence: list = None):
    """Standard response wrapper every tool should return."""
    return {
        "result": result,
        "confidence": round(confidence, 4),
        "source_count": source_count,
        "evidence": evidence or []
    }

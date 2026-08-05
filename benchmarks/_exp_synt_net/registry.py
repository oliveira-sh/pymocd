import os

import numpy as np
from functools import wraps
from typing import Callable

ALGORITHM_REGISTRY = {}

# Comma-separated algorithm names; empty means "register everything".
_ALG_FILTER = {
    s.strip() for s in os.environ.get("PYMOCD_BENCH_ALGS", "").split(",") if s.strip()
}


def algorithm(
    name: str,
    needs_conversion: bool = True,
    parallel: bool = True,
    max_nodes: int | None = None,
):
    """Register a detector.

    ``max_nodes`` caps the graph sizes the detector is asked to run on; sweeps
    silently skip larger graphs instead of hanging. ``None`` means uncapped.
    """

    def decorator(func: Callable):
        if _ALG_FILTER and name not in _ALG_FILTER:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        ALGORITHM_REGISTRY[name] = {
            "function": wrapper,
            "needs_conversion": needs_conversion,
            "parallel": parallel,
            "max_nodes": max_nodes,
        }
        return wrapper

    return decorator


def _with_seed(func: Callable):
    @wraps(func)
    def wrapper(G, seed=None, *args, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return func(G, *args, **kwargs)

    return wrapper


def _safe(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return {}

    return wrapper

"""Disk-backed caching for RULER synthetic task generation.

Usage::

    @ruler_cached
    def niah_single_1(**kwargs):
        ...
        return {"test": datasets.Dataset.from_list(...)}

Enable by setting ``LM_EVAL_CACHE_DIR`` (e.g. ``/data/lm_eval_cache``).
RULER samples are stored under ``$LM_EVAL_CACHE_DIR/ruler/``.
When unset, the decorator is a zero-cost pass-through.
"""

import functools
import hashlib
import json
import logging

from lm_eval.caching.diskcache import clear, get_cache


eval_logger = logging.getLogger(__name__)

NAMESPACE = "ruler"


def _extract_cache_params(kwargs: dict) -> tuple[str, list[int], bool]:
    tokenizer_name = kwargs.get("tokenizer", kwargs.get("pretrained", ""))
    seq_lengths = kwargs.get("max_seq_lengths", [])
    instruct = kwargs.get("instruct", False)
    return tokenizer_name, sorted(seq_lengths), instruct


def _make_key(task_name: str, tokenizer: str, seq_lengths: list[int], instruct: bool) -> str:
    payload = json.dumps(
        {"task": task_name, "tokenizer": tokenizer, "seq": seq_lengths, "instruct": instruct},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{task_name}:{digest}"


def ruler_cached(fn):
    """Decorator that disk-caches a RULER task generator.

    The wrapped function must have signature ``(**kwargs) -> dict[str, Dataset]``
    where kwargs contains at least ``tokenizer``/``pretrained`` and
    ``max_seq_lengths``.  The function name becomes the cache task key.

    When ``LM_EVAL_CACHE_DIR`` is unset this is a zero-cost pass-through.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs):
        cache = get_cache(NAMESPACE)
        if cache is None:
            return fn(**kwargs)

        tokenizer_name, seq_lengths, instruct = _extract_cache_params(kwargs)
        key = _make_key(fn.__name__, tokenizer_name, seq_lengths, instruct)

        cached = cache.get(key)
        if cached is not None:
            import datasets

            eval_logger.info(f"RULER cache hit: {fn.__name__} ({key})")
            return {"test": datasets.Dataset.from_list(cached, split=datasets.Split.TEST)}

        result = fn(**kwargs)
        samples = [dict(row) for row in result["test"]]
        cache.set(key, samples)
        eval_logger.info(f"RULER cache store: {fn.__name__} ({key}, {len(samples)} samples)")
        return result

    return wrapper


def clear_cache() -> None:
    """Remove all cached RULER data."""
    clear(NAMESPACE)

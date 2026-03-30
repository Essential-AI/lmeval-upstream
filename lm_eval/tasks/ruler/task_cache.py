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

import contextlib
import functools
import hashlib
import json
import logging
from pathlib import Path

from lm_eval.caching.diskcache import clear, get_cache


eval_logger = logging.getLogger(__name__)

NAMESPACE = "ruler"
_RULER_PKG_DIR = Path(__file__).resolve().parent


def _extract_cache_params(kwargs: dict) -> tuple[str, list[int], bool]:
    tokenizer_name = kwargs.get("tokenizer", kwargs.get("pretrained", ""))
    seq_lengths = kwargs.get("max_seq_lengths", [])
    instruct = kwargs.get("instruct", False)
    return tokenizer_name, sorted(seq_lengths), instruct


def _make_key(
    task_name: str,
    tokenizer: str,
    seq_lengths: list[int],
    instruct: bool,
    source_hash: str = "",
) -> str:
    payload = json.dumps(
        {
            "task": task_name,
            "tokenizer": tokenizer,
            "seq": seq_lengths,
            "instruct": instruct,
            "src": source_hash,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{task_name}:{digest}"


def _compute_source_hash(_fn=None) -> str:
    """Hash every ``.py`` file in the RULER task directory.

    Any change to generation code, prompt templates, utility helpers, etc.
    automatically invalidates cached data.
    """
    h = hashlib.sha256()
    for py_file in sorted(_RULER_PKG_DIR.glob("*.py")):
        with contextlib.suppress(OSError):
            h.update(py_file.read_bytes())
    return h.hexdigest()[:16]


def ruler_cached(fn):
    """Decorator that disk-caches a RULER task generator.

    The wrapped function must have signature ``(**kwargs) -> dict[str, Dataset]``
    where kwargs contains at least ``tokenizer``/``pretrained`` and
    ``max_seq_lengths``.  The function name becomes the cache task key.

    The cache key incorporates a hash of the RULER source code so that any
    change to generation logic, prompt templates, or utilities automatically
    invalidates stale entries.

    When ``LM_EVAL_CACHE_DIR`` is unset this is a zero-cost pass-through.
    """
    src_hash = _compute_source_hash(fn)

    @functools.wraps(fn)
    def wrapper(**kwargs):
        cache = get_cache(NAMESPACE)
        if cache is None:
            return fn(**kwargs)

        tokenizer_name, seq_lengths, instruct = _extract_cache_params(kwargs)
        key = _make_key(fn.__name__, tokenizer_name, seq_lengths, instruct, source_hash=src_hash)

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

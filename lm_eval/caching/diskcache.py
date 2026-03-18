"""Shared disk-backed cache for lm_eval.

Set ``LM_EVAL_CACHE_DIR`` to a persistent directory (e.g. a PVC mount at
``/data/lm_eval_cache``) to enable caching.  Each subsystem gets its own
subdirectory automatically::

    /data/lm_eval_cache/
    +-- ruler/          # RULER synthetic task samples
    +-- datasets/       # (future) downloaded dataset artefacts
    +-- ...

When the env var is unset, :func:`get_cache` returns ``None`` and all
callers should treat caching as disabled (zero-cost no-op).

The backing store is :class:`~lm_eval.caching.file_cache.FileCache`, a
flat-file cache that avoids SQLite/flock and is safe on CephFS.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lm_eval.caching.file_cache import FileCache


eval_logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("LM_EVAL_CACHE_DIR", "")
_caches: dict[str, FileCache] = {}


def get_cache(namespace: str):
    """Return a :class:`~lm_eval.caching.file_cache.FileCache` for *namespace*.

    The cache lives at ``$LM_EVAL_CACHE_DIR/<namespace>/``.  Returns
    ``None`` when ``LM_EVAL_CACHE_DIR`` is unset.

    Caches are singletons per namespace for the lifetime of the process.
    """
    if not CACHE_DIR:
        return None

    if namespace in _caches:
        return _caches[namespace]

    from lm_eval.caching.file_cache import FileCache

    directory = os.path.join(CACHE_DIR, namespace)
    cache = FileCache(root=directory)
    _caches[namespace] = cache
    eval_logger.info(f"Disk cache enabled: {directory}")
    return cache


def clear(namespace: str | None = None) -> None:
    """Clear cached data.

    If *namespace* is given, only that subdirectory is cleared.
    Otherwise all known namespaces are cleared.
    """
    if namespace is not None:
        cache = get_cache(namespace)
        if cache is not None:
            cache.clear()
            eval_logger.info(f"Disk cache cleared: {namespace}")
    else:
        for ns in list(_caches):
            _caches[ns].clear()
            eval_logger.info(f"Disk cache cleared: {ns}")

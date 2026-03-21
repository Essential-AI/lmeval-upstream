"""Vendored copy of DiskCache 5.6.3 (https://github.com/grantjenks/python-diskcache).

Only the core Cache class is exposed. See LICENSE in this directory for terms
(Apache 2.0).
"""

from .core import (
    DEFAULT_SETTINGS,
    ENOVAL,
    EVICTION_POLICY,
    UNKNOWN,
    Cache,
    Disk,
    EmptyDirWarning,
    JSONDisk,
    Timeout,
    UnknownFileWarning,
)


__all__ = [
    "Cache",
    "DEFAULT_SETTINGS",
    "Disk",
    "ENOVAL",
    "EVICTION_POLICY",
    "EmptyDirWarning",
    "JSONDisk",
    "Timeout",
    "UNKNOWN",
    "UnknownFileWarning",
]

__title__ = "diskcache"
__version__ = "5.6.3"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2016-2023 Grant Jenks"

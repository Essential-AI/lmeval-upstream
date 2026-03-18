"""Flat-file cache that avoids SQLite / flock – safe on CephFS and similar.

Each entry is a single pickle file addressed by the SHA-256 of its key.
Writes use atomic ``os.replace`` so readers never see partial data.
"""

import contextlib
import hashlib
import os
import pickle
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class FileCache:
    """Key/value store backed by ordinary files.

    Parameters
    ----------
    root : str
        Directory that will hold the cached files.  Created automatically.
    default_ttl : float | None
        Optional time-to-live in seconds.  ``None`` means entries never expire.
    """

    def __init__(self, root: str, default_ttl: float | None = None):
        self.root = Path(root)
        self.default_ttl = default_ttl
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path_for_key(key)
        try:
            with path.open("rb") as f:
                expires_at = pickle.load(f)
                if expires_at is not None and time.time() >= expires_at:
                    return default
                return pickle.load(f)
        except FileNotFoundError:
            return default
        except Exception:
            return default

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        expires_at = None
        ttl = self.default_ttl if ttl is None else ttl
        if ttl is not None:
            expires_at = time.time() + ttl

        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)

        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(expires_at, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
        finally:
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._path_for_key(key).unlink()

    def get_or_set(
        self,
        key: str,
        fn: Callable[[], Any],
        ttl: float | None = None,
    ) -> Any:
        value = self.get(key, default=None)
        if value is not None:
            return value

        value = fn()
        self.set(key, value, ttl=ttl)
        return value

    def has(self, key: str) -> bool:
        sentinel = object()
        return self.get(key, default=sentinel) is not sentinel

    def clear(self) -> None:
        for path in self.root.rglob("*"):
            if path.is_file():
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / digest[2:4] / digest

"""Tests for the FileCache flat-file cache implementation."""

import time

import pytest

from lm_eval.caching.file_cache import FileCache


@pytest.fixture()
def cache(tmp_path):
    return FileCache(root=str(tmp_path / "cache"))


class TestGetSet:
    def test_miss_returns_default(self, cache):
        assert cache.get("nope") is None
        assert cache.get("nope", default=42) == 42

    def test_roundtrip_string(self, cache):
        cache.set("k", "hello")
        assert cache.get("k") == "hello"

    def test_roundtrip_list(self, cache):
        cache.set("ids", [1, 2, 3])
        assert cache.get("ids") == [1, 2, 3]

    def test_roundtrip_dict(self, cache):
        payload = {"a": 1, "nested": {"b": [True, None, 3.14]}}
        cache.set("d", payload)
        assert cache.get("d") == payload

    def test_roundtrip_bytes(self, cache):
        cache.set("bin", b"\x00\xff")
        assert cache.get("bin") == b"\x00\xff"

    def test_overwrite(self, cache):
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"


class TestTTL:
    def test_not_expired(self, cache):
        cache.set("k", "val", ttl=60)
        assert cache.get("k") == "val"

    def test_expired_returns_default(self, cache):
        cache.set("k", "val", ttl=0.01)
        time.sleep(0.05)
        assert cache.get("k") is None
        assert cache.get("k", default="miss") == "miss"

    def test_default_ttl(self, tmp_path):
        cache = FileCache(root=str(tmp_path / "ttl_cache"), default_ttl=0.01)
        cache.set("k", "val")
        time.sleep(0.05)
        assert cache.get("k") is None

    def test_per_key_ttl_overrides_default(self, tmp_path):
        cache = FileCache(root=str(tmp_path / "ttl_cache"), default_ttl=0.01)
        cache.set("k", "val", ttl=60)
        time.sleep(0.05)
        assert cache.get("k") == "val"


class TestDelete:
    def test_delete_existing(self, cache):
        cache.set("k", "v")
        cache.delete("k")
        assert cache.get("k") is None

    def test_delete_missing_is_noop(self, cache):
        cache.delete("nonexistent")


class TestHas:
    def test_has_existing(self, cache):
        cache.set("k", "v")
        assert cache.has("k") is True

    def test_has_missing(self, cache):
        assert cache.has("k") is False


class TestGetOrSet:
    def test_populates_on_miss(self, cache):
        result = cache.get_or_set("k", lambda: 42)
        assert result == 42
        assert cache.get("k") == 42

    def test_returns_cached_on_hit(self, cache):
        cache.set("k", "original")
        call_count = 0

        def expensive():
            nonlocal call_count
            call_count += 1
            return "new"

        result = cache.get_or_set("k", expensive)
        assert result == "original"
        assert call_count == 0


class TestClear:
    def test_clear_removes_all(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_clear_empty_is_noop(self, cache):
        cache.clear()


class TestFanoutStructure:
    def test_creates_two_level_dirs(self, cache):
        cache.set("mykey", "val")
        path = cache._path_for_key("mykey")
        assert path.exists()
        assert path.parent.parent.parent == cache.root


class TestCorruptFile:
    def test_corrupt_file_returns_default(self, cache):
        path = cache._path_for_key("bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not valid pickle data")
        assert cache.get("bad") is None
        assert cache.get("bad", default="fallback") == "fallback"

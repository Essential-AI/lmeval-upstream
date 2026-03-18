"""Tests for the @ruler_cached decorator and disk-cache integration."""

import datasets
import pytest

import lm_eval.caching.diskcache as diskcache_mod
from lm_eval.tasks.ruler.task_cache import (
    _extract_cache_params,
    _make_key,
    clear_cache,
    ruler_cached,
)


def _make_dataset(samples: list[dict]) -> dict[str, datasets.Dataset]:
    return {"test": datasets.Dataset.from_list(samples, split=datasets.Split.TEST)}


FAKE_SAMPLES = [
    {"index": 0, "input": "hello world", "outputs": ["42"], "length": 10, "max_length": 4096, "gen_prefix": "A:"},
    {"index": 1, "input": "foo bar", "outputs": ["7"], "length": 8, "max_length": 4096, "gen_prefix": "A:"},
]


# ---------------------------------------------------------------------------
# Unit tests for key helpers (no disk needed)
# ---------------------------------------------------------------------------

class TestExtractCacheParams:
    def test_extracts_tokenizer(self):
        name, seqs, inst = _extract_cache_params({"tokenizer": "tok/a", "max_seq_lengths": [8192, 4096]})
        assert name == "tok/a"
        assert seqs == [4096, 8192]
        assert inst is False

    def test_falls_back_to_pretrained(self):
        name, _, _ = _extract_cache_params({"pretrained": "model/b"})
        assert name == "model/b"

    def test_tokenizer_takes_precedence(self):
        name, _, _ = _extract_cache_params({"tokenizer": "tok", "pretrained": "pre"})
        assert name == "tok"

    def test_instruct_flag(self):
        _, _, inst = _extract_cache_params({"instruct": True})
        assert inst is True

    def test_defaults(self):
        name, seqs, inst = _extract_cache_params({})
        assert name == ""
        assert seqs == []
        assert inst is False


class TestMakeKey:
    def test_deterministic(self):
        k1 = _make_key("task_a", "tok", [4096, 8192], False)
        k2 = _make_key("task_a", "tok", [4096, 8192], False)
        assert k1 == k2

    def test_different_task_names(self):
        k1 = _make_key("task_a", "tok", [4096], False)
        k2 = _make_key("task_b", "tok", [4096], False)
        assert k1 != k2

    def test_different_tokenizers(self):
        k1 = _make_key("t", "tok_a", [4096], False)
        k2 = _make_key("t", "tok_b", [4096], False)
        assert k1 != k2

    def test_different_seq_lengths(self):
        k1 = _make_key("t", "tok", [4096], False)
        k2 = _make_key("t", "tok", [8192], False)
        assert k1 != k2

    def test_different_instruct(self):
        k1 = _make_key("t", "tok", [4096], False)
        k2 = _make_key("t", "tok", [4096], True)
        assert k1 != k2

    def test_prefix_contains_task_name(self):
        k = _make_key("niah_single_1", "tok", [4096], False)
        assert k.startswith("niah_single_1:")


# ---------------------------------------------------------------------------
# Integration tests: decorator with real disk cache
# ---------------------------------------------------------------------------

@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Point the shared diskcache module at a fresh temp dir and reset singletons."""
    d = str(tmp_path / "lm_eval_cache")
    monkeypatch.setattr(diskcache_mod, "CACHE_DIR", d)
    monkeypatch.setattr(diskcache_mod, "_caches", {})
    yield d
    diskcache_mod._caches = {}


@pytest.fixture()
def no_cache(monkeypatch):
    """Ensure caching is disabled."""
    monkeypatch.setattr(diskcache_mod, "CACHE_DIR", "")
    monkeypatch.setattr(diskcache_mod, "_caches", {})
    yield
    diskcache_mod._caches = {}


class TestRulerCachedDisabled:
    def test_passthrough_when_no_cache_dir(self, no_cache):
        call_count = 0

        @ruler_cached
        def my_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        result = my_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1
        assert len(result["test"]) == 2

        result2 = my_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 2, "Should call through every time when cache is off"

    def test_preserves_function_metadata(self, no_cache):
        @ruler_cached
        def my_named_task(**kwargs):
            """My docstring."""
            return _make_dataset([])

        assert my_named_task.__name__ == "my_named_task"
        assert "My docstring" in my_named_task.__doc__


class TestRulerCachedEnabled:
    def test_first_call_generates_second_call_cached(self, cache_dir):
        call_count = 0

        @ruler_cached
        def gen_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        r1 = gen_task(pretrained="tok/x", max_seq_lengths=[4096])
        assert call_count == 1
        assert len(r1["test"]) == 2

        r2 = gen_task(pretrained="tok/x", max_seq_lengths=[4096])
        assert call_count == 1, "Second call should be served from cache"
        assert len(r2["test"]) == 2

    def test_cached_result_is_equivalent(self, cache_dir):
        @ruler_cached
        def equiv_task(**kwargs):
            return _make_dataset(FAKE_SAMPLES)

        r_fresh = equiv_task(pretrained="tok", max_seq_lengths=[4096])
        r_cached = equiv_task(pretrained="tok", max_seq_lengths=[4096])

        fresh_rows = [dict(row) for row in r_fresh["test"]]
        cached_rows = [dict(row) for row in r_cached["test"]]
        assert fresh_rows == cached_rows

    def test_different_kwargs_are_separate_entries(self, cache_dir):
        call_count = 0

        @ruler_cached
        def multi_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        multi_task(pretrained="tok_a", max_seq_lengths=[4096])
        assert call_count == 1

        multi_task(pretrained="tok_b", max_seq_lengths=[4096])
        assert call_count == 2, "Different tokenizer should miss cache"

        multi_task(pretrained="tok_a", max_seq_lengths=[8192])
        assert call_count == 3, "Different seq_lengths should miss cache"

        multi_task(pretrained="tok_a", max_seq_lengths=[4096])
        assert call_count == 3, "Original params should still be cached"

    def test_instruct_flag_separates_cache(self, cache_dir):
        call_count = 0

        @ruler_cached
        def instruct_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        instruct_task(pretrained="tok", max_seq_lengths=[4096], instruct=False)
        assert call_count == 1

        instruct_task(pretrained="tok", max_seq_lengths=[4096], instruct=True)
        assert call_count == 2, "instruct=True should be a separate cache entry"

    def test_clear_cache_invalidates(self, cache_dir):
        call_count = 0

        @ruler_cached
        def clearable_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        clearable_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1

        clearable_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1

        clear_cache()

        clearable_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 2, "Should regenerate after cache clear"

    def test_cache_persists_across_singleton_reset(self, cache_dir):
        """Simulate a new process by resetting the singleton but keeping the dir."""
        call_count = 0

        @ruler_cached
        def persist_task(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_dataset(FAKE_SAMPLES)

        persist_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1

        diskcache_mod._caches = {}

        persist_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1, "Should still hit disk cache after singleton reset"

    def test_complex_sample_roundtrip(self, cache_dir):
        """Verify that samples with various dtypes survive pickle roundtrip."""
        complex_samples = [
            {
                "index": 0,
                "input": "Some special magic numbers are hidden...\ncontext here",
                "outputs": ["1234567", "8901234"],
                "length": 4000,
                "max_length": 4096,
                "gen_prefix": "The special magic number for foo mentioned in the provided text is",
            },
        ]

        @ruler_cached
        def complex_task(**kwargs):
            return _make_dataset(complex_samples)

        r_fresh = complex_task(pretrained="tok", max_seq_lengths=[4096])
        r_cached = complex_task(pretrained="tok", max_seq_lengths=[4096])

        fresh = dict(r_fresh["test"][0])
        cached = dict(r_cached["test"][0])
        assert fresh == cached

    def test_kwargs_pop_does_not_break_caching(self, cache_dir):
        """The real RULER tasks pop max_seq_lengths from kwargs.
        The decorator must snapshot the value before the function mutates kwargs."""
        call_count = 0

        @ruler_cached
        def popping_task(**kwargs):
            nonlocal call_count
            call_count += 1
            kwargs.pop("max_seq_lengths", None)
            return _make_dataset(FAKE_SAMPLES)

        popping_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1

        popping_task(pretrained="tok", max_seq_lengths=[4096])
        assert call_count == 1, "Cache should still work even though fn pops kwargs"


# ---------------------------------------------------------------------------
# Tests for the shared diskcache layer directly
# ---------------------------------------------------------------------------

class TestSharedDiskCache:
    def test_get_cache_returns_none_when_disabled(self, no_cache):
        from lm_eval.caching.diskcache import get_cache

        assert get_cache("anything") is None

    def test_get_cache_returns_cache_when_enabled(self, cache_dir):
        from lm_eval.caching.diskcache import get_cache

        c = get_cache("test_ns")
        assert c is not None

    def test_namespaces_are_isolated(self, cache_dir):
        from lm_eval.caching.diskcache import get_cache

        c1 = get_cache("ns_a")
        c2 = get_cache("ns_b")
        c1.set("key", "value_a")
        c2.set("key", "value_b")
        assert c1.get("key") == "value_a"
        assert c2.get("key") == "value_b"

    def test_same_namespace_returns_same_instance(self, cache_dir):
        from lm_eval.caching.diskcache import get_cache

        c1 = get_cache("ruler")
        c2 = get_cache("ruler")
        assert c1 is c2

    def test_clear_single_namespace(self, cache_dir):
        from lm_eval.caching.diskcache import clear, get_cache

        c1 = get_cache("ns_a")
        c2 = get_cache("ns_b")
        c1.set("k", 1)
        c2.set("k", 2)

        clear("ns_a")
        assert c1.get("k") is None
        assert c2.get("k") == 2

    def test_clear_all_namespaces(self, cache_dir):
        from lm_eval.caching.diskcache import clear, get_cache

        c1 = get_cache("ns_a")
        c2 = get_cache("ns_b")
        c1.set("k", 1)
        c2.set("k", 2)

        clear()
        assert c1.get("k") is None
        assert c2.get("k") is None

    def test_subdirectory_structure(self, cache_dir):
        import os

        from lm_eval.caching.diskcache import get_cache

        get_cache("ruler")
        get_cache("datasets")
        assert os.path.isdir(os.path.join(cache_dir, "ruler"))
        assert os.path.isdir(os.path.join(cache_dir, "datasets"))

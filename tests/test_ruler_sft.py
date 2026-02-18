"""Tests for RULER prompt template registry.

The prompt template is resolved automatically from the pretrained model name
when ``instruct`` is set in metadata:

    --metadata='{"instruct": true}'
"""

import pytest

from lm_eval.tasks.ruler.instruct_utils import (
    PROMPT_TEMPLATES,
    get_prompt_template,
    maybe_apply_prompt_template,
    register_prompt_template,
    resolve_prompt_template,
)


class TestRegisterPromptTemplate:
    def test_decorator_registers(self):
        @register_prompt_template("org/_test_dummy")
        def _dummy(task_template: str, answer_prefix: str) -> str:
            return f"[{task_template}]"

        assert "org/_test_dummy" in PROMPT_TEMPLATES
        assert PROMPT_TEMPLATES["org/_test_dummy"] is _dummy
        del PROMPT_TEMPLATES["org/_test_dummy"]

    def test_decorator_returns_original_function(self):
        @register_prompt_template("org/_test_identity")
        def _identity(task_template: str, answer_prefix: str) -> str:
            return task_template

        assert callable(_identity)
        del PROMPT_TEMPLATES["org/_test_identity"]


class TestGetPromptTemplate:
    def test_builtin_rnj1(self):
        fn = get_prompt_template("EssentialAI/rnj-1-instruct")
        assert callable(fn)

    def test_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="No prompt template registered"):
            get_prompt_template("nonexistent/model")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="EssentialAI/rnj-1-instruct"):
            get_prompt_template("bad/name")


class TestResolvePromptTemplate:
    def test_returns_none_when_not_instruct(self):
        assert resolve_prompt_template("EssentialAI/rnj-1-instruct", instruct=False) is None

    def test_returns_fn_when_instruct(self):
        fn = resolve_prompt_template("EssentialAI/rnj-1-instruct", instruct=True)
        assert callable(fn)

    def test_raises_for_unknown_model_when_instruct(self):
        with pytest.raises(ValueError, match="No prompt template registered"):
            resolve_prompt_template("unknown/model", instruct=True)


class TestRnj1InstructTemplate:
    def test_wraps_input_and_prefix(self):
        fn = get_prompt_template("EssentialAI/rnj-1-instruct")
        new_input = fn("What is 2+2?", "Answer:")
        assert "<|begin_of_text|>" in new_input
        assert "What is 2+2?" in new_input
        assert "Answer:" in new_input

    def test_strips_trailing_eot(self):
        fn = get_prompt_template("EssentialAI/rnj-1-instruct")
        new_input = fn("hello", "world")
        assert not new_input.endswith("<|eot_id|>")
        assert "world" in new_input

    def test_empty_answer_prefix(self):
        fn = get_prompt_template("EssentialAI/rnj-1-instruct")
        new_input = fn("prompt text", "")
        assert "prompt text" in new_input


class TestMaybeApplyPromptTemplate:
    def test_noop_when_not_instruct(self):
        samples = [
            {"input": "hello", "gen_prefix": "Answer:"},
            {"input": "world", "gen_prefix": "Result:"},
        ]
        original_inputs = [s["input"] for s in samples]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct", instruct=False
        )
        assert [s["input"] for s in result] == original_inputs

    def test_noop_when_instruct_not_provided(self):
        samples = [{"input": "hello", "gen_prefix": "Answer:"}]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct"
        )
        assert result[0]["input"] == "hello"

    def test_rewrites_when_instruct(self):
        samples = [
            {"input": "hello", "gen_prefix": "Answer:"},
            {"input": "world", "gen_prefix": "Result:"},
        ]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct", instruct=True
        )
        for s in result:
            assert "<|begin_of_text|>" in s["input"]
            assert s["gen_prefix"] == ""

    def test_modifies_in_place(self):
        samples = [{"input": "test", "gen_prefix": "A:"}]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct", instruct=True
        )
        assert result is samples

    def test_handles_missing_gen_prefix(self):
        samples = [{"input": "test"}]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct", instruct=True
        )
        assert "<|begin_of_text|>" in result[0]["input"]
        assert result[0]["gen_prefix"] == ""

    def test_preserves_other_fields(self):
        samples = [
            {
                "index": 0,
                "input": "question",
                "gen_prefix": "A:",
                "outputs": ["42"],
                "length": 100,
                "max_length": 4096,
            }
        ]
        result = maybe_apply_prompt_template(
            samples, pretrained="EssentialAI/rnj-1-instruct", instruct=True
        )
        assert result[0]["index"] == 0
        assert result[0]["outputs"] == ["42"]
        assert result[0]["length"] == 100
        assert result[0]["max_length"] == 4096

    def test_unknown_model_raises_when_instruct(self):
        samples = [{"input": "test", "gen_prefix": "A:"}]
        with pytest.raises(ValueError, match="No prompt template registered"):
            maybe_apply_prompt_template(
                samples, pretrained="unknown/model", instruct=True
            )

    def test_ignores_extra_kwargs(self):
        samples = [{"input": "test", "gen_prefix": "A:"}]
        result = maybe_apply_prompt_template(
            samples,
            pretrained="EssentialAI/rnj-1-instruct",
            instruct=True,
            max_seq_lengths=[4096],
            version=1.0,
        )
        assert "<|begin_of_text|>" in result[0]["input"]

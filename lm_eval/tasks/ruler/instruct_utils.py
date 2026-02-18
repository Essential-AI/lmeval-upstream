"""Prompt template registry for RULER instruct variants.

The RULER paper (Table 6) decomposes the input prompt into a *model template*
(the chat format wrapping) and *task templates* (instruction + context + query).
This module provides a registry of model-specific prompt templates that are
resolved automatically from the pretrained model name when ``instruct`` is set
in metadata::

    --metadata='{"instruct": true}'

Each registered callable has signature::

    (task_template: str, answer_prefix: str) -> (formatted_input: str, new_gen_prefix: str)

To add support for a new model, define a function with that signature and
decorate it with ``@register_prompt_template("OrgName/model-name")``.
"""

from collections.abc import Callable

PromptTemplateFn = Callable[[str, str], tuple[str, str]]
PROMPT_TEMPLATES: dict[str, PromptTemplateFn] = {}


def register_prompt_template(name: str):
    """Decorator that registers a prompt template function under *name*.

    *name* should match the HuggingFace model identifier that users pass as
    ``pretrained`` (e.g. ``"EssentialAI/rnj-1-instruct"``).
    """
    def decorator(fn: PromptTemplateFn) -> PromptTemplateFn:
        PROMPT_TEMPLATES[name] = fn
        return fn
    return decorator


def get_prompt_template(name: str) -> PromptTemplateFn:
    """Look up a registered prompt template by model name."""
    if name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES)) or "(none)"
        raise ValueError(
            f"No prompt template registered for {name!r}. "
            f"Available: {available}. "
            f"Register new templates with @register_prompt_template in instruct_utils.py."
        )
    return PROMPT_TEMPLATES[name]


def resolve_prompt_template(
    pretrained: str, *, instruct: bool = False
) -> PromptTemplateFn | None:
    """Return the prompt template for *pretrained*, or ``None`` if not in
    instruct mode."""
    if not instruct:
        return None
    return get_prompt_template(pretrained)


def maybe_apply_prompt_template(samples: list[dict], **kwargs) -> list[dict]:
    """If ``kwargs["instruct"]`` is truthy, look up the prompt template for the
    model identified by ``kwargs["tokenizer"]`` or ``kwargs["pretrained"]`` and
    rewrite every sample's ``input`` / ``gen_prefix``.

    Accepts the full metadata/dataset kwargs dict so callers can simply forward
    ``**kwargs`` without unpacking.  Modifies the list **in-place** and returns it.
    """
    if not kwargs.get("instruct", False):
        return samples
    pretrained = kwargs.get("tokenizer", kwargs.get("pretrained", ""))
    fn = get_prompt_template(pretrained)
    for sample in samples:
        sample["input"], sample["gen_prefix"] = fn(
            sample["input"], sample.get("gen_prefix", "")
        )
    return samples


# ---------------------------------------------------------------------------
# Built-in prompt templates
# ---------------------------------------------------------------------------

@register_prompt_template("EssentialAI/rnj-1-instruct")
def _rnj_1_instruct(task_template: str, answer_prefix: str) -> tuple[str, str]:
    _TEMPLATE = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are rnj-1, a foundation model trained by Essential AI.\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "{assistant_message}<|eot_id|>"
    )
    formatted = _TEMPLATE.format(
        user_message=task_template,
        assistant_message=answer_prefix,
    )
    last_eot = formatted.rfind("<|eot_id|>")
    if last_eot != -1:
        formatted = formatted[:last_eot]
    return formatted, ""

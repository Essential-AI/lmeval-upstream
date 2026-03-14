import logging
import re
from functools import cache
from typing import TYPE_CHECKING, Union

from transformers import AutoTokenizer

if TYPE_CHECKING:
    import transformers


eval_logger = logging.getLogger(__name__)

DEFAULT_SEQ_LENGTHS = [4096, 8192, 16384, 32000]

@cache
def get_tokenizer(
    tokenizer=None, pretrained=None, **kwargs
) -> Union["transformers.PreTrainedTokenizer", "transformers.PreTrainedTokenizerFast"]:
    pretrained = tokenizer or pretrained
    assert pretrained, "No tokenizer or pretrained provided."
    eval_logger.info(f"Using tokenizer {pretrained} for synthetic tasks.")
    return AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)


def postprocess_pred(prediction: list[str]) -> list[str]:
    res = []
    for predict_str in prediction:
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()
        res.append(predict_str)

    return res


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    score = max(
        [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
    ) / len(preds)
    return score


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_all(pred, [doc["outputs"]])
    return {str(input_len): score}


def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    input_len = doc["max_length"]
    pred = postprocess_pred(results)
    score = string_match_part(pred, [doc["outputs"]])
    return {str(input_len): score}


def aggregate_metrics(metrics: list[float]) -> float:
    res = [x for x in metrics if x != -1]
    if not res:
        return -1
    return sum(res) / len(res)


def build_metric_list(metadata: dict | None = None) -> list[dict]:
    """Build metric_list entries from max_seq_lengths in metadata.

    Intended to be referenced via `!function` in RULER task YAMLs so the
    metric list is derived at runtime from --metadata rather than hardcoded.
    """
    seq_lengths = (metadata or {}).get("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return [
        {
            "metric": str(length),
            "aggregation": aggregate_metrics,
            "higher_is_better": True,
        }
        for length in seq_lengths
    ]


def build_aggregate_metric_list(metadata: dict | None = None) -> list[dict]:
    """Build aggregate_metric_list entries from max_seq_lengths in metadata.

    Intended to be referenced via `!function` in the RULER group YAML so the
    aggregate metric list is derived at runtime from --metadata.
    """
    seq_lengths = (metadata or {}).get("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return [
        {"metric": str(length), "weight_by_size": False}
        for length in seq_lengths
    ]

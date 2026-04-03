"""Tests for wandb_logger._generate_dataset, specifically around
metrics that are only present on a subset of samples (e.g. RULER
sequence-length metrics like '4096', '8192', etc.).
"""

import math

from lm_eval.loggers.wandb_logger import WandbLogger


def _make_sample(doc_id, seq_len, score):
    """Build a minimal logged-sample dict like evaluator.evaluate produces."""
    return {
        "doc_id": doc_id,
        "target": "needle",
        "arguments": [("context text", )],
        "resps": [["needle"]],
        "filtered_resps": ["needle"],
        "doc": {"input": "...", "outputs": ["needle"], "max_length": seq_len},
        str(seq_len): score,
    }


def _ruler_config(seq_lengths):
    """Build a minimal task config mimicking a RULER task.
    """
    return {
        "output_type": "generate_until",
        "metric_list": [{"metric": str(l)} for l in seq_lengths],
    }


class TestGenerateDatasetSparseMetrics:
    """Regression test: _generate_dataset must tolerate samples that only
    carry their own sequence-length metric key, not all metric keys.
    """

    def test_sparse_metrics_raise_key_error(self):
        """Reproduces the KeyError: '4096' bug reported when logging RULER
        samples to W&B.
        """
        seq_lengths = [4096, 8192]
        config = _ruler_config(seq_lengths)
        data = [
            _make_sample(0, 4096, 0.9),
            _make_sample(1, 8192, 0.8),
        ]

        # Before the fix this raises KeyError because sample 0 has no '8192'
        # key and sample 1 has no '4096' key.
        df = WandbLogger._generate_dataset(None, data, config)

        assert len(df) == 2
        assert df["4096"].iloc[0] == 0.9
        assert math.isnan(df["4096"].iloc[1])
        assert math.isnan(df["8192"].iloc[0])
        assert df["8192"].iloc[1] == 0.8

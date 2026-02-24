"""Tests for RULER variable tracking (vt) and vt_final tasks."""

import random

from lm_eval.tasks.ruler.vt_utils import (
    generate_chains,
    generate_input_output,
    get_dataset,
)


class TestGenerateChains:
    def test_generates_correct_chain_structure(self):
        random.seed(42)
        vars_ret, chains_ret = generate_chains(num_chains=1, num_hops=3)
        assert len(vars_ret) == 1
        assert len(vars_ret[0]) == 4  # num_hops + 1 variables
        assert len(chains_ret) == 1
        assert len(chains_ret[0]) == 4  # 1 initial + 3 hops

    def test_chain_assignments_follow_value_flow(self):
        random.seed(42)
        vars_ret, chains_ret = generate_chains(num_chains=1, num_hops=2)
        chain = chains_ret[0]
        # First: VAR v0 = <number>
        assert "VAR" in chain[0] and "=" in chain[0]
        # Then: VAR v1 = VAR v0, VAR v2 = VAR v1
        for i in range(1, len(chain)):
            assert f"VAR {vars_ret[0][i]}" in chain[i]
            assert f"VAR {vars_ret[0][i-1]}" in chain[i]


class TestGenerateInputOutput:
    def test_vt_returns_all_variables(self):
        random.seed(42)
        input_text, answer = generate_input_output(
            num_noises=5,
            num_chains=1,
            num_hops=3,
            is_icl=False,
            config_key="variable_tracking",
        )
        assert isinstance(answer, list)
        assert len(answer) == 4  # num_hops + 1
        assert "Find all variables" in input_text
        assert "they are:" in input_text

    def test_vt_final_returns_only_last_variable(self):
        random.seed(42)
        input_text, answer = generate_input_output(
            num_noises=5,
            num_chains=1,
            num_hops=3,
            is_icl=False,
            config_key="variable_tracking_final",
        )
        assert isinstance(answer, list)
        assert len(answer) == 1
        assert "final variable" in input_text
        assert "The final variable in the chain is:" in input_text

    def test_vt_final_answer_is_last_in_chain(self):
        random.seed(123)
        _, answer_final = generate_input_output(
            num_noises=5,
            num_chains=1,
            num_hops=4,
            config_key="variable_tracking_final",
        )
        random.seed(123)
        _, answer_all = generate_input_output(
            num_noises=5,
            num_chains=1,
            num_hops=4,
            config_key="variable_tracking",
        )
        assert answer_final[0] == answer_all[-1]


class TestExampleOutputs:
    """Tests that demonstrate the actual output format of each variant."""

    def test_vt_example_shows_all_variables_in_chain(self):
        """variable_tracking: model must predict every variable in the chain."""
        random.seed(99)
        input_text, answer = generate_input_output(
            num_noises=3,
            num_chains=1,
            num_hops=2,
            config_key="variable_tracking",
        )
        # Same seed produces same chain; answer lists all vars: [first, ..., last]
        assert len(answer) == 3  # num_hops + 1
        expected_answer_str = " ".join(answer)

        # Run with: pytest tests/test_ruler_vt.py -s -k "ExampleOutputs"
        print("\n--- variable_tracking (ruler_vt) ---")
        print("Question asks: Find ALL variables assigned the value")
        print(f"Expected answer (all {len(answer)} vars): {answer}")
        print(f"Target string for eval: {expected_answer_str}")
        q_start = input_text.find("Question:")
        print(f"Question: {input_text[q_start:q_start+120]}...")

        assert "Find all variables" in input_text
        assert "they are:" in input_text
        assert all(isinstance(v, str) and len(v) >= 3 for v in answer)

    def test_vt_final_example_shows_only_last_variable(self):
        """variable_tracking_final: model predicts only the final variable."""
        random.seed(99)
        input_text, answer = generate_input_output(
            num_noises=3,
            num_chains=1,
            num_hops=2,
            config_key="variable_tracking_final",
        )
        assert len(answer) == 1
        expected_answer_str = answer[0]

        print("\n--- variable_tracking_final (ruler_vt_final) ---")
        print("Question asks: What is the FINAL variable in the chain?")
        print(f"Expected answer (1 var): {answer}")
        print(f"Target string for eval: {expected_answer_str}")
        q_start = input_text.find("Question:")
        print(f"Question: {input_text[q_start:q_start+120]}...")

        assert "final variable" in input_text
        assert "The final variable in the chain is:" in input_text
        assert isinstance(answer[0], str) and len(answer[0]) >= 3

    def test_both_variants_same_chain_different_answers(self):
        """Same chain, same query value: vt returns all vars, vt_final returns only the last."""
        random.seed(42)
        input_vt, answer_vt = generate_input_output(
            num_noises=4, num_chains=1, num_hops=3, config_key="variable_tracking"
        )
        random.seed(42)
        input_vt_final, answer_vt_final = generate_input_output(
            num_noises=4, num_chains=1, num_hops=3, config_key="variable_tracking_final"
        )

        # Both use the same chain (same seed)
        assert answer_vt_final[0] == answer_vt[-1]
        assert len(answer_vt) == 4
        assert len(answer_vt_final) == 1

        print("\n--- Contrast: same chain, same value ---")
        print(f"variable_tracking answer: {answer_vt}")
        print(f"variable_tracking_final answer: {answer_vt_final}")
        print(f"vt_final's single var is the last var of vt: {answer_vt_final[0] == answer_vt[-1]}")


class TestGetDataset:
    def test_vt_dataset_produces_valid_samples(self):
        from unittest.mock import MagicMock

        def mock_tokenizer(text):
            return MagicMock(input_ids=list(range(max(1, len(text) // 4))))

        tokenizer = MagicMock(side_effect=mock_tokenizer)
        samples = get_dataset(
            tokenizer=tokenizer,
            seq=2000,
            config_key="variable_tracking",
            num_samples=3,
        )
        assert len(samples) == 3
        for s in samples:
            assert "input" in s
            assert "outputs" in s
            assert "gen_prefix" in s
            assert isinstance(s["outputs"], list)
            assert len(s["outputs"]) > 1  # vt returns all vars in chain

    def test_vt_final_dataset_produces_single_var_outputs(self):
        from unittest.mock import MagicMock

        def mock_tokenizer(text):
            return MagicMock(input_ids=list(range(max(1, len(text) // 4))))

        tokenizer = MagicMock(side_effect=mock_tokenizer)
        samples = get_dataset(
            tokenizer=tokenizer,
            seq=2000,
            config_key="variable_tracking_final",
            num_samples=3,
        )
        assert len(samples) == 3
        for s in samples:
            assert len(s["outputs"]) == 1
            assert "final variable" in s["input"]

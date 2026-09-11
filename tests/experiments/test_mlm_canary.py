"""The canary case configuration determines batches without changing supervision."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


def load_canary():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/15-modernbert-mlm-scaling/canary.py"
    )
    spec = importlib.util.spec_from_file_location("mlm_canary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    all_special_ids = [101, 102, 103]
    mask_token_id = 103

    def __len__(self):
        return 30522


def test_cases_change_only_execution_microbatch_at_fixed_scientific_batch():
    canary = load_canary()
    small = canary.CASES["large-batch-micro8"]
    large = canary.CASES["large-batch-micro32"]
    assert canary.CONFIG.tokens_per_update == 32768
    assert small.tokens_per_update == large.tokens_per_update == 131072
    a, b = asdict(small), asdict(large)
    assert a.pop("local_microbatch") == 8
    assert b.pop("local_microbatch") == 32
    assert a == b
    for config in (small, large):
        for devices in (1, 2):
            batch = config.tokens_per_update // 128
            accumulation = batch // (config.local_microbatch * devices)
            assert accumulation * config.local_microbatch * devices == 1024


def test_host_batch_uses_selected_budget_and_preserves_masking_across_microbatches():
    canary = load_canary()
    tokens = np.arange(262144, dtype=np.int32) % 30000 + 104
    first, first_hashes = canary.host_batch(
        tokens, Tokenizer(), 128, 0, canary.CASES["large-batch-micro8"]
    )
    second, second_hashes = canary.host_batch(
        tokens, Tokenizer(), 128, 0, canary.CASES["large-batch-micro32"]
    )
    assert first.inputs.input_ids.shape == second.inputs.input_ids.shape == (1024, 128)
    assert first_hashes == second_hashes
    np.testing.assert_array_equal(first.labels, second.labels)
    _, next_hashes = canary.host_batch(
        tokens, Tokenizer(), 128, 1, canary.CASES["large-batch-micro8"]
    )
    assert next_hashes["tokens_sha256"] == canary.digest(tokens[131072:])
    assert next_hashes["tokens_sha256"] != first_hashes["tokens_sha256"]


def test_sequence_screen_keeps_global_tokens_and_matched_supervision():
    canary = load_canary()
    tokens = np.arange(131072, dtype=np.int32) % 30000 + 104
    original_hashes = set()
    for length in (128, 512, 1024):
        hashes = []
        for microbatch in (2, 4, 8):
            config = canary.CASES[f"length-{length}-micro{microbatch}"]
            assert config.tokens_per_update == 131072
            assert config.lengths == (length,)
            assert length <= canary.MODEL.max_position_embeddings
            for devices in (1, 2):
                assert 131072 % (length * microbatch * devices) == 0
            batch, fingerprint = canary.host_batch(
                tokens, Tokenizer(), length, 0, config
            )
            assert batch.inputs.input_ids.shape == (131072 // length, length)
            hashes.append(fingerprint)
        assert hashes[0] == hashes[1] == hashes[2]
        original_hashes.add(hashes[0]["tokens_sha256"])
    assert len(original_hashes) == 1


def test_long_sequence_micro16_changes_only_execution_batch():
    canary = load_canary()
    small = canary.CASES["length-1024-micro8"]
    large = canary.CASES["length-1024-micro16"]
    a, b = asdict(small), asdict(large)
    assert a.pop("local_microbatch") == 8
    assert b.pop("local_microbatch") == 16
    assert a == b
    assert large.tokens_per_update // (1024 * large.local_microbatch * 2) == 4
    tokens = np.arange(131072, dtype=np.int32) % 30000 + 104
    _, small_hashes = canary.host_batch(tokens, Tokenizer(), 1024, 0, small)
    _, large_hashes = canary.host_batch(tokens, Tokenizer(), 1024, 0, large)
    assert small_hashes == large_hashes


def test_ddp_cases_change_only_sharding_and_preserve_supervision():
    canary = load_canary()
    tokens = np.arange(131072, dtype=np.int32) % 30000 + 104
    for length in (128, 512, 1024):
        for microbatch in (2, 4, 8):
            name = f"length-{length}-micro{microbatch}"
            fsdp, ddp = canary.CASES[name], canary.CASES[f"ddp-{name}"]
            a, b = asdict(fsdp), asdict(ddp)
            assert a.pop("sharding") == "fsdp"
            assert b.pop("sharding") == "ddp"
            assert a == b
            _, x = canary.host_batch(tokens, Tokenizer(), length, 0, fsdp)
            _, y = canary.host_batch(tokens, Tokenizer(), length, 0, ddp)
            assert x == y


def test_scaling_cases_have_fixed_local_or_global_work():
    canary = load_canary()
    for devices in (1, 2, 4):
        weak = canary.CASES[f"weak-ddp-{devices}gpu"]
        strong = canary.CASES["strong-ddp"]
        assert weak.tokens_per_update == 512 * 8 * devices
        assert weak.local_microbatch == strong.local_microbatch == 8
        assert weak.steps_per_length == strong.steps_per_length == 21
        assert strong.tokens_per_update == 131072
        assert strong.tokens_per_update // (512 * 8 * devices) == 32 // devices
        assert strong.repeat_corpus


def test_collective_summary_distinguishes_matrix_gradients_from_scalar_statistics():
    canary = load_canary()
    rows = canary.collective_summary("""
  %a = f32[64,64] all-reduce(%x), metadata={op_name="jit(step)/reshard"}
  %b = f32[] all-reduce(%y), metadata={op_name="jit(step)/while/body/reduce_sum"}
  %c = bf16[15728640] all-reduce-start(%z), metadata={op_name="while/body"}
""")
    assert [r["in_loop"] for r in rows] == [False, True, True]
    assert [r["largest_float_array_elements"] for r in rows] == [4096, 1, 15728640]


def test_tuned_ddp_cases_change_only_physical_microbatch():
    canary = load_canary()
    expected = asdict(canary.CASES["strong-ddp"])
    expected.pop("local_microbatch")
    for microbatch in (8, 16, 32, 64):
        config = canary.CASES[f"tuned-ddp-micro{microbatch}"]
        actual = asdict(config)
        assert actual.pop("local_microbatch") == microbatch
        assert actual == expected
        for devices in (1, 2, 4):
            assert config.tokens_per_update % (512 * microbatch * devices) == 0

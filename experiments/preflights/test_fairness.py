"""Focused numerical checks for corrected benchmark controls, on CPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.preflights.fairness import initialize_torch_reward, scalar_head


def test_xla_gather_backward_includes_remote_query_gradients(monkeypatch):
    import sys
    from types import ModuleType
    from experiments.preflights import accelerator
    from experiments.preflights.fairness import xla_all_gather_with_grad

    xla, core, xm = (ModuleType(name) for name in
                     ("torch_xla", "torch_xla.core", "torch_xla.core.xla_model"))
    xla.core, core.xla_model = core, xm
    for module in (xla, core, xm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(accelerator, "torch_rank", lambda: 1)
    xm.all_gather = lambda value, **kwargs: torch.cat((value + 1, value))
    xm.all_reduce = lambda kind, value, **kwargs: value + 4
    value = torch.tensor([[2.], [3.]], requires_grad=True)
    result = xla_all_gather_with_grad(value)
    torch.testing.assert_close(result, torch.tensor([[3.], [4.], [2.], [3.]]))
    result.backward(torch.arange(4.).reshape(4, 1))
    torch.testing.assert_close(value.grad, torch.tensor([[6.], [7.]]))


@pytest.mark.parametrize("with_bias", [False, True])
def test_xla_reduction_reassigns_mean_before_clipping(monkeypatch, with_bias):
    import sys
    from types import ModuleType
    from experiments.preflights import accelerator
    from experiments.preflights.fairness import XlaGradientSynchronization

    xla, core, xm = (ModuleType(name) for name in
                     ("torch_xla", "torch_xla.core", "torch_xla.core.xla_model"))
    xla.core, core.xla_model = core, xm
    for module in (xla, core, xm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(accelerator, "torch_is_tpu", lambda: True)
    monkeypatch.setattr(accelerator, "torch_world_size", lambda: 2)
    def reduce(kind, values, *, scale, pin_layout):
        assert kind == "sum" and scale == .5 and pin_layout is False
        return (values + (values + 4)) * scale
    xm.all_reduce = reduce
    model = torch.nn.Linear(2, 1, bias=with_bias)
    model.weight.grad = torch.tensor([[1., 2.]])
    if with_bias:
        model.bias.grad = torch.tensor([4.])
    synchronized = []
    def synchronize():
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]))
        assert model.weight.grad._base is None
        if with_bias:
            torch.testing.assert_close(model.bias.grad, torch.tensor([6.]))
            assert model.bias.grad._base is None
        synchronized.append(True)
    monkeypatch.setattr(accelerator, "torch_synchronize", synchronize)
    trainer = XlaGradientSynchronization()
    trainer.args = SimpleNamespace(max_grad_norm=1.)
    trainer.accelerator = SimpleNamespace(
        gradient_state=SimpleNamespace(is_xla_gradients_synced=False))
    norm = trainer._clip_grad_norm(model)
    expected_norm = torch.tensor(61. if with_bias else 25.).sqrt()
    torch.testing.assert_close(norm, expected_norm)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]) / expected_norm)
    if with_bias:
        torch.testing.assert_close(model.bias.grad, torch.tensor([6.]) / expected_norm)
    assert trainer.accelerator.gradient_state.is_xla_gradients_synced
    assert synchronized == [True]


@pytest.mark.parametrize("platform,batch,devices,chunk", [
    ("gpu", 32, 1, 1), ("tpu", 48, 16, None),
])
def test_audio_execution_mode(monkeypatch, tmp_path, platform, batch, devices, chunk):
    from pathlib import Path
    from experiments.preflights import audio_text

    (tmp_path / "manifest.json").write_text(json.dumps({
        "training_presentations": batch * 22, "relevant_documents": {"0": [0]},
    }))
    monkeypatch.setattr(audio_text, "frozen_contract",
                        lambda: SimpleNamespace(model_revision="frozen"))
    job = audio_text._representax_job(checkpoint=Path("/unused/checkpoint"),
                          data_directory=tmp_path, steps=22,
                          seed=7, batch_size=batch, world_size=devices,
                          platform=platform)
    actual = job.training.grad_cache
    assert (None if actual is None else actual.micro_batch_size) == chunk
    assert job.training.global_batch_size == batch


def test_late_checkpoint_changes_only_config(tmp_path, monkeypatch):
    from experiments.preflights import late_interaction
    from experiments.preflights.fairness import prepare_late_checkpoint

    monkeypatch.setattr(late_interaction, "frozen_contract", lambda: SimpleNamespace(
        maximum_query_length=32, maximum_document_length=256))
    source, target = tmp_path / "source", tmp_path / "prepared"
    source.mkdir()
    metadata = source / "config_sentence_transformers.json"
    metadata.write_text(json.dumps({"query_length": 48, "document_length": 300,
                                    "query_prefix": "[Q] "}))
    (source / "model.safetensors").write_bytes(b"unchanged")
    prepare_late_checkpoint(source, target)
    assert json.loads(metadata.read_text())["document_length"] == 300
    assert json.loads((target / metadata.name).read_text()) == {
        "query_length": 32, "document_length": 256, "query_prefix": "[Q] "}
    assert (target / "model.safetensors").resolve() == source / "model.safetensors"
    assert prepare_late_checkpoint(source, target) == target


def test_xla_attention_casts_qk_without_changing_mask(monkeypatch):
    import transformers.integrations.sdpa_attention as sdpa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from experiments.preflights.fairness import enable_xla_mixed_precision_attention

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    observed = []
    def record(module, query, key, value, mask, **kwargs):
        observed.append((query, key, value, mask, kwargs))
        return value, None
    monkeypatch.setattr(sdpa, "sdpa_attention_forward", record)
    try:
        enable_xla_mixed_precision_attention()
        q = torch.randn(1, 2, 3, 4)
        v = q.to(torch.bfloat16)
        mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
        ALL_ATTENTION_FUNCTIONS["sdpa"](None, q, q, v, mask, is_causal=False)
        query, key, value, passed_mask, kwargs = observed[0]
        assert query.dtype == key.dtype == value.dtype == torch.bfloat16
        assert value is v and passed_mask is mask and kwargs == {"is_causal": False}
        torch.testing.assert_close(query, q.bfloat16(), rtol=0, atol=0)
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


@pytest.mark.parametrize("tpu", [False, True])
def test_explicit_xla_autocast_preserves_other_platforms(monkeypatch, tpu):
    from experiments.preflights.fairness import XlaMixedPrecisionTrainer

    class Base:
        def autocast_smart_context_manager(self, **kwargs):
            return ("parent", kwargs)

    class Trainer(XlaMixedPrecisionTrainer, Base):
        pass

    monkeypatch.setenv("PJRT_DEVICE", "TPU" if tpu else "CPU")
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: (args, kwargs))
    result = Trainer().autocast_smart_context_manager(cache_enabled=False)
    expected = (("xla",), {"dtype": torch.bfloat16, "cache_enabled": False}) if tpu else (
        "parent", {"cache_enabled": False})
    assert result == expected


@pytest.mark.parametrize("rounded", [False, True])
def test_shared_reward_head(tmp_path, rounded):
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": 7, "initializer_range": .02,
    }))
    module = torch.nn.Module()
    module.score = torch.nn.Linear(7, 1, bias=False).to(torch.bfloat16 if rounded else torch.float32)
    initialize_torch_reward(module, tmp_path, 7, round_bfloat16=rounded)
    expected = torch.from_numpy(scalar_head(tmp_path, 7))
    if rounded:
        expected = expected.to(torch.bfloat16).float()
    torch.testing.assert_close(module.score.weight, expected, rtol=0, atol=0)
    assert module.score.weight.dtype == torch.float32
    assert np.array_equal(scalar_head(tmp_path, 7), scalar_head(tmp_path, 7))
    assert not np.array_equal(scalar_head(tmp_path, 7), scalar_head(tmp_path, 42))
    optimizer = torch.optim.AdamW(module.parameters())
    module.score.weight.sum().backward()
    optimizer.step()
    assert optimizer.state[module.score.weight]["exp_avg"].dtype == torch.float32


def test_pylate_cached_mean_gradients():
    from pylate import losses
    from experiments.preflights.late_interaction import _pylate_loss

    model = torch.nn.Identity()
    model.do_query_expansion = False
    objective = _pylate_loss(losses, model, "gpu")
    objective.score_mini_batch_size = 2
    torch.manual_seed(71)
    originals = [torch.randn(5, 3, 4), torch.randn(5, 4, 4)]
    masks = [torch.ones(5, x.shape[1], dtype=torch.bool) for x in originals]
    def leaves():
        return [[x.detach().clone().requires_grad_() for x in value.split(2)]
                for value in originals]
    direct, cached = leaves(), leaves()
    expected = objective.calculate_loss(direct, masks, with_backward=False)
    expected.backward()
    observed = objective.calculate_loss_and_cache_gradients(cached, masks)
    torch.testing.assert_close(observed, expected)
    for actual_group, expected_group in zip(objective.cache, direct, strict=True):
        for actual, expected in zip(actual_group, expected_group, strict=True):
            torch.testing.assert_close(actual, expected.grad, rtol=2e-5, atol=2e-5)


def test_pylate_batch_order(monkeypatch):
    import sentence_transformers
    from experiments.preflights.late_interaction import _reference_arguments
    from datasets import Dataset
    monkeypatch.setattr(sentence_transformers, "SentenceTransformerTrainingArguments", SimpleNamespace)
    args = _reference_arguments("/tmp/unused-fairness", max_steps=22,
                                save_steps=11, seed=7, save=False)
    table = Dataset.from_dict({"query": [str(n) for n in range(8)]})
    sampler = args.batch_sampler(table, 4, True)
    assert list(sampler) == [list(range(4)), list(range(4, 8))]

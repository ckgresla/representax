"""Focused numerical checks for corrected benchmark controls, on CPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.preflights.fairness import initialize_torch_reward, scalar_head


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

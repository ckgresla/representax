from __future__ import annotations

from types import SimpleNamespace

import pytest
from experiments.preflights import accelerator
from experiments.preflights.accelerator import (
    data_parallel_job,
    deterministic_tpu_cached_mnr,
    use_fixed_text_padding,
)


@pytest.mark.parametrize(
    ("global_batch", "preferred_micro", "devices", "micro", "accumulation"),
    (
        (256, 32, 16, 16, 1),
        (128, 32, 16, 8, 1),
        (64, 8, 16, 4, 1),
        (256, 6, 16, 4, 4),
    ),
)
def test_data_parallel_job_preserves_global_batch(
    global_batch: int,
    preferred_micro: int,
    devices: int,
    micro: int,
    accumulation: int,
) -> None:
    from representax.config import BatchConfig, TrainingConfig

    job = SimpleNamespace(
        training=TrainingConfig(
            global_batch_size=global_batch,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=preferred_micro),
        ),
        logging=SimpleNamespace(
            accelerator=True,
            model_copy=lambda *, update: SimpleNamespace(**update),
        ),
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    result = data_parallel_job(
        job,
        device_count=devices,
        platform="tpu",
    )

    assert result.training.global_batch_size == global_batch
    assert result.training.mesh.axis_shapes == (devices,)
    assert result.training.batch.micro_batch_size == micro
    assert result.training.batch.gradient_accumulation_steps == accumulation
    assert not result.logging.accelerator


def test_data_parallel_job_rejects_fractional_local_batches() -> None:
    from representax.config import BatchConfig, TrainingConfig

    job = SimpleNamespace(
        training=TrainingConfig(
            global_batch_size=10,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=2),
        ),
        logging=SimpleNamespace(
            accelerator=False,
            model_copy=lambda *, update: SimpleNamespace(**update),
        ),
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    with pytest.raises(ValueError, match="does not divide"):
        data_parallel_job(job, device_count=4)


def test_fixed_text_padding_preserves_other_processing_settings() -> None:
    module = SimpleNamespace(
        processing_kwargs={
            "text": {"padding_side": "right"},
            "audio": {"sampling_rate": 16_000},
        }
    )

    use_fixed_text_padding([module], 256)

    assert module.processing_kwargs == {
        "text": {
            "padding_side": "right",
            "padding": "max_length",
            "truncation": True,
            "max_length": 256,
        },
        "audio": {"sampling_rate": 16_000},
    }


def test_torch_xla_checkpointing_replaces_the_transformers_backend(monkeypatch) -> None:
    marker = object()
    modeling_utils = SimpleNamespace(checkpoint=None)

    def import_module(name: str):
        if name == "transformers.modeling_utils":
            return modeling_utils
        if name == "torch_xla.utils.checkpoint":
            return SimpleNamespace(checkpoint=marker)
        raise AssertionError(name)

    monkeypatch.setattr(accelerator, "import_module", import_module)

    accelerator.install_torch_xla_checkpointing()

    assert modeling_utils.checkpoint is marker


def test_torch_xla_checkpointing_bypasses_the_trainer_wrapper(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        accelerator,
        "install_torch_xla_checkpointing",
        lambda: calls.append("install"),
    )
    model = SimpleNamespace(gradient_checkpointing_enable=calls.append)

    accelerator.enable_torch_xla_checkpointing(model)

    assert calls == ["install", {"use_reentrant": True}]


def test_deterministic_tpu_cached_mnr_disables_rng_snapshots() -> None:
    class Loss:
        def __init__(self, model, **options):
            self.model = model
            self.options = options

        def embed_minibatch(
            self,
            sentence_feature,
            begin,
            end,
            with_grad,
            copy_random_state,
            random_state=None,
        ):
            return copy_random_state, random_state

    class Model:
        def named_modules(self):
            return ()

        def __getitem__(self, _index):
            config = SimpleNamespace(to_dict=lambda: {0: {"dropout": 0.0}})
            return SimpleNamespace(auto_model=SimpleNamespace(config=config))

    loss = deterministic_tpu_cached_mnr(Loss, Model(), scale=20.0)

    assert loss.options == {"scale": 20.0}
    assert loss.embed_minibatch({}, 0, 1, False, True) == (False, None)


def test_deterministic_tpu_cached_mnr_rejects_active_dropout() -> None:
    import torch

    class Model:
        def named_modules(self):
            return (("dropout", torch.nn.Dropout(0.1)),)

        def __getitem__(self, _index):
            return SimpleNamespace(auto_model=None)

    with pytest.raises(RuntimeError, match="active dropout"):
        deterministic_tpu_cached_mnr(object, Model())

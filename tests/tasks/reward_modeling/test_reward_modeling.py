"""Native reward-modeling contracts and objective parity."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.tasks import build_task
from representax.tasks.reward_modeling import (
    BradleyTerryConfig,
    ListwiseRewardBatch,
    ListwiseRewardConfig,
    ListwiseRewardTask,
    PairwiseRewardBatch,
    PairwiseRewardConfig,
    PairwiseRewardTask,
    PlackettLuceConfig,
    PointwiseRewardBatch,
    PointwiseRewardConfig,
    PointwiseRewardLossConfig,
    PointwiseRewardTask,
    ProcessRewardBatch,
    ProcessRewardConfig,
    ProcessRewardLossConfig,
    ProcessRewardTask,
    bradley_terry_loss,
    plackett_luce_loss,
)
from representax.train import build_train_step, init_train_state


class Inputs(eqx.Module):
    values: jax.Array


class RewardModel(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def logits(self, inputs: Inputs, *, key=None):
        del key
        return inputs.values @ self.weight + self.bias


def model(outputs: int = 1) -> RewardModel:
    return RewardModel(
        weight=jnp.arange(3 * outputs, dtype=jnp.float32).reshape(3, outputs) / 10,
        bias=jnp.linspace(-0.1, 0.1, outputs),
    )


def test_reward_tasks_build_from_the_registry() -> None:
    assert isinstance(
        build_task(PairwiseRewardConfig(), BradleyTerryConfig()),
        PairwiseRewardTask,
    )
    assert build_task(ListwiseRewardConfig(), PlackettLuceConfig()) is not None
    assert isinstance(
        build_task(PointwiseRewardConfig(), PointwiseRewardLossConfig()),
        PointwiseRewardTask,
    )
    assert isinstance(
        build_task(ProcessRewardConfig(), ProcessRewardLossConfig()),
        ProcessRewardTask,
    )


def test_pairwise_reward_matches_trl_formula_and_compiles() -> None:
    candidate = model()
    batch = PairwiseRewardBatch(
        chosen=Inputs(jnp.asarray(((1.0, 0.0, 0.5), (0.2, 1.0, 0.4)))),
        rejected=Inputs(jnp.asarray(((0.0, 0.5, 0.1), (0.8, 0.1, 0.2)))),
        margins=jnp.asarray((0.2, 0.0)),
        valid=jnp.asarray((True, True)),
    )
    task = PairwiseRewardTask(center_rewards_coefficient=0.01)
    chosen = candidate.logits(batch.chosen)[:, 0]
    rejected = candidate.logits(batch.rejected)[:, 0]
    expected = -jnp.mean(jax.nn.log_sigmoid(chosen - rejected - batch.margins))
    expected += 0.01 * jnp.mean(jnp.square(chosen + rejected))
    np.testing.assert_allclose(task.loss(candidate, batch).loss, expected)

    optimizer = optax.sgd(1e-2)
    result = build_train_step(task, optimizer)(
        init_train_state(candidate, optimizer), batch, None
    )
    assert int(result.state.step) == 1
    assert np.isfinite(result.metrics.loss)


def test_listwise_plackett_luce_matches_manual_probability() -> None:
    rewards = jnp.asarray(((2.0, 1.0, 0.0),))
    preferences = jnp.asarray(((3.0, 2.0, 1.0),))
    valid = jnp.ones((1, 3), dtype=jnp.bool_)
    expected = -(2.0 - jax.nn.logsumexp(rewards[0]))
    expected -= 1.0 - jax.nn.logsumexp(rewards[0, 1:])
    np.testing.assert_allclose(
        plackett_luce_loss(rewards, preferences, valid), expected, rtol=1e-6
    )


def test_listwise_and_process_rewards_are_jittable() -> None:
    reward_model = model()
    listwise = ListwiseRewardBatch(
        candidates=Inputs(
            jnp.arange(2 * 3 * 3, dtype=jnp.float32).reshape(2, 3, 3) / 10
        ),
        preferences=jnp.asarray(((3.0, 1.0, 2.0), (2.0, 1.0, 0.0))),
        valid=jnp.ones((2, 3), dtype=jnp.bool_),
    )
    list_task = build_task(ListwiseRewardConfig(), PlackettLuceConfig())
    assert np.isfinite(
        jax.jit(lambda m: list_task.loss(m, listwise).loss)(reward_model)
    )

    process_model = model(outputs=4)
    process = ProcessRewardBatch(
        inputs=Inputs(jnp.ones((2, 3))),
        labels=jnp.asarray(((1.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0))),
        valid=jnp.asarray(((True, True, True, False), (True, True, False, False))),
    )
    process_task = ProcessRewardTask()
    assert np.isfinite(
        jax.jit(lambda m: process_task.loss(m, process).loss)(process_model)
    )


def _assert_accumulation_matches(task, candidate, batch) -> None:
    optimizer = optax.sgd(1e-2)
    state = init_train_state(candidate, optimizer)
    direct = build_train_step(task, optimizer, max_grad_norm=None)(state, batch, None)
    accumulated = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        gradient_accumulation_steps=2,
    )(state, batch, None)

    np.testing.assert_allclose(accumulated.metrics.loss, direct.metrics.loss, rtol=1e-6)
    for name in direct.metrics.task:
        np.testing.assert_allclose(
            accumulated.metrics.task[name],
            direct.metrics.task[name],
            rtol=1e-6,
            atol=1e-7,
        )
    for actual, expected in zip(
        jax.tree.leaves(accumulated.state),
        jax.tree.leaves(direct.state),
        strict=True,
    ):
        if eqx.is_array(actual):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_all_reward_families_accumulate_exactly() -> None:
    rows = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    valid_rows = jnp.asarray((True, False, True, True))
    _assert_accumulation_matches(
        PairwiseRewardTask(center_rewards_coefficient=0.01),
        model(),
        PairwiseRewardBatch(
            chosen=Inputs(rows + 0.2),
            rejected=Inputs(rows - 0.1),
            margins=jnp.asarray((0.2, 0.0, 0.1, 0.3)),
            valid=valid_rows,
        ),
    )
    _assert_accumulation_matches(
        ListwiseRewardTask(),
        model(),
        ListwiseRewardBatch(
            candidates=Inputs(
                jnp.arange(4 * 3 * 3, dtype=jnp.float32).reshape(4, 3, 3) / 10
            ),
            preferences=jnp.asarray(
                (
                    (3.0, 2.0, 1.0),
                    (2.0, 1.0, 0.0),
                    (1.0, 3.0, 2.0),
                    (2.0, 1.0, 3.0),
                )
            ),
            valid=jnp.asarray(
                (
                    (True, True, True),
                    (True, True, False),
                    (True, True, True),
                    (True, False, False),
                )
            ),
        ),
    )
    _assert_accumulation_matches(
        PointwiseRewardTask(objective="mse"),
        model(),
        PointwiseRewardBatch(
            inputs=Inputs(rows),
            labels=jnp.asarray((0.1, -0.2, 0.4, 0.8)),
            valid=valid_rows,
        ),
    )
    _assert_accumulation_matches(
        ProcessRewardTask(),
        model(outputs=3),
        ProcessRewardBatch(
            inputs=Inputs(rows),
            labels=jnp.asarray(
                (
                    (1.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            ),
            valid=jnp.asarray(
                (
                    (True, True, True),
                    (True, True, False),
                    (True, False, False),
                    (True, True, True),
                )
            ),
        ),
    )


@pytest.mark.parity
def test_bradley_terry_value_and_gradient_match_torch() -> None:
    torch = pytest.importorskip("torch")
    chosen_np = np.asarray((1.2, -0.3, 0.7), dtype=np.float32)
    rejected_np = np.asarray((0.1, 0.4, -0.2), dtype=np.float32)
    margins_np = np.asarray((0.2, 0.0, 0.1), dtype=np.float32)
    torch_values = torch.tensor(
        np.concatenate((chosen_np, rejected_np)), requires_grad=True
    )
    expected = -torch.nn.functional.logsigmoid(
        torch_values[:3] - torch_values[3:] - torch.tensor(margins_np)
    ).mean()
    expected.backward()

    def native(values):
        return bradley_terry_loss(
            values[:3],
            values[3:],
            jnp.asarray(margins_np),
            jnp.ones((3,), dtype=jnp.bool_),
        )

    actual, gradient = jax.value_and_grad(native)(
        jnp.asarray(np.concatenate((chosen_np, rejected_np)))
    )
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(gradient, torch_values.grad.numpy(), rtol=1e-6)

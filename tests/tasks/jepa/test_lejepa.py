"""Native LeJEPA objective and compiled-training contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.config import (
    BatchConfig,
    ComponentConfig,
    DataConfig,
    JobConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.core import EncoderMetadata, Modality, Route
from representax.data import mix, source
from representax.tasks import build_task
from representax.tasks.jepa import (
    JEPABatch,
    JEPAConfig,
    LeJEPAConfig,
    LeJEPATask,
    invariance_loss,
    sigreg_loss,
)
from representax.train import build_train_step, init_train_state


class Inputs(eqx.Module):
    values: jax.Array


class Projection(eqx.Module):
    weight: jax.Array
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route=None, key=None):
        del route, key
        return inputs.values @ self.weight


def test_lejepa_builds_and_trains_with_the_ordinary_step() -> None:
    task = build_task(
        JEPAConfig(),
        LeJEPAConfig(knots=9, slices=16, regularization_weight=0.1),
    )
    assert isinstance(task, LeJEPATask)
    assert task.global_views == 2
    batch = JEPABatch(
        views=Inputs(
            jnp.asarray(
                (
                    ((1.0, 0.0, 0.5), (0.9, 0.1, 0.4)),
                    ((0.0, 1.0, 0.2), (0.1, 0.8, 0.3)),
                    ((0.6, 0.4, 0.0), (0.5, 0.3, 0.2)),
                ),
                dtype=jnp.float32,
            )
        ),
        valid=jnp.ones((3, 2), dtype=jnp.bool_),
    )
    model = Projection(
        jnp.eye(3, dtype=jnp.float32),
        EncoderMetadata(
            model_id="test-projection",
            revision="test",
            output_dimension=3,
            routes=frozenset({Route.GENERIC}),
            modalities=frozenset({Modality.TEXT}),
        ),
    )
    optimizer = optax.sgd(1e-2)
    result = build_train_step(task, optimizer)(
        init_train_state(model, optimizer),
        batch,
        jax.random.key(7),
    )
    assert int(result.state.step) == 1
    assert np.isfinite(result.metrics.loss)


def test_invariance_uses_only_global_views_for_each_samples_center() -> None:
    projections = jnp.asarray(
        (
            ((1.0, 2.0), (3.0, 4.0), (99.0, 99.0)),
            ((0.0, 0.0), (2.0, 2.0), (4.0, 4.0)),
        )
    )
    valid = jnp.asarray(((True, True, False), (True, True, True)))
    np.testing.assert_allclose(
        invariance_loss(projections, valid, global_views=2),
        2.6,
    )


def test_invariance_matches_the_official_global_center_formula() -> None:
    projections = jnp.arange(3 * 8 * 5, dtype=jnp.float32).reshape(3, 8, 5) / 20

    def native(values):
        return invariance_loss(
            values,
            jnp.ones((3, 8), dtype=jnp.bool_),
            global_views=2,
        )

    expected = jnp.mean(
        jnp.square(projections - jnp.mean(projections[:, :2], axis=1, keepdims=True))
    )
    value, gradient = jax.value_and_grad(native)(projections)
    expected_value, expected_gradient = jax.value_and_grad(
        lambda values: jnp.mean(
            jnp.square(values - jnp.mean(values[:, :2], axis=1, keepdims=True))
        )
    )(projections)
    np.testing.assert_allclose(value, expected)
    np.testing.assert_allclose(value, expected_value)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-7)


def test_sigreg_is_finite_and_differentiable() -> None:
    projections = jnp.arange(4 * 2 * 5, dtype=jnp.float32).reshape(4, 2, 5) / 20
    valid = jnp.ones((4, 2), dtype=jnp.bool_)
    directions = jax.random.normal(jax.random.key(3), (5, 11))
    value, gradient = jax.value_and_grad(sigreg_loss)(
        projections,
        valid,
        directions,
        knots=9,
        max_frequency=2.5,
    )
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_lejepa_rejects_inexact_gradient_accumulation() -> None:
    with pytest.raises(ValueError, match="does not decompose exactly"):
        JobConfig(
            name="lejepa-accumulation-gate",
            model=ModelConfig(target="tests.models.ToyEncoder"),
            task=JEPAConfig(),
            loss=LeJEPAConfig(),
            data=DataConfig(
                distribution=mix(
                    source("file:///tmp/unused.jsonl", map="tests.data.identity")
                )
            ),
            optimization=OptimizationConfig(
                optimizer=ComponentConfig(
                    target="optax.adamw",
                    parameters={"learning_rate": 1e-3},
                )
            ),
            training=TrainingConfig(
                global_batch_size=4,
                max_steps=1,
                seed=0,
                batch=BatchConfig(
                    micro_batch_size=2,
                    gradient_accumulation_steps=2,
                ),
            ),
        )


@pytest.mark.parity
def test_sigreg_matches_a_torch_formula_oracle() -> None:
    torch = pytest.importorskip("torch")
    projections_np = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3) / 10
    directions_np = np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, -1.0)), dtype=np.float32)
    projections = torch.tensor(projections_np, requires_grad=True)
    directions = torch.tensor(directions_np)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    knots = 7
    max_frequency = 2.0
    t = torch.linspace(0.0, max_frequency, knots)
    dt = max_frequency / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt)
    weights[0] = weights[-1] = dt
    phi = torch.exp(-(t**2) / 2.0)
    weights = weights * phi
    sliced = torch.einsum("bvd,ds->vbs", projections, directions)
    values = sliced[..., None] * t
    error = (values.cos().mean(dim=1) - phi).square() + values.sin().mean(
        dim=1
    ).square()
    expected = (error * weights).sum(dim=-1).mul(projections.shape[0]).mean()
    expected.backward()

    def native(values):
        return sigreg_loss(
            values,
            jnp.ones((4, 2), dtype=jnp.bool_),
            jnp.asarray(directions_np),
            knots=knots,
            max_frequency=max_frequency,
        )

    actual, gradient = jax.value_and_grad(native)(jnp.asarray(projections_np))
    np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(gradient, projections.grad.numpy(), rtol=1e-5, atol=1e-6)

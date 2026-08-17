from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.models import (
    DenoisingAutoEncoder,
    DenseEncoder,
    EncoderPair,
    PairClassifier,
    TokenReconstructionDecoder,
)
from representax.tasks.classification import (
    SoftmaxClassificationTask,
    pair_classification_batch,
)
from representax.tasks.contrastive_tension import (
    ContrastiveTensionInBatchTask,
    ContrastiveTensionTask,
    contrastive_tension_batch,
    contrastive_tension_examples,
)
from representax.tasks.mega_batch import MegaBatchMarginTask, mega_batch
from representax.tasks.reconstruction import DenoisingAutoEncoderTask, denoising_batch
from representax.tasks.regularization import (
    GlobalOrthogonalRegularizationTask,
    regularization_batch,
)
from representax.train import MegaBatchMining, build_train_step, init_train_state


def _step(task, model, batch, *, execution=None):
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    result = build_train_step(task, optimizer, execution=execution)(
        state,
        batch,
        jax.random.key(17),
    )
    assert bool(result.metrics.numeric_finite)
    assert int(result.state.step) == 1
    return result


def test_classification_contrastive_tension_and_gor_compile():
    encoder = DenseEncoder(7, 5, key=jax.random.key(1))
    left = jax.random.normal(jax.random.key(2), (8, 7))
    right = jax.random.normal(jax.random.key(3), (8, 7))

    classifier = PairClassifier.init(
        encoder,
        feature_dimension=20,
        class_count=3,
        key=jax.random.key(4),
    )
    classification = pair_classification_batch(
        left=left,
        right=right,
        labels=jnp.arange(8) % 3,
    )
    _step(
        SoftmaxClassificationTask(concatenate_product=True),
        classifier,
        classification,
    )

    encoder_pair = EncoderPair.from_encoder(encoder)
    tension = contrastive_tension_batch(
        first=left,
        second=right,
        labels=jnp.arange(8) % 2,
    )
    _step(ContrastiveTensionTask(), encoder_pair, tension)

    scaled_pair = EncoderPair.from_encoder(encoder, scale=20.0)
    examples = contrastive_tension_examples(left)
    _step(ContrastiveTensionInBatchTask(), scaled_pair, examples)

    regularization = regularization_batch((left, right))
    _step(GlobalOrthogonalRegularizationTask(), encoder, regularization)


def test_denoising_and_bounded_mega_batch_execution_compile():
    encoder = DenseEncoder(7, 5, key=jax.random.key(5))
    damaged = jax.random.normal(jax.random.key(6), (8, 7))
    target_ids = jnp.asarray(
        [
            [1, 2, 3, 4, 0],
            [1, 3, 4, 0, 0],
            [1, 4, 5, 2, 0],
            [1, 2, 2, 3, 0],
            [1, 5, 4, 3, 0],
            [1, 3, 2, 4, 0],
            [1, 2, 5, 3, 0],
            [1, 4, 3, 2, 0],
        ]
    )
    autoencoder = DenoisingAutoEncoder(
        encoder=encoder,
        decoder=TokenReconstructionDecoder.init(
            vocabulary_size=6,
            hidden_size=5,
            key=jax.random.key(7),
        ),
    )
    reconstruction = denoising_batch(
        damaged=damaged,
        target_input_ids=target_ids,
    )
    _step(DenoisingAutoEncoderTask(pad_token_id=0), autoencoder, reconstruction)

    positive = jax.random.normal(jax.random.key(8), (8, 7))
    mined = mega_batch(anchor=damaged, positive=positive)
    task = MegaBatchMarginTask()
    direct = _step(task, encoder, mined)
    bounded = _step(
        task,
        encoder,
        mined,
        execution=MegaBatchMining(micro_batch_size=3, loss_row_chunk_size=2),
    )
    np.testing.assert_allclose(
        bounded.metrics.loss,
        direct.metrics.loss,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        bounded.state.model.projection.weight,
        direct.state.model.projection.weight,
        rtol=3e-5,
        atol=3e-5,
    )

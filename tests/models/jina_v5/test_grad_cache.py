"""Direct/rematerialized/custom-VJP parity on complete, distinct multimodal samples."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.models.jina_v5 import batch_from_processor_output
from representax.tasks.modifiers import MatryoshkaTask
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache
from tests.models.jina_v5.test_omni import tiny_config, tiny_model


def media_batch(modality, *, reverse_visual_order=False):
    config = tiny_config()
    ids = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
        ],
        dtype=np.int32,
    )
    features = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    if modality in {"image", "video"}:
        ids[:, 1] = config.image_token_id
        ids[1, 2] = config.image_token_id
        # Middle sample has two visual tokens and a different grid/frame count.
        grids = [[1, 2, 2], [1, 2, 4], [1, 2, 2]]
        if modality == "video":
            grids[1] = [2, 2, 2]
        if reverse_visual_order:
            features["_visual_token_positions"] = np.array(
                [[2, 1], [1, 1], [1, 2], [0, 1]], dtype=np.int32
            )
        features["image_grid_thw" if modality == "image" else "video_grid_thw"] = (
            np.array(grids)
        )
        features["pixel_values" if modality == "image" else "pixel_values_videos"] = (
            np.random.default_rng(7)
            .normal(size=(16, config.vision.patch_dimension))
            .astype(np.float32)
        )
    if modality == "audio":
        ids[:, 1:4] = config.audio_token_id
        features["input_features"] = (
            np.random.default_rng(8).normal(size=(3, 4, 13)).astype(np.float32)
        )
        features["feature_attention_mask"] = np.ones((3, 13), dtype=np.int32)
    return batch_from_processor_output(
        features,
        config,
        sequence_length_buckets=(8,),
        patch_count_buckets=(4, 8, 16),
        audio_chunk_count_buckets=(2,),
        audio_token_count_buckets=(3,),
    )


def assert_close(actual, expected):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if eqx.is_array(left):
            np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-5)


def test_visual_lookup_handles_non_row_ordered_media():
    batch = media_batch("image", reverse_visual_order=True)
    np.testing.assert_array_equal(
        batch.visual_group_indices, [[3, -1], [1, 2], [0, -1]]
    )
    chunks = batch.batch_to_scan(local_chunk_size=2)
    selected = np.array([12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11])
    valid = np.asarray(chunks.patch_valid[0])
    np.testing.assert_array_equal(
        chunks.pixel_values[0][valid], batch.pixel_values[selected]
    )
    np.testing.assert_array_equal(
        chunks.position_interpolation_indices[0][:, valid],
        batch.position_interpolation_indices[:, selected],
    )
    np.testing.assert_array_equal(
        chunks.visual_token_indices[0][chunks.visual_token_valid[0]], [1, 9, 10]
    )
    # Last row is repeated for padding, but the copies have distinct vision segments.
    assert int(chunks.vision_segment_ids[1, 0]) != int(chunks.vision_segment_ids[1, 8])


@pytest.mark.parametrize("modality", ["text", "image", "audio", "video"])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_chunk_layout_preserves_embeddings_and_partial_rows(modality, chunk_size):
    batch = media_batch(modality)
    model = tiny_model(tiny_config())
    chunks = batch.batch_to_scan(local_chunk_size=chunk_size)
    direct = eqx.filter_jit(lambda x: model.encode(x, route=Route.DOCUMENT))(batch)
    chunked = eqx.filter_jit(
        lambda xs: jax.lax.map(lambda x: model.encode(x, route=Route.DOCUMENT), xs)
    )(chunks).reshape(-1, 8)
    expected = direct[jnp.minimum(jnp.arange(chunked.shape[0]), 2)]
    assert_close(chunked, expected)
    if modality in {"image", "video"}:
        # A one-sample replay carries at most the per-sample bucket, not the full batch.
        assert chunks.pixel_values.shape[1] == chunk_size * 8
        padding_rows = ((3 + chunk_size - 1) // chunk_size) * chunk_size - 3
        assert int(chunks.visual_token_valid.sum()) == 4 + padding_rows


@pytest.mark.runtime
@pytest.mark.parametrize("modality", ["text", "image", "audio", "video"])
@pytest.mark.parametrize("modified", [False, True])
def test_all_modalities_match_loss_gradients_and_optimizer_update(modality, modified):
    model = tiny_model(tiny_config())
    batch = retrieval_batch(
        query=media_batch("text"),
        document=media_batch(modality),
        positive_mask=jnp.eye(3, dtype=bool),
    )
    task = MNRTask(scale=3, symmetric=True)
    if modified:
        task = MatryoshkaTask(task, (4, 8), weights=(1.0, 2.0))
    key = jax.random.key(9)
    optimizer = optax.adamw(1e-4)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def run(execution):
        def loss(candidate):
            output = (
                task.loss(candidate, batch, key=key)
                if execution is None
                else execution.evaluate(task, candidate, batch, key=key)
            )
            return output.loss, output.metrics

        (value, metrics), gradients = eqx.filter_jit(
            eqx.filter_value_and_grad(loss, has_aux=True)
        )(model)
        updates, opt_state = optimizer.update(gradients, state, model)
        return value, metrics, gradients, eqx.apply_updates(model, updates), opt_state

    expected = run(None)
    for implementation in ("rematerialized", "custom_vjp"):
        actual = run(
            GradCache(
                query_chunk_size=2, document_chunk_size=2, implementation=implementation
            )
        )
        assert_close(actual, expected)

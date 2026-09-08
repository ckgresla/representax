"""Sample-preserving chunks for Jina's flattened visual and row-major audio inputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .omni import JinaV5OmniBatch


def batch_to_scan(batch: JinaV5OmniBatch, *, local_chunk_size: int) -> JinaV5OmniBatch:
    if local_chunk_size <= 0 or batch.batch_size <= 0:
        raise ValueError("chunk size and batch size must be positive")
    if batch.pixel_values is not None and batch.visual_group_indices is None:
        raise ValueError(
            "visual chunking requires processor-built visual_group_indices"
        )
    count = (batch.batch_size + local_chunk_size - 1) // local_chunk_size
    rows = jnp.minimum(
        jnp.arange(count * local_chunk_size), batch.batch_size - 1
    ).reshape(count, local_chunk_size)

    def select(selected):
        values = {
            "input_ids": batch.input_ids[selected],
            "attention_mask": batch.attention_mask[selected],
            "position_ids": (
                None if batch.position_ids is None else batch.position_ids[:, selected]
            ),
        }
        for name in (
            "input_features",
            "audio_feature_valid",
            "audio_after_cnn_valid",
            "audio_pool_indices",
            "audio_token_indices",
            "audio_token_valid",
        ):
            value = getattr(batch, name)
            values[name] = None if value is None else value[selected]

        if batch.pixel_values is not None:
            group_table = batch.visual_group_indices[selected]
            groups = group_table.reshape(-1)
            safe_groups = jnp.maximum(groups, 0)
            merge = batch.pixel_values.shape[0] // batch.visual_token_indices.shape[0]
            patches = (safe_groups[:, None] * merge + jnp.arange(merge)).reshape(-1)
            valid = (groups >= 0) & batch.visual_token_valid[safe_groups]
            local_rows = jnp.repeat(jnp.arange(local_chunk_size), group_table.shape[1])
            sequence = batch.input_ids.shape[1]
            values.update(
                pixel_values=batch.pixel_values[patches],
                patch_valid=batch.patch_valid[patches] & jnp.repeat(valid, merge),
                # Repeated padding rows must not attend to the original row.
                vision_segment_ids=(
                    batch.vision_segment_ids[patches]
                    + jnp.repeat(local_rows, merge) * batch.pixel_values.shape[0]
                ),
                vision_position_ids=batch.vision_position_ids[patches],
                position_interpolation_indices=batch.position_interpolation_indices[
                    :, patches
                ],
                position_interpolation_weights=batch.position_interpolation_weights[
                    :, patches
                ],
                visual_token_indices=(
                    batch.visual_token_indices[safe_groups] % sequence
                    + local_rows * sequence
                ),
                visual_token_valid=valid,
            )
        return JinaV5OmniBatch(**values)

    return jax.vmap(select)(rows)

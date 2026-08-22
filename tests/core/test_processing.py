"""Host-side processor primitive tests."""

from __future__ import annotations

import hashlib
from typing import Any

import equinox as eqx
import grain
import jax
import numpy as np
import pytest

from representax.core import Modality, Route
from representax.data import Artifact, build_data_loader
from representax.models import (
    Processor,
    make_audio_processor,
    make_image_processor,
    make_video_processor,
    select_static_shape_bucket,
)


class ImageInputs(eqx.Module):
    pixel_values: jax.Array
    pixel_mask: jax.Array


class AudioInputs(eqx.Module):
    audio_values: jax.Array
    audio_mask: jax.Array


class VideoInputs(eqx.Module):
    pixel_values: jax.Array
    frame_mask: jax.Array


def image_batch(*, pixel_values, pixel_mask):
    return ImageInputs(pixel_values=pixel_values, pixel_mask=pixel_mask)


def audio_batch(*, audio_values, audio_mask):
    return AudioInputs(audio_values=audio_values, audio_mask=audio_mask)


def video_batch(*, pixel_values, frame_mask):
    return VideoInputs(pixel_values=pixel_values, frame_mask=frame_mask)


def metadata_shape(artifact: Artifact, *, route: Route) -> tuple[int, ...]:
    del route
    return tuple(artifact.metadata["required_shape"])


def test_processor_is_a_serializable_host_boundary():
    processor = Processor(
        process=lambda values, *, route, seed: (tuple(values), route, seed),
        contract={"kind": "identity", "shape": [2]},
    )

    values, route, seed = processor(("a", "b"), seed=3)

    assert values == ("a", "b")
    assert route.value == "generic"
    assert seed == 3
    assert processor.data_contract() == {"kind": "identity", "shape": [2]}


def test_image_processor_probes_before_lazy_decode_and_pads_one_bucket(tmp_path):
    stored = {
        "small": np.ones((120, 160, 3), dtype=np.uint8),
        "large": np.ones((180, 200, 3), dtype=np.uint8),
    }
    references: dict[str, tuple[str, tuple[int, int], str]] = {}
    for name, image in stored.items():
        prefix = f"ignored-{name}:".encode()
        payload = image.tobytes()
        path = tmp_path / f"{name}.bin"
        path.write_bytes(prefix + payload + b":also-ignored")
        references[name] = (
            path.as_uri(),
            (len(prefix), len(prefix) + len(payload)),
            "sha256:" + hashlib.sha256(payload).hexdigest(),
        )
    decoded: list[str] = []

    def prepare(artifact, *, bucket, route, rng):
        del route, rng
        name = str(artifact.metadata["name"])
        decoded.append(name)
        image = np.frombuffer(artifact.read_bytes(), dtype=np.uint8).reshape(
            artifact.metadata["source_shape"]
        )
        height, width = bucket
        pixels = np.zeros((3, height, width), dtype=np.float32)
        valid_height = min(height, image.shape[0])
        valid_width = min(width, image.shape[1])
        pixels[:, :valid_height, :valid_width] = np.moveaxis(
            image[:valid_height, :valid_width] / 255.0,
            -1,
            0,
        )
        mask = np.zeros((height, width), dtype=bool)
        mask[:valid_height, :valid_width] = True
        return {"pixel_values": pixels, "pixel_mask": mask}

    processor = make_image_processor(
        admitted_shapes=((128, 192), (224, 224), (336, 336)),
        probe=metadata_shape,
        prepare=prepare,
        batch_builder=image_batch,
        configuration={"resize": "pad", "scale": 1 / 255},
    )
    artifacts = (
        Artifact.ref(
            Modality.IMAGE,
            uri=references["small"][0],
            byte_range=references["small"][1],
            checksum=references["small"][2],
            metadata={
                "name": "small",
                "required_shape": (120, 160),
                "source_shape": stored["small"].shape,
            },
        ),
        Artifact.ref(
            Modality.IMAGE,
            uri=references["large"][0],
            byte_range=references["large"][1],
            checksum=references["large"][2],
            metadata={
                "name": "large",
                "required_shape": (180, 200),
                "source_shape": stored["large"].shape,
            },
        ),
    )

    assert decoded == []
    batch = processor(artifacts, route=Route.DOCUMENT)

    assert decoded == ["small", "large"]
    assert batch.pixel_values.shape == (2, 3, 224, 224)
    assert batch.pixel_mask.shape == (2, 224, 224)
    assert processor.data_contract()["admitted_shapes"] == [
        [128, 192],
        [224, 224],
        [336, 336],
    ]


def test_media_processor_is_a_native_grain_batch_mapper_and_fingerprint():
    def prepare(artifact, *, bucket, route, rng):
        del route, rng
        height, width = bucket
        pixels = np.zeros((3, height, width), dtype=np.float32)
        source = np.asarray(artifact.data)
        pixels[:, : source.shape[0], : source.shape[1]] = np.moveaxis(
            source,
            -1,
            0,
        )
        return {
            "pixel_values": pixels,
            "pixel_mask": np.any(pixels != 0, axis=0),
        }

    def processor(buckets):
        return make_image_processor(
            admitted_shapes=buckets,
            probe=metadata_shape,
            prepare=prepare,
            batch_builder=image_batch,
            configuration={"resize": "pad"},
        )

    artifacts = [
        Artifact.inline(
            Modality.IMAGE,
            np.ones((6, 7, 3), dtype=np.float32),
            metadata={"required_shape": (6, 7)},
        ),
        Artifact.inline(
            Modality.IMAGE,
            np.ones((8, 8, 3), dtype=np.float32),
            metadata={"required_shape": (8, 8)},
        ),
    ]
    media = processor(((8, 8), (16, 16)))
    loader = build_data_loader(
        grain.MapDataset.source(artifacts),
        batch_size=2,
        batch_fn=media,
        num_threads=0,
        prefetch_buffer_size=0,
        data_contract={"name": "inline-images", "revision": "1"},
    )

    batch = next(iter(loader))

    assert batch.pixel_values.shape == (2, 3, 8, 8)
    assert loader.data_contract["source"]["batch_mapper"]["implementation"][
        "state_sha256"
    ].startswith("sha256:")
    expanded = build_data_loader(
        grain.MapDataset.source(artifacts),
        batch_size=2,
        batch_fn=processor(((16, 16),)),
        num_threads=0,
        prefetch_buffer_size=0,
        data_contract={"name": "inline-images", "revision": "1"},
    )
    assert loader.data_fingerprint != expanded.data_fingerprint


def test_audio_processor_selects_seeded_byte_window_before_decode(tmp_path):
    def prepare(artifact, *, bucket, route, rng):
        del route
        assert rng is not None
        samples = bucket[0]
        available = int(artifact.metadata["samples"])
        start = int(rng.integers(0, available - samples + 1))
        item_size = np.dtype(np.float32).itemsize
        selected = artifact.with_byte_range(
            start * item_size,
            (start + samples) * item_size,
        )
        values = np.frombuffer(selected.read_bytes(), dtype=np.float32)
        return {
            "audio_values": values,
            "audio_mask": np.ones((samples,), dtype=bool),
        }

    processor = make_audio_processor(
        admitted_shapes=((256,), (512,)),
        probe=metadata_shape,
        prepare=prepare,
        batch_builder=audio_batch,
        configuration={"window": "random", "sample_rate": 16_000},
    )
    path = tmp_path / "audio.f32"
    path.write_bytes(np.arange(2048, dtype=np.float32).tobytes())
    artifact = Artifact.ref(
        Modality.AUDIO,
        uri=path.as_uri(),
        metadata={"required_shape": (256,), "samples": 2048},
    )

    first = processor((artifact,), seed=11)
    repeated = processor((artifact,), seed=11)
    different = processor((artifact,), seed=12)

    assert first.audio_values.shape == (1, 256)
    np.testing.assert_array_equal(first.audio_values, repeated.audio_values)
    assert not np.array_equal(first.audio_values, different.audio_values)


def test_video_processor_selects_ranges_before_decoding_static_frames(tmp_path):
    def prepare(artifact, *, bucket, route, rng):
        del route
        assert rng is not None
        frames, height, width = bucket
        source_shape = tuple(artifact.metadata["source_shape"])
        indices = np.sort(rng.choice(source_shape[0], size=frames, replace=False))
        frame_values = int(np.prod(source_shape[1:]))
        frame_bytes = frame_values * np.dtype(np.float32).itemsize
        selected = np.stack(
            [
                np.frombuffer(
                    artifact.with_byte_range(
                        int(index) * frame_bytes,
                        (int(index) + 1) * frame_bytes,
                    ).read_bytes(),
                    dtype=np.float32,
                ).reshape(source_shape[1:])
                for index in indices
            ]
        )
        pixels = np.zeros((frames, 3, height, width), dtype=np.float32)
        copy_height = min(height, selected.shape[1])
        copy_width = min(width, selected.shape[2])
        pixels[:, :, :copy_height, :copy_width] = np.moveaxis(
            selected[:, :copy_height, :copy_width],
            -1,
            1,
        )
        return {
            "pixel_values": pixels,
            "frame_mask": np.ones((frames,), dtype=bool),
        }

    processor = make_video_processor(
        admitted_shapes=((8, 128, 128), (16, 224, 224)),
        probe=metadata_shape,
        prepare=prepare,
        batch_builder=video_batch,
        configuration={"frames": "random-without-replacement"},
    )
    source = np.broadcast_to(
        np.arange(24, dtype=np.float32)[:, None, None, None],
        (24, 96, 112, 3),
    ).copy()
    path = tmp_path / "video.f32"
    path.write_bytes(source.tobytes())
    artifact = Artifact.ref(
        Modality.VIDEO,
        uri=path.as_uri(),
        metadata={
            "required_shape": (8, 96, 112),
            "source_shape": source.shape,
        },
    )

    batch = processor((artifact,), seed=7)
    repeated = processor((artifact,), seed=7)
    different = processor((artifact,), seed=8)

    assert batch.pixel_values.shape == (1, 8, 3, 128, 128)
    assert batch.frame_mask.shape == (1, 8)
    np.testing.assert_array_equal(batch.pixel_values, repeated.pixel_values)
    assert not np.array_equal(batch.pixel_values, different.pixel_values)


def test_media_processors_reject_wrong_modalities_and_unstable_outputs():
    def unstable(artifact: Artifact, *, bucket, route, rng) -> dict[str, Any]:
        del bucket, route, rng
        size = int(artifact.metadata["output"])
        return {"values": np.ones((size,), dtype=np.float32)}

    processor = make_image_processor(
        admitted_shapes=((8, 8),),
        probe=metadata_shape,
        prepare=unstable,
        batch_builder=lambda **arrays: arrays,
        configuration={"test": "unstable"},
    )
    with pytest.raises(TypeError, match="cannot consume audio"):
        processor(
            (
                Artifact.inline(
                    Modality.AUDIO,
                    np.zeros((8,)),
                    metadata={"required_shape": (8, 8)},
                ),
            )
        )
    with pytest.raises(ValueError, match="unstable shapes"):
        processor(
            (
                Artifact.inline(
                    Modality.IMAGE,
                    np.zeros((8, 8, 3)),
                    metadata={"required_shape": (8, 8), "output": 3},
                ),
                Artifact.inline(
                    Modality.IMAGE,
                    np.zeros((8, 8, 3)),
                    metadata={"required_shape": (8, 8), "output": 4},
                ),
            )
        )


def test_static_shape_bucket_selects_the_smallest_containing_shape():
    assert select_static_shape_bucket(
        (7, 200, 200),
        ((8, 224, 224), (16, 224, 224), (8, 336, 336)),
    ) == (8, 224, 224)


def test_static_shape_bucket_selection_is_order_independent():
    shapes = ((16, 256), (32, 512))

    assert select_static_shape_bucket((7, 240), shapes) == (16, 256)
    assert select_static_shape_bucket((7, 240), tuple(reversed(shapes))) == (
        16,
        256,
    )


def test_static_shape_bucket_rejects_incomparable_media_tradeoffs():
    with pytest.raises(ValueError, match="model processor must choose"):
        select_static_shape_bucket((7, 240), ((16, 256), (8, 512)))


@pytest.mark.parametrize(
    ("required", "buckets", "message"),
    [
        ((), ((8,),), "required_shape"),
        ((4,), (), "at least one"),
        ((4, 4), ((8,),), "match required_shape rank"),
        ((9,), ((4,), (8,)), "exceeds admitted buckets"),
        ((4.0,), ((8,),), "positive integer dimensions"),
    ],
)
def test_static_shape_bucket_rejects_invalid_or_unadmitted_shapes(
    required,
    buckets,
    message,
):
    with pytest.raises(ValueError, match=message):
        select_static_shape_bucket(required, buckets)

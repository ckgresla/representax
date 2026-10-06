"""CPU-only media fixtures and lazy decoding contracts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from representax.data import media
from representax.data.media import decode_audio, decode_image, decode_video


def _image(path: Path) -> np.ndarray:
    image = pytest.importorskip("PIL.Image")
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    image.fromarray(pixels).save(path, format="PNG")
    return pixels


def _wav(path: Path, values: np.ndarray, rate: int, *, subtype="FLOAT") -> bytes:
    sf = pytest.importorskip("soundfile")
    sf.write(path, values, rate, format="WAV", subtype=subtype)
    return path.read_bytes()


def _video(path: Path, *, count: int = 7) -> np.ndarray:
    av = pytest.importorskip("av")
    values = np.stack(
        [np.full((12, 16, 3), index * 30, dtype=np.uint8) for index in range(count)]
    )
    with av.open(str(path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height = 16, 12
        stream.pix_fmt = "bgr0"
        for array in values:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return values


def test_codec_imports_are_lazy():
    script = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('media', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not {'PIL', 'soundfile', 'scipy', 'av'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", script, media.__file__], check=True)


def test_image_sources_and_payload_preservation(tmp_path):
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "payload.cache"
    expected = _image(path)
    payload = path.read_bytes()
    with image.open(path) as pil:
        for value in (path, str(path), payload, pil, expected):
            actual = decode_image(value)
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous
        assert pil.getpixel((0, 0)) == tuple(expected[0, 0])
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("channels", [None, 1, 4])
def test_image_grayscale_alpha_and_rounding(channels):
    pytest.importorskip("PIL")
    shape = (3, 5) if channels is None else (3, 5, channels)
    values = np.full(shape, 10.6, dtype=np.float32)
    original = values.copy()
    result = decode_image(values)
    np.testing.assert_array_equal(result, np.full((3, 5, 3), 11, dtype=np.uint8))
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    "value",
    [
        np.empty((0, 2, 3)),
        np.ones((2, 2, 2)),
        np.ones(4),
        np.full((2, 2), np.nan),
        np.full((2, 2), np.inf),
        np.full((2, 2), -1),
        np.full((2, 2), 256),
        np.array([["x"]]),
    ],
)
def test_invalid_image_arrays(value):
    pytest.importorskip("PIL")
    with pytest.raises(ValueError):
        decode_image(value)


def test_nonfinite_pil_and_encoded_images(tmp_path):
    image = pytest.importorskip("PIL.Image")
    pil = image.fromarray(np.full((2, 3), np.nan, dtype=np.float32))
    path = tmp_path / "nonfinite.tiff"
    pil.save(path)
    for value in (pil, path, path.read_bytes()):
        with pytest.raises(ValueError, match="finite"):
            decode_image(value)


def test_audio_sources_and_payload_preservation(tmp_path):
    path = tmp_path / "payload.cache"
    samples = np.tile(np.array([[0.2, 0.6]], dtype=np.float32), (80, 1))
    payload = _wav(path, samples, 16_000)
    sources = [
        path,
        str(path),
        payload,
        {"bytes": payload, "path": "missing"},
        {"bytes": None, "path": path},
        {"array": samples, "sampling_rate": 16_000},
    ]
    for value in sources:
        actual = decode_audio(value)
        np.testing.assert_allclose(actual, np.full(80, 0.4), atol=1e-7)
        assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("dtype,subtype", [(np.int16, "PCM_16"), (np.int32, "PCM_32")])
def test_pcm_scaling_precedes_channel_average(tmp_path, dtype, subtype):
    scale = 2 ** (np.dtype(dtype).itemsize * 8 - 1)
    samples = np.array([[scale // 2, 0], [-scale, 0]], dtype=dtype)
    payload = _wav(tmp_path / "pcm.wav", samples, 16_000, subtype=subtype)
    expected = np.array([0.25, -0.5], dtype=np.float32)
    np.testing.assert_array_equal(decode_audio(payload), expected)
    np.testing.assert_array_equal(
        decode_audio({"array": samples, "sampling_rate": 16_000}), expected
    )


def test_unsigned_pcm_and_float_amplitude():
    result = decode_audio(
        {"array": np.array([0, 128, 255], dtype=np.uint8), "sampling_rate": 16_000}
    )
    np.testing.assert_array_equal(result, [-1, 0, 127 / 128])
    samples = np.array([0.0, 0.01, -0.25, 2.0], dtype=np.float32)
    original = samples.copy()
    result = decode_audio({"array": samples, "sampling_rate": 16_000})
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize(
    "source_rate,target_rate", [(8000, 16_000), (44100, 16_000), (16000, 8000)]
)
def test_audio_resampling_and_prefix_policy(tmp_path, source_rate, target_rate):
    from math import ceil, floor, gcd

    signal = pytest.importorskip("scipy.signal")
    samples = (0.2 * np.sin(np.arange(source_rate // 20) * 0.1)).astype(np.float32)
    payload = _wav(tmp_path / "audio.wav", samples, source_rate)
    seconds = 0.01317
    divisor = gcd(source_rate, target_rate)
    expected = signal.resample_poly(
        samples[: ceil(seconds * source_rate)].astype(np.float64),
        target_rate // divisor,
        source_rate // divisor,
    )[: floor(seconds * target_rate)].astype(np.float32)
    for value in (payload, {"array": samples, "sampling_rate": source_rate}):
        actual = decode_audio(value, sampling_rate=target_rate, max_seconds=seconds)
        np.testing.assert_array_equal(actual, expected)
        full = decode_audio(value, sampling_rate=target_rate)
        assert full.shape == (ceil(len(samples) * target_rate / source_rate),)


def test_audio_clipping_is_bounded_and_does_not_pad(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    payload = _wav(tmp_path / "audio.wav", np.full(1000, 0.25), 16_000)
    reads = []
    original = sf.SoundFile.read

    def read(self, frames=-1, **kwargs):
        reads.append(frames)
        return original(self, frames=frames, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", read)
    actual = decode_audio(payload, max_seconds=0.01)
    assert reads == [160]
    np.testing.assert_array_equal(actual, np.full(160, 0.25))
    assert decode_audio(payload, max_seconds=1).shape == (1000,)


@pytest.mark.parametrize(
    "value", [[], [np.nan], [np.inf], [-np.inf], [[[]]], [1e100], [1j], ["bad"]]
)
def test_invalid_audio_arrays(value):
    with pytest.raises(ValueError):
        decode_audio({"array": np.asarray(value), "sampling_rate": 16_000})


@pytest.mark.parametrize("rate", [0, -1, 1.5, True, None, np.nan])
def test_invalid_audio_rates(rate):
    value = {"array": np.ones(10), "sampling_rate": 16_000}
    with pytest.raises(ValueError, match="positive integer"):
        decode_audio(value, sampling_rate=rate)
    with pytest.raises(ValueError, match="positive integer"):
        decode_audio({**value, "sampling_rate": rate})


@pytest.mark.parametrize("seconds", [0, -1, np.nan, np.inf, True, "1", 1e-9])
def test_invalid_audio_duration(seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        decode_audio(
            {"array": np.ones(10), "sampling_rate": 16_000}, max_seconds=seconds
        )


def test_invalid_encoded_audio_and_hf_mapping(tmp_path):
    pytest.importorskip("soundfile")
    path = tmp_path / "empty.wav"
    payload = _wav(path, np.empty(0, dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="empty"):
        decode_audio(payload)
    with pytest.raises(ValueError, match="empty"):
        decode_audio({"bytes": b"", "path": path})
    with pytest.raises(TypeError):
        decode_audio({"bytes": None, "path": None})
    with pytest.raises(ValueError, match="finite"):
        decode_audio(_wav(path, np.array([np.nan]), 16_000))


@pytest.mark.parametrize("frames", [1, 3, 5, 10])
def test_video_uniform_sampling_and_payload_preservation(tmp_path, frames):
    path = tmp_path / "payload.cache"
    expected = _video(path)
    payload = path.read_bytes()
    indices = np.rint(np.linspace(0, len(expected) - 1, frames)).astype(int)
    for value in (path, str(path), payload):
        actual = decode_video(value, frames=frames)
        np.testing.assert_array_equal(actual, expected[indices])
        assert actual.dtype == np.uint8 and actual.flags.c_contiguous
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]


def test_single_frame_video(tmp_path):
    path = tmp_path / "single.mkv"
    expected = _video(path, count=1)
    np.testing.assert_array_equal(
        decode_video(path, frames=4), np.repeat(expected, 4, axis=0)
    )


def _fake_video(monkeypatch, passes):
    av = pytest.importorskip("av")
    converted, closed = [], []

    class Frame:
        def __init__(self, index, *, corrupt=False, shape=(2, 3, 3)):
            self.index, self.is_corrupt, self.shape = index, corrupt, shape

        def to_ndarray(self, *, format):
            assert format == "rgb24"
            converted.append(self.index)
            return np.full(self.shape, self.index, dtype=np.uint8)

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(frames=0)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def decode(self, *, video):
            assert video == 0
            yield from next(passes)

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    return Frame, converted, closed


def test_video_only_converts_selected_unique_frames(monkeypatch):
    passes = []
    frame, converted, closed = _fake_video(monkeypatch, iter(passes))
    passes.extend([[frame(i) for i in range(101)] for _ in range(2)])
    result = decode_video(b"payload", frames=3)
    assert converted == [0, 50, 100]
    assert result.shape == (3, 2, 3, 3)
    assert len(closed) == 2
    passes.extend([[frame(0), frame(1)] for _ in range(2)])
    decode_video(b"payload", frames=5)
    assert converted[3:] == [0, 1]


@pytest.mark.parametrize("failure", ["empty", "corrupt", "short", "shape"])
def test_video_stream_failures_close_containers(monkeypatch, failure):
    passes = []
    frame, _, closed = _fake_video(monkeypatch, iter(passes))
    if failure == "empty":
        passes.append([])
    elif failure == "corrupt":
        passes.append([frame(0), frame(1, corrupt=True)])
    else:
        passes.append([frame(0), frame(1)])
        passes.append(
            [frame(0)] if failure == "short" else [frame(0), frame(1, shape=(3, 3, 3))]
        )
    with pytest.raises(ValueError):
        decode_video(b"payload", frames=2)
    assert len(closed) == (1 if failure in ("empty", "corrupt") else 2)


@pytest.mark.parametrize("frames", [0, -1, 2.5, True, None])
def test_invalid_video_frame_count(frames):
    with pytest.raises(ValueError, match="positive integer"):
        decode_video(b"payload", frames=frames)


def test_video_rejects_audio_only_file(tmp_path):
    pytest.importorskip("av")
    payload = _wav(tmp_path / "audio.wav", np.ones(16), 16_000)
    with pytest.raises(ValueError, match="no video stream"):
        decode_video(payload, frames=2)


def test_jina_processor_input_contracts(tmp_path):
    from representax.models.jina_v5.processing import _audio_array
    from representax.models.qwen2_5_omni.processing import _video_frames

    audio = decode_audio(_wav(tmp_path / "audio.wav", np.full(80, 0.25), 16_000))
    waveform, rate = _audio_array(audio)
    np.testing.assert_array_equal(waveform, audio)
    assert rate == 16_000
    path = tmp_path / "video.mkv"
    _video(path)
    video = decode_video(path, frames=4)
    accepted, _ = _video_frames(video)
    np.testing.assert_array_equal(accepted, video)


@pytest.mark.parametrize("kind", ["image", "audio", "video"])
def test_bad_payload_and_missing_path(kind, tmp_path):
    pytest.importorskip({"image": "PIL", "audio": "soundfile", "video": "av"}[kind])
    decode = {"image": decode_image, "audio": decode_audio, "video": decode_video}[kind]
    kwargs = {"frames": 2} if kind == "video" else {}
    for value in (b"", b"invalid codec data", tmp_path / "missing", ""):
        with pytest.raises((ValueError, OSError, RuntimeError)):
            decode(value, **kwargs)
    with pytest.raises(TypeError):
        decode(object(), **kwargs)  # ty: ignore[invalid-argument-type]

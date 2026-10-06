"""On-demand host media decoding; codec dependencies are imported only on use.

These helpers never write to source paths or create decoded payload caches.
Pillow, soundfile, scipy (for resampling), and av are optional runtime dependencies.
"""

from __future__ import annotations

import io
import math
import os
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np


def _source(value: Any) -> io.BytesIO | str:
    if isinstance(value, bytes):
        if not value:
            raise ValueError("media payload must not be empty")
        return io.BytesIO(value)
    if isinstance(value, (str, os.PathLike)):
        path = os.fsdecode(value)
        if not path:
            raise ValueError("media path must not be empty")
        return path
    raise TypeError("encoded media must be bytes or a filesystem path")


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not array.size:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return array


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def decode_image(value: Any) -> np.ndarray:
    """Return contiguous RGB uint8 HWC pixels from bytes, a path, PIL, or an array.

    No resizing or EXIF rotation is performed. Pillow converts the first encoded
    image (or current PIL frame) to RGB; alpha is discarded, not composited.
    Arrays are HW grayscale or HWC
    with 1, 3, or 4 channels; values must be in [0, 255] and are rounded to uint8.
    Floating arrays are pixel intensities, not implicitly scaled from [0, 1].
    Caller-owned PIL images and arrays are not modified or closed.
    """
    from PIL import Image

    if isinstance(value, np.ndarray):
        array = _finite_array(value, name="image")
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] in (3, 4)):
            raise ValueError("image must be HW or HWC with 1, 3, or 4 channels")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("image pixel intensities must be in [0, 255]")
        value = Image.fromarray(np.rint(array).astype(np.uint8))
    if isinstance(value, Image.Image):
        if value.mode == "F":
            _finite_array(np.asarray(value), name="image")
        result = np.array(value.convert("RGB"), dtype=np.uint8)
    else:
        with Image.open(_source(value)) as image:
            return decode_image(image)
    _finite_array(result, name="image")
    return np.ascontiguousarray(result)


def decode_audio(
    value: Any,
    *,
    sampling_rate: int = 16_000,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Return contiguous mono float32 audio at the requested sampling rate.

    Accept bytes/paths, HF {'bytes', 'path'} (non-None bytes take precedence),
    or {'array', 'sampling_rate'}. Decoded arrays are samples or samples/channels,
    never channel-first. Integer PCM is scaled by its dtype's full scale, with
    unsigned PCM centered first, then channels are averaged. Floating amplitude
    is preserved: there is no peak normalization, amplitude clipping, or padding.

    max_seconds keeps a prefix: read at most ceil(seconds * source_rate) samples,
    resample that prefix with scipy.signal.resample_poly's default zero boundary
    and FIR filter using the reduced integer rate ratio, then truncate to
    floor(seconds * sampling_rate) samples. A cap shorter than one output sample
    is invalid. Only the read prefix of encoded audio is validated; decoded array
    inputs are validated in full. Codec/read errors and nonfinite values propagate
    as errors, never silence. Non-default output rates need an explicit rate
    mapping when passed to processors that otherwise assume 16 kHz.
    """
    sampling_rate = _positive_integer(sampling_rate, name="sampling_rate")
    limit = None
    if max_seconds is not None:
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, Real)
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
            or not math.isfinite(max_seconds * sampling_rate)
        ):
            raise ValueError("max_seconds must be finite and positive")
        limit = math.floor(max_seconds * sampling_rate)
        if limit < 1:
            raise ValueError("max_seconds must allow at least one output sample")

    if isinstance(value, Mapping) and "array" in value:
        source_rate = _positive_integer(
            value.get("sampling_rate"), name="source sampling_rate"
        )
        audio = _finite_array(value["array"], name="audio")
    else:
        import soundfile as sf

        if isinstance(value, Mapping):
            value = (
                value["bytes"] if value.get("bytes") is not None else value.get("path")
            )
        with sf.SoundFile(_source(value), mode="r") as source:
            source_rate = source.samplerate
            count = -1 if max_seconds is None else math.ceil(max_seconds * source_rate)
            audio = source.read(frames=count, dtype="float32", always_2d=False)
        audio = _finite_array(audio, name="audio")
    if audio.ndim not in (1, 2):
        raise ValueError("audio must be samples or samples/channels")
    if max_seconds is not None:
        audio = audio[: math.ceil(max_seconds * source_rate)]
    dtype = audio.dtype
    audio = audio.astype(np.float64)
    if dtype.kind in "ui":
        scale = float(2 ** (dtype.itemsize * 8 - 1))
        if dtype.kind == "u":
            audio -= scale
        audio /= scale
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate != sampling_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(source_rate, sampling_rate)
        audio = resample_poly(audio, sampling_rate // divisor, source_rate // divisor)
    with np.errstate(over="ignore", invalid="ignore"):
        audio = np.ascontiguousarray(audio[:limit], dtype=np.float32)
    _finite_array(audio, name="decoded audio")
    return audio


def decode_video(value: bytes | str | os.PathLike[str], *, frames: int) -> np.ndarray:
    """Return exactly ``frames`` contiguous RGB uint8 frames, shaped THWC.

    Sample the first video stream by decoded frame index, not timestamp, using
    rint(linspace(0, N - 1, frames)) (ties to even). Endpoints are included when
    frames > 1; frames=1 selects the first frame. Short clips repeat indices.
    No resizing, rotation, or audio extraction is performed.

    A streaming count pass avoids unreliable/missing container frame counts.
    A second sequential pass converts only selected frames to arrays, retaining
    O(frames) images, never an entire decoded clip. Inter-frame codecs still
    decode intervening frames. Sources are reopened read-only, not rewritten.
    Empty streams, corrupt frames, and inconsistent selected dimensions fail.
    """
    frames = _positive_integer(frames, name="frames")
    import av

    with av.open(_source(value), mode="r") as container:
        if not container.streams.video:
            raise ValueError("media contains no video stream")
        count = 0
        for frame in container.decode(video=0):
            if frame.is_corrupt:
                raise ValueError("video contains a corrupt frame")
            count += 1
    if count == 0:
        raise ValueError("video contains no frames")
    indices = np.rint(np.linspace(0, count - 1, frames)).astype(np.int64)
    selected = []
    cursor = 0
    with av.open(_source(value), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.is_corrupt:
                raise ValueError("video contains a corrupt frame")
            if index != indices[cursor]:
                continue
            array = frame.to_ndarray(format="rgb24")
            _finite_array(array, name="video frame")
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("video frame must have RGB HWC dimensions")
            if selected and array.shape != selected[0].shape:
                raise ValueError("selected video frames have inconsistent dimensions")
            while cursor < frames and indices[cursor] == index:
                selected.append(array)
                cursor += 1
            if cursor == frames:
                break
    if cursor != frames:
        raise ValueError("video ended before all selected frames were decoded")
    return np.ascontiguousarray(np.stack(selected), dtype=np.uint8)

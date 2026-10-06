#!/usr/bin/env python3
"""Verify local ILSVRC2012 validation data against the official devkit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

OFFICIAL_DEVKIT_URL = (
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
)
OFFICIAL_DEVKIT_MD5 = "fa75699e90414af021442c21a62c3abf"
OFFICIAL_DEVKIT_SHA256 = (
    "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
)
OFFICIAL_VAL_ARCHIVE_MD5 = "29b22e2961454d5413ddabcf34fc5622"
OFFICIAL_VAL_ARCHIVE_SHA256 = (
    "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"
)
OFFICIAL_CLASS_ID_TO_SYNSET_SHA256 = (
    "33cac7974cc0bb3935c6fca46b93729c29a6481fa635e92f5f0852859aecd361"
)
OFFICIAL_FILENAME_TO_SYNSET_TSV_SHA256 = (
    "b5b25a74f93140f3e3febc504cd1e77411d38604e61da4801e2cde971771ba54"
)

DEVKIT_ROOT = "ILSVRC2012_devkit_t12"
GROUND_TRUTH = f"{DEVKIT_ROOT}/data/ILSVRC2012_validation_ground_truth.txt"
META = f"{DEVKIT_ROOT}/data/meta.mat"
README = f"{DEVKIT_ROOT}/readme.txt"
COPYING = f"{DEVKIT_ROOT}/COPYING"
DEVKIT_MEMBER_SHA256 = {
    GROUND_TRUTH: ("88d44095bd81cb785618db8e72eeb025fa1f646adf359d9b0efbf625244e0df3"),
    META: "e159ee2fe4bcc4d03e7429e7ff35e6ec553b2a901018b559f400a2b0ae117a30",
    README: "5172745470d3c087804f037ef18a21973e8dbaa96535334f23e2a42256ff2f72",
    COPYING: ("5f4d5e6ba018a1c22c12bcb5d7d9c42c5ffe9672ab0165491203a4111a0b4463"),
}

TORCHVISION_COMMIT = "8fb87713a24951e639c494b0f2a8a81b5f8e33a6"
TORCHVISION_SOURCE = (
    "https://github.com/pytorch/vision/blob/"
    f"{TORCHVISION_COMMIT}/torchvision/datasets/imagenet.py"
)
TORCHVISION_SOURCE_SHA256 = (
    "afa033ca33efe53930c5c9d7a26419caee1c134a0888761d5415e4157f04ff2d"
)

EXPECTED_IMAGES = 50_000
EXPECTED_CLASSES = 1_000
EXPECTED_META_SYNSETS = 1_860
EXPECTED_PER_CLASS = 50


class VerificationError(RuntimeError):
    """Raised when local data does not match the pinned official inputs."""


@dataclass(frozen=True, slots=True)
class MappingRecord:
    filename: str
    class_id: int
    synset: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hashes(path: Path) -> dict[str, str | int]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _require_hashes(
    path: Path,
    *,
    md5: str,
    sha256: str,
    label: str,
) -> dict[str, str | int]:
    actual = file_hashes(path)
    if actual["md5"] != md5 or actual["sha256"] != sha256:
        raise VerificationError(f"{label} checksum mismatch: {actual}")
    return actual


def devkit_members(path: Path) -> dict[str, bytes]:
    _require_hashes(
        path,
        md5=OFFICIAL_DEVKIT_MD5,
        sha256=OFFICIAL_DEVKIT_SHA256,
        label="devkit",
    )
    with tarfile.open(path, "r:gz") as archive:
        values: dict[str, bytes] = {}
        for name, expected in DEVKIT_MEMBER_SHA256.items():
            stream = archive.extractfile(name)
            if stream is None:
                raise VerificationError(f"devkit is missing {name}")
            value = stream.read()
            if sha256_bytes(value) != expected:
                raise VerificationError(f"devkit member checksum mismatch: {name}")
            values[name] = value
    return values


def class_synsets_from_meta(
    meta: bytes,
    *,
    expected_synsets: int = EXPECTED_META_SYNSETS,
    expected_classes: int = EXPECTED_CLASSES,
) -> tuple[str, ...]:
    """Read ordered WNIDs from the pinned MATLAB v5 metadata file."""

    if not meta.startswith(b"MATLAB 5.0 MAT-file") or meta[126:128] != b"IM":
        raise VerificationError("meta.mat is not the expected MATLAB v5 file")

    payloads: list[bytes] = []
    offset = 128
    while offset + 8 <= len(meta):
        data_type, size = struct.unpack_from("<II", meta, offset)
        start, end = offset + 8, offset + 8 + size
        if end > len(meta):
            raise VerificationError("meta.mat contains a truncated element")
        value = meta[start:end]
        if data_type == 15:  # MATLAB miCOMPRESSED
            try:
                value = zlib.decompress(value)
            except zlib.error as error:
                raise VerificationError(
                    "meta.mat contains invalid zlib data"
                ) from error
        payloads.append(value)
        offset = (end + 7) & ~7

    wnids = tuple(
        match.decode("ascii") for match in re.findall(rb"n[0-9]{8}", b"".join(payloads))
    )
    if len(wnids) != expected_synsets or len(set(wnids)) != expected_synsets:
        raise VerificationError(
            f"expected {expected_synsets} unique ordered WNIDs, found {len(wnids)}"
        )
    return wnids[:expected_classes]


def canonical_mapping(
    ground_truth: bytes,
    class_synsets: tuple[str, ...],
    *,
    expected_images: int = EXPECTED_IMAGES,
    expected_per_class: int = EXPECTED_PER_CLASS,
) -> tuple[MappingRecord, ...]:
    try:
        class_ids = tuple(int(line) for line in ground_truth.splitlines())
    except ValueError as error:
        raise VerificationError("ground truth contains a non-integer ID") from error
    counts = Counter(class_ids)
    if (
        len(class_ids) != expected_images
        or set(counts) != set(range(1, len(class_synsets) + 1))
        or set(counts.values()) != {expected_per_class}
    ):
        raise VerificationError("ground truth has the wrong IDs or class counts")
    return tuple(
        MappingRecord(
            f"ILSVRC2012_val_{index:08d}.JPEG",
            class_id,
            class_synsets[class_id - 1],
        )
        for index, class_id in enumerate(class_ids, start=1)
    )


def mapping_tsv(records: tuple[MappingRecord, ...]) -> bytes:
    rows = ["filename\tILSVRC2012_ID\tsynset"]
    rows.extend(
        f"{record.filename}\t{record.class_id}\t{record.synset}" for record in records
    )
    return ("\n".join(rows) + "\n").encode()


def _class_mapping_hash(class_synsets: tuple[str, ...]) -> str:
    value = "".join(
        f"{index}\t{synset}\n" for index, synset in enumerate(class_synsets, start=1)
    ).encode()
    return sha256_bytes(value)


def verify_validation_tree(
    root: Path,
    records: tuple[MappingRecord, ...],
) -> dict[str, Any]:
    expected = {record.filename: record.synset for record in records}
    local: dict[str, str] = {}
    malformed = 0
    duplicates = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if len(path.relative_to(root).parts) != 2:
            malformed += 1
        elif path.name in local:
            duplicates += 1
        else:
            local[path.name] = path.parent.name

    missing = set(expected) - set(local)
    unexpected = set(local) - set(expected)
    misassigned = {
        name for name in expected.keys() & local.keys() if expected[name] != local[name]
    }
    counts = Counter(local.values())
    expected_counts = Counter(expected.values())
    directories = {path.name for path in root.iterdir() if path.is_dir()}
    if (
        malformed
        or duplicates
        or missing
        or unexpected
        or misassigned
        or counts != expected_counts
        or directories != set(expected_counts)
    ):
        raise VerificationError(
            "validation tree mismatch: "
            f"malformed={malformed}, duplicates={duplicates}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"misassigned={len(misassigned)}"
        )
    return {
        "image_count": len(local),
        "class_count": len(counts),
        "images_per_class_min": min(counts.values()),
        "images_per_class_max": max(counts.values()),
        "exact_filename_set": True,
        "exact_parent_synset_assignments": True,
    }


def _compare_streams(archive: IO[bytes], local: IO[bytes]) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        left, right = archive.read(1024 * 1024), local.read(1024 * 1024)
        if left != right:
            raise VerificationError("local image differs from its archive member")
        if not left:
            return size, digest.hexdigest()
        digest.update(left)
        size += len(left)


def verify_images_against_archive(
    archive_path: Path,
    val_root: Path,
    records: tuple[MappingRecord, ...],
    *,
    expected_md5: str = OFFICIAL_VAL_ARCHIVE_MD5,
    expected_sha256: str = OFFICIAL_VAL_ARCHIVE_SHA256,
) -> dict[str, Any]:
    hashes = _require_hashes(
        archive_path,
        md5=expected_md5,
        sha256=expected_sha256,
        label="validation archive",
    )
    expected_names = {record.filename for record in records}
    content_manifest = hashlib.sha256()
    total_bytes = 0
    with tarfile.open(archive_path, "r:") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive
            if member.isfile()
        }
        if set(members) != expected_names:
            raise VerificationError("archive filenames differ from the devkit")
        for record in records:
            stream = archive.extractfile(members[record.filename])
            if stream is None:
                raise VerificationError(f"could not read {record.filename}")
            local_path = val_root / record.synset / record.filename
            with stream, local_path.open("rb") as local:
                size, digest = _compare_streams(stream, local)
            total_bytes += size
            content_manifest.update(f"{record.filename}\t{size}\t{digest}\n".encode())
    return {
        "archive": hashes,
        "archive_member_count": len(members),
        "image_bytes_total": total_bytes,
        "image_content_manifest_sha256": content_manifest.hexdigest(),
        "local_files_byte_identical_to_archive": True,
    }


def _download_devkit(path: Path) -> None:
    request = urllib.request.Request(
        OFFICIAL_DEVKIT_URL,
        headers={"User-Agent": "representax-imagenet-provenance/1"},
    )
    with (  # noqa: S310
        urllib.request.urlopen(request, timeout=60) as response,
        path.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="imagenet-devkit-") as temporary:
        if arguments.devkit is None:
            devkit_path = Path(temporary) / "ILSVRC2012_devkit_t12.tar.gz"
            _download_devkit(devkit_path)
            access = "public HTTPS download without authentication"
        else:
            devkit_path = arguments.devkit.resolve()
            access = f"existing local file: {devkit_path}"

        members = devkit_members(devkit_path)
        synsets = class_synsets_from_meta(members[META])
        records = canonical_mapping(members[GROUND_TRUTH], synsets)
        mapping = mapping_tsv(records)
        mapping_hash = sha256_bytes(mapping)
        class_hash = _class_mapping_hash(synsets)
        if class_hash != OFFICIAL_CLASS_ID_TO_SYNSET_SHA256:
            raise VerificationError("class-ID-to-synset hash mismatch")
        if mapping_hash != OFFICIAL_FILENAME_TO_SYNSET_TSV_SHA256:
            raise VerificationError("filename-to-synset hash mismatch")

        tree = verify_validation_tree(arguments.val_root.resolve(), records)
        images = verify_images_against_archive(
            arguments.val_archive.resolve(),
            arguments.val_root.resolve(),
            records,
        )
        devkit_hashes = file_hashes(devkit_path)

    result = {
        "schema_version": "representax-imagenet-validation-provenance-v1",
        "status": "pass",
        "official_quality_eval_admissible": True,
        "canonical_mapping": {
            "rows": len(records),
            "classes": len(synsets),
            "images_per_class": EXPECTED_PER_CLASS,
            "class_id_to_synset_sha256": class_hash,
            "filename_to_synset_tsv_sha256": mapping_hash,
        },
        "local_dataset": {
            "validation_root": str(arguments.val_root.resolve()),
            "validation_archive": str(arguments.val_archive.resolve()),
            **tree,
            **images,
        },
        "sources": {
            "official_devkit": {
                "publisher": "ImageNet Project",
                "release": "ILSVRC2012_devkit_t12",
                "url": OFFICIAL_DEVKIT_URL,
                "access": access,
                "archive": devkit_hashes,
                "members_sha256": DEVKIT_MEMBER_SHA256,
                "license_file": COPYING,
                "license_sha256": DEVKIT_MEMBER_SHA256[COPYING],
            },
            "independent_checksum_reference": {
                "project": "torchvision",
                "tag": "v0.28.0",
                "commit": TORCHVISION_COMMIT,
                "source_url": TORCHVISION_SOURCE,
                "source_sha256": TORCHVISION_SOURCE_SHA256,
                "devkit_md5": OFFICIAL_DEVKIT_MD5,
                "validation_archive_md5": OFFICIAL_VAL_ARCHIVE_MD5,
            },
            "credentials_accessed": False,
            "redistributed_images_or_devkit": False,
        },
    }

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "filename-to-synset.tsv").write_bytes(mapping)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--val-archive", type=Path, required=True)
    parser.add_argument("--devkit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    result = verify(_parser().parse_args())
    mapping = result["canonical_mapping"]["filename_to_synset_tsv_sha256"]
    print(f"PASS: official ImageNet-1K validation mapping {mapping}")


if __name__ == "__main__":
    main()

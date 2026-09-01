"""Contracts for the dependency-free ILSVRC2012 validation verifier."""

from __future__ import annotations

import io
import struct
import tarfile
import zlib
from pathlib import Path

import pytest
from scripts.verify_imagenet_validation import (
    OFFICIAL_CLASS_ID_TO_SYNSET_SHA256,
    OFFICIAL_DEVKIT_MD5,
    OFFICIAL_DEVKIT_SHA256,
    OFFICIAL_FILENAME_TO_SYNSET_TSV_SHA256,
    OFFICIAL_VAL_ARCHIVE_MD5,
    OFFICIAL_VAL_ARCHIVE_SHA256,
    TORCHVISION_COMMIT,
    MappingRecord,
    VerificationError,
    canonical_mapping,
    class_synsets_from_meta,
    file_hashes,
    verify_images_against_archive,
    verify_validation_tree,
)


def _mat_v5_with_wnids(wnids: tuple[str, ...]) -> bytes:
    header = bytearray(128)
    header[:19] = b"MATLAB 5.0 MAT-file"
    header[124:126] = b"\x00\x01"
    header[126:128] = b"IM"
    compressed = zlib.compress(" ".join(wnids).encode())
    element = struct.pack("<II", 15, len(compressed)) + compressed
    element += b"\x00" * (-len(element) % 8)
    return bytes(header) + element


def _small_records() -> tuple[MappingRecord, ...]:
    return canonical_mapping(
        b"1\n2\n1\n2\n",
        ("n00000001", "n00000002"),
        expected_images=4,
        expected_per_class=2,
    )


def test_official_archive_hashes_and_independent_revision_are_pinned() -> None:
    assert OFFICIAL_DEVKIT_MD5 == "fa75699e90414af021442c21a62c3abf"
    assert OFFICIAL_DEVKIT_SHA256 == (
        "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
    )
    assert OFFICIAL_VAL_ARCHIVE_MD5 == "29b22e2961454d5413ddabcf34fc5622"
    assert OFFICIAL_VAL_ARCHIVE_SHA256 == (
        "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0"
    )
    assert OFFICIAL_CLASS_ID_TO_SYNSET_SHA256 == (
        "33cac7974cc0bb3935c6fca46b93729c29a6481fa635e92f5f0852859aecd361"
    )
    assert OFFICIAL_FILENAME_TO_SYNSET_TSV_SHA256 == (
        "b5b25a74f93140f3e3febc504cd1e77411d38604e61da4801e2cde971771ba54"
    )
    assert TORCHVISION_COMMIT == "8fb87713a24951e639c494b0f2a8a81b5f8e33a6"


def test_dependency_free_meta_parser_preserves_ilsvrc_id_order() -> None:
    wnids = ("n00000003", "n00000001", "n00000002")
    meta = _mat_v5_with_wnids(wnids)
    assert (
        class_synsets_from_meta(
            meta,
            expected_synsets=3,
            expected_classes=2,
        )
        == wnids[:2]
    )


def test_ground_truth_lines_become_exact_filenames_and_synsets() -> None:
    records = _small_records()
    assert records == (
        MappingRecord("ILSVRC2012_val_00000001.JPEG", 1, "n00000001"),
        MappingRecord("ILSVRC2012_val_00000002.JPEG", 2, "n00000002"),
        MappingRecord("ILSVRC2012_val_00000003.JPEG", 1, "n00000001"),
        MappingRecord("ILSVRC2012_val_00000004.JPEG", 2, "n00000002"),
    )


def test_validation_tree_requires_every_exact_parent_assignment(tmp_path: Path) -> None:
    records = _small_records()
    for record in records:
        path = tmp_path / record.synset / record.filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(record.filename.encode())

    report = verify_validation_tree(tmp_path, records)
    assert report["image_count"] == 4
    assert report["class_count"] == 2
    assert report["exact_parent_synset_assignments"] is True

    misplaced = tmp_path / records[0].synset / records[0].filename
    wrong = tmp_path / records[1].synset / records[0].filename
    misplaced.replace(wrong)
    with pytest.raises(VerificationError, match="misassigned=1"):
        verify_validation_tree(tmp_path, records)


def test_small_flat_archive_can_represent_the_local_tree(tmp_path: Path) -> None:
    records = _small_records()
    archive_path = tmp_path / "val.tar"
    with tarfile.open(archive_path, "w") as archive:
        for record in records:
            value = record.filename.encode()
            local = tmp_path / "val" / record.synset / record.filename
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(value)
            member = tarfile.TarInfo(record.filename)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))

    hashes = file_hashes(archive_path)
    report = verify_images_against_archive(
        archive_path,
        tmp_path / "val",
        records,
        expected_md5=str(hashes["md5"]),
        expected_sha256=str(hashes["sha256"]),
    )
    assert report["archive_member_count"] == 4
    assert report["local_files_byte_identical_to_archive"] is True

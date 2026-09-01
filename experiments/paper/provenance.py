"""Immutable source and installed-environment evidence for paper references."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

REFERENCE_MANIFEST = (
    Path(__file__).parents[2] / "benchmarks" / "configs" / "paper-references-v1.json"
)


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    name: str
    repository: str
    commit: str
    release: str | None = None
    distribution: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _commit(value: Any) -> str:
    commit = str(value)
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("reference commits must be full lowercase Git SHAs")
    return commit


def reference_source(
    name: str,
    *,
    manifest: Path = REFERENCE_MANIFEST,
) -> ReferenceSource:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != "representax-paper-references-v1":
        raise ValueError("unsupported paper reference schema")
    try:
        value = document["references"][name]
    except KeyError as error:
        raise KeyError(f"unknown paper reference {name!r}") from error
    return ReferenceSource(
        name=name,
        repository=str(value["repository"]),
        commit=_commit(value["commit"]),
        release=str(value["release"]) if "release" in value else None,
        distribution=(str(value["distribution"]) if "distribution" in value else None),
    )


def installed_reference_provenance(name: str) -> dict[str, Any]:
    """Verify a packaged reference and fingerprint its installed distribution."""

    source = reference_source(name)
    if source.distribution is None:
        raise ValueError(f"reference {name!r} is not an installed distribution")
    try:
        installed = distribution(source.distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"reference distribution {source.distribution!r} is not installed"
        ) from error
    if source.release is not None and installed.version != source.release:
        raise RuntimeError(
            f"expected {source.distribution}=={source.release}, "
            f"found {installed.version}"
        )

    direct_url: dict[str, Any] | None = None
    record_sha256: str | None = None
    installer: str | None = None
    wheel: tuple[str, ...] = ()
    for relative in installed.files or ():
        path = Path(installed.locate_file(relative))
        if relative.name == "direct_url.json" and path.is_file():
            direct_url = json.loads(path.read_text(encoding="utf-8"))
        elif relative.name == "RECORD" and path.is_file():
            record_sha256 = "sha256:" + _sha256(path)
        elif relative.name == "INSTALLER" and path.is_file():
            installer = path.read_text(encoding="utf-8").strip()
        elif relative.name == "WHEEL" and path.is_file():
            wheel = tuple(
                line.partition(":")[2].strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("Tag:")
            )
    return {
        "name": source.name,
        "repository": source.repository,
        "commit": source.commit,
        "release": source.release,
        "distribution": source.distribution,
        "installed_version": installed.version,
        "installed_record_sha256": record_sha256,
        "installer": installer,
        "wheel_tags": wheel,
        "direct_url": direct_url,
    }


def checkout_reference_provenance(name: str, checkout: Path) -> dict[str, Any]:
    """Verify a clean source checkout at the frozen reference commit."""

    source = reference_source(name)

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", "-C", str(checkout), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    actual = git("rev-parse", "HEAD")
    if actual != source.commit:
        raise RuntimeError(f"expected {name} commit {source.commit}, found {actual}")
    dirty = bool(git("status", "--porcelain"))
    if dirty:
        raise RuntimeError(f"reference checkout {checkout} has local changes")
    remotes = {
        remote: git("remote", "get-url", remote)
        for remote in git("remote").splitlines()
        if remote
    }
    return {
        "name": source.name,
        "repository": source.repository,
        "commit": actual,
        "release": source.release,
        "distribution": source.distribution,
        "checkout": str(checkout.resolve()),
        "remotes": remotes,
        "dirty": False,
    }


def reference_runtime_provenance(
    name: str,
    *,
    checkout: Path | None = None,
) -> dict[str, Any]:
    """Describe the exact reference source and process that executed a run."""

    source = reference_source(name)
    packaged = (
        installed_reference_provenance(name)
        if source.distribution is not None
        else None
    )
    checked_out = (
        checkout_reference_provenance(name, checkout) if checkout is not None else None
    )
    return {
        "schema_version": "representax-reference-provenance-v1",
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_abi": sys.abiflags,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "source": {
            "name": source.name,
            "repository": source.repository,
            "commit": source.commit,
            "release": source.release,
        },
        "distribution": packaged,
        "checkout": checked_out,
    }


def write_reference_result(
    path: Path,
    report: Mapping[str, Any],
    *,
    reference: str,
    checkout: Path | None = None,
) -> dict[str, Any]:
    """Write one report plus a deduplicated content-addressed provenance record."""

    provenance = reference_runtime_provenance(reference, checkout=checkout)
    canonical = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fingerprint = "sha256:" + hashlib.sha256(canonical).hexdigest()
    sidecar = path.parent / "provenance" / f"{fingerprint.removeprefix('sha256:')}.json"
    _atomic_json(sidecar, {**provenance, "fingerprint": fingerprint})
    result = {
        **report,
        "reference": {"id": reference, "provenance": fingerprint},
    }
    _atomic_json(path, result)
    return result


__all__ = [
    "REFERENCE_MANIFEST",
    "ReferenceSource",
    "checkout_reference_provenance",
    "installed_reference_provenance",
    "reference_source",
    "reference_runtime_provenance",
    "write_reference_result",
]

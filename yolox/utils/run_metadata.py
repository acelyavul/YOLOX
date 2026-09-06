#!/usr/bin/env python3
"""Persist reproducibility metadata beside YOLOX training artifacts."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Union

import torch


PathLike = Union[str, os.PathLike]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution: str) -> Optional[str]:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _yolox_revision() -> Optional[str]:
    """Return the checked-out YOLOX Git revision when available."""
    yolox_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(yolox_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_record(path: Path) -> dict:
    """Build a path and digest record for one artifact."""
    resolved_path = path.resolve()
    return {
        "path": str(resolved_path),
        "sha256": _sha256_file(resolved_path),
    }


def write_run_metadata(
    run_directory: PathLike,
    batch_size: int,
    seed: Optional[int],
    start_checkpoint: Optional[PathLike],
    artifacts: Optional[Mapping[str, PathLike]] = None,
) -> Path:
    """Write run metadata atomically inside the experiment run directory."""
    run_path = Path(run_directory).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    checkpoint_record = None
    if start_checkpoint is not None:
        checkpoint_record = _file_record(Path(start_checkpoint))

    artifact_records = {}
    for name, artifact_path in sorted((artifacts or {}).items()):
        path = Path(artifact_path)
        if path.is_file():
            artifact_records[name] = _file_record(path)

    metadata = {
        "run_directory": str(run_path),
        "seed": seed,
        "yolox_revision": _yolox_revision(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "batch_size": batch_size,
        "start_checkpoint": checkpoint_record,
        "command": [sys.executable, *sys.argv],
        "artifacts": artifact_records,
    }

    metadata_path = run_path / "run_metadata.json"
    temporary_path = run_path / ".run_metadata.json.tmp"
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, metadata_path)
    return metadata_path

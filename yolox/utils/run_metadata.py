#!/usr/bin/env python3
"""Persist reproducibility metadata beside YOLOX training artifacts."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Union

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
    """Build a path, digest, and size record for one artifact."""
    resolved_path = path.resolve()
    return {
        "path": str(resolved_path),
        "sha256": _sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def _runtime_metadata() -> dict:
    """Return runtime library and accelerator information."""
    return {
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "mlflow_version": _package_version("mlflow"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> Path:
    """Write a JSON object atomically with deterministic key ordering."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            document,
            output_file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    os.replace(temporary_path, path)
    return path


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
        "batch_size": batch_size,
        "start_checkpoint": checkpoint_record,
        "command": [sys.executable, *sys.argv],
        "artifacts": artifact_records,
    }
    metadata.update(_runtime_metadata())

    metadata_path = run_path / "run_metadata.json"
    return _write_json(metadata_path, metadata)


def write_evaluation_results(
    run_directory: PathLike,
    metrics: Mapping[str, Optional[float]],
    evaluation_run_id: Optional[str],
    source_training_run_id: Optional[str],
    provenance_mode: Optional[str],
    workflow_stage: str,
    tested_checkpoint: PathLike,
    test_split: PathLike,
    test_split_display_path: str,
    log_file: PathLike,
) -> Path:
    """Write the evaluation result document beside the evaluation log."""
    run_path = Path(run_directory).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_record = _file_record(Path(tested_checkpoint))
    test_split_record = _file_record(Path(test_split))
    log_record = _file_record(Path(log_file))

    document = {
        "schema_version": 1,
        "workflow_stage": workflow_stage,
        "provenance_mode": provenance_mode,
        "evaluation_run_id": evaluation_run_id,
        "source_training_run_id": source_training_run_id,
        "tested_checkpoint_sha256": checkpoint_record["sha256"],
        "test_split": {
            "annotation_path": test_split_display_path,
            "annotation_sha256": test_split_record["sha256"],
        },
        "evaluation_log": {
            "path": Path(log_file).name,
            "sha256": log_record["sha256"],
        },
        "metrics_source": Path(log_file).name,
        "metrics": dict(metrics),
    }
    return _write_json(run_path / "evaluation_results.json", document)


def write_evaluation_run_metadata(
    run_directory: PathLike,
    batch_size: int,
    devices: Optional[int],
    seed: Optional[int],
    fp16: bool,
    experiment_id: str,
    tested_checkpoint: PathLike,
    test_split: PathLike,
    confidence_threshold: float,
    nms_threshold: float,
    evaluation_completed_at: str,
    mlflow_experiment_name: Optional[str],
    mlflow_run_name: Optional[str],
    mlflow_run_id: Optional[str],
    source_training_run_id: Optional[str],
    provenance_mode: Optional[str],
    workflow_stage: str,
    artifacts: Mapping[str, PathLike],
) -> Path:
    """Write runtime and provenance metadata for one evaluation process."""
    run_path = Path(run_directory).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_records = {
        name: _file_record(Path(artifact_path))
        for name, artifact_path in sorted(artifacts.items())
        if Path(artifact_path).is_file()
    }
    metadata = {
        "schema_version": 1,
        "workflow_stage": workflow_stage,
        "provenance_mode": provenance_mode,
        "run_directory": str(run_path),
        "experiment_id": experiment_id,
        "evaluation_completed_at": evaluation_completed_at,
        "seed": seed,
        "batch_size": batch_size,
        "devices": devices,
        "fp16": fp16,
        "test_confidence_threshold": confidence_threshold,
        "nms_threshold": nms_threshold,
        "source_training_run_id": source_training_run_id,
        "mlflow_experiment_name": mlflow_experiment_name,
        "mlflow_run_name": mlflow_run_name,
        "mlflow_run_id": mlflow_run_id,
        "tested_checkpoint": _file_record(Path(tested_checkpoint)),
        "test_split": _file_record(Path(test_split)),
        "yolox_revision": _yolox_revision(),
        "command": [sys.executable, *sys.argv],
        "artifacts": artifact_records,
    }
    metadata.update(_runtime_metadata())
    return _write_json(run_path / "run_metadata.json", metadata)

#!/usr/bin/env python3
"""Run two isolated YOLOX micro-trainings and compare deterministic outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


YOLOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIRECTORY = PROJECT_ROOT / "images" / "wheel"
DEFAULT_ANNOTATION_FILE = (
    PROJECT_ROOT
    / "cvat"
    / "wheel-detection-and-rim-segmentation-dataset"
    / "train-batch-003"
    / "instances_default.json"
)
DEFAULT_OUTPUT_DIRECTORY = YOLOX_ROOT / "YOLOX_outputs" / "determinism_smoke"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_sort_key(path: Path) -> list[int | str]:
    """Return a case-insensitive key that orders numeric filename parts naturally."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_coco_document(annotation_file: Path) -> dict[str, Any]:
    """Load one COCO annotation document."""
    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
    with annotation_file.open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a COCO object: {annotation_file}")
    return document


def select_source_images(
    source_directory: Path,
    document: dict[str, Any],
    count: int,
) -> list[Path]:
    """Select naturally ordered source images that have COCO annotations."""
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_directory}")

    annotated_image_ids = {
        int(annotation["image_id"])
        for annotation in document.get("annotations", [])
    }
    images = sorted(
        (
            source_directory / Path(image["file_name"]).name
            for image in document.get("images", [])
            if int(image["id"]) in annotated_image_ids
        ),
        key=natural_sort_key,
    )
    images = [
        path
        for path in images
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if len(images) < count:
        raise ValueError(
            f"At least {count} annotated source images are required; found "
            f"{len(images)} in {source_directory}."
        )
    return images[:count]


def stage_coco_dataset(
    source_images: list[Path],
    documents: list[dict[str, Any]],
    dataset_directory: Path,
) -> dict[str, Any]:
    """Build an isolated COCO dataset without modifying source images or exports."""
    images_by_name: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    categories = documents[0].get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO categories are missing.")

    for document in documents:
        if document.get("categories") != categories:
            raise ValueError("COCO category definitions differ across annotation files.")
        document_images_by_id = {
            int(image["id"]): image for image in document.get("images", [])
        }
        document_annotations: dict[int, list[dict[str, Any]]] = {
            image_id: [] for image_id in document_images_by_id
        }
        for annotation in document.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in document_annotations:
                raise ValueError(
                    f"COCO annotation references an unknown image ID: {image_id}"
                )
            document_annotations[image_id].append(annotation)
        for image in document.get("images", []):
            name = Path(image["file_name"]).name
            if name in images_by_name:
                raise ValueError(f"Duplicate COCO image filename: {name}")
            images_by_name[name] = (
                image,
                document_annotations[int(image["id"])],
            )

    selected_records = []
    selected_annotations = []
    destination_images = dataset_directory / "train2017"
    destination_annotations = dataset_directory / "annotations"
    destination_images.mkdir(parents=True)
    destination_annotations.mkdir(parents=True)

    next_annotation_id = 1
    for next_image_id, source_path in enumerate(source_images, start=1):
        image_and_annotations = images_by_name.get(source_path.name)
        if image_and_annotations is None:
            raise ValueError(f"Source image has no COCO record: {source_path.name}")
        image, image_annotations = image_and_annotations
        if not image_annotations:
            raise ValueError(f"Source image has no COCO annotations: {source_path.name}")
        staged_image = dict(image)
        staged_image["id"] = next_image_id
        staged_image["file_name"] = source_path.name
        selected_records.append(staged_image)
        for annotation in image_annotations:
            staged_annotation = dict(annotation)
            staged_annotation["id"] = next_annotation_id
            staged_annotation["image_id"] = next_image_id
            selected_annotations.append(staged_annotation)
            next_annotation_id += 1
        destination_path = destination_images / source_path.name
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)

    staged_document = {
        "info": documents[0].get("info", {}),
        "licenses": documents[0].get("licenses", []),
        "images": selected_records,
        "annotations": selected_annotations,
        "categories": categories,
    }
    annotation_path = destination_annotations / "instances_train2017.json"
    with annotation_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(staged_document, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return staged_document


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def yolox_revision() -> str | None:
    """Return the checked-out YOLOX Git revision when available."""
    result = subprocess.run(
        ["git", "-C", str(YOLOX_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def update_tensor_hash(digest: Any, name: str, tensor: Any) -> None:
    """Add a named tensor to a canonical state digest."""
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Hash state tensors independently of the torch.save container."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        update_tensor_hash(digest, name, state_dict[name])
    return digest.hexdigest()


def batch_sha256(inputs: Any, targets: Any, image_ids: Any) -> str:
    """Hash one augmented DataLoader batch."""
    digest = hashlib.sha256()
    update_tensor_hash(digest, "inputs", inputs)
    update_tensor_hash(digest, "targets", targets)
    update_tensor_hash(digest, "image_ids", image_ids)
    return digest.hexdigest()


def load_initial_weights(model: Any, checkpoint_path: Path | None) -> None:
    """Load an optional YOLOX model checkpoint."""
    if checkpoint_path is None:
        return
    import torch

    from yolox.utils import load_ckpt

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    load_ckpt(model, state_dict)


def run_training_child(arguments: argparse.Namespace) -> None:
    """Execute one isolated deterministic micro-training."""
    sys.path.insert(0, str(YOLOX_ROOT))

    import cv2
    import numpy as np
    import torch

    from yolox.exp import Exp as YOLOXExp
    from yolox.utils import configure_determinism

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the YOLOX determinism smoke test.")

    configure_determinism(arguments.seed)
    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    class SmokeExp(YOLOXExp):
        """Minimal YOLOX-S configuration using the staged COCO subset."""

        def __init__(self) -> None:
            super().__init__()
            self.depth = 0.33
            self.width = 0.50
            self.num_classes = arguments.num_classes
            self.input_size = (arguments.input_size, arguments.input_size)
            self.test_size = self.input_size
            self.multiscale_range = 2
            self.data_dir = str(arguments.dataset_directory)
            self.train_ann = "instances_train2017.json"
            self.seed = arguments.seed
            self.data_num_workers = arguments.workers
            self.mosaic_prob = 1.0
            self.mixup_prob = 1.0
            self.hsv_prob = 1.0
            self.flip_prob = 0.5
            self.degrees = 10.0
            self.translate = 0.1
            self.mosaic_scale = (0.1, 2.0)
            self.enable_mixup = True
            self.mixup_scale = (0.5, 1.5)
            self.shear = 2.0

    experiment = SmokeExp()
    model = experiment.get_model()
    load_initial_weights(model, arguments.checkpoint)
    initial_model_state_digest = state_dict_sha256(model.state_dict())
    model.cuda()
    model.train()
    optimizer = experiment.get_optimizer(arguments.batch_size)
    data_loader = experiment.get_data_loader(
        batch_size=arguments.batch_size,
        is_distributed=False,
        no_aug=False,
    )
    data_iterator = iter(data_loader)
    learning_rate_scheduler = experiment.get_lr_scheduler(
        experiment.basic_lr_per_img * arguments.batch_size,
        len(data_loader),
    )
    input_size = experiment.input_size
    trace = []
    metrics = []

    for step in range(arguments.steps):
        inputs, targets, _, image_ids = next(data_iterator)
        trace.append(
            {
                "step": step + 1,
                "input_size": list(input_size),
                "batch_sha256": batch_sha256(inputs, targets, image_ids),
            }
        )

        inputs = inputs.cuda(non_blocking=False).float()
        targets = targets.cuda(non_blocking=False).float()
        inputs, targets = experiment.preprocess(inputs, targets, input_size)
        outputs = model(inputs, targets)
        loss = outputs["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        metrics.append(
            {
                "step": step + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_loss": float(loss.detach().cpu()),
                "iou_loss": float(outputs["iou_loss"].detach().cpu()),
                "conf_loss": float(outputs["conf_loss"].detach().cpu()),
                "cls_loss": float(outputs["cls_loss"].detach().cpu()),
            }
        )
        learning_rate = learning_rate_scheduler.update_lr(step + 1)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        if (step + 1) % 10 == 0:
            input_size = experiment.random_resize(
                data_loader,
                epoch=step // max(len(data_loader), 1),
                rank=0,
                is_distributed=False,
            )

    arguments.run_directory.mkdir(parents=True, exist_ok=False)
    trace_path = arguments.run_directory / "augmentation_trace.json"
    metrics_path = arguments.run_directory / "metrics.json"
    checkpoint_path = arguments.run_directory / "model.pth"
    with trace_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(trace, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with metrics_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(metrics, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    model_state = model.state_dict()
    model_state_digest = state_dict_sha256(model_state)
    torch.save({"model": model_state}, checkpoint_path)
    metadata = {
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "steps": arguments.steps,
        "workers": arguments.workers,
        "input_size": arguments.input_size,
        "start_checkpoint": (
            str(arguments.checkpoint) if arguments.checkpoint is not None else None
        ),
        "start_checkpoint_sha256": (
            sha256_file(arguments.checkpoint)
            if arguments.checkpoint is not None
            else None
        ),
        "yolox_revision": yolox_revision(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "command": sys.argv,
        "augmentation_trace_sha256": sha256_file(trace_path),
        "metrics_sha256": sha256_file(metrics_path),
        "initial_model_state_sha256": initial_model_state_digest,
        "model_state_sha256": model_state_digest,
        "weights_changed": model_state_digest != initial_model_state_digest,
        "model_artifact_sha256": sha256_file(checkpoint_path),
    }
    metadata_path = arguments.run_directory / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Run metadata: {metadata_path}")


def child_command(
    arguments: argparse.Namespace,
    dataset_directory: Path,
    run_directory: Path,
    num_classes: int,
) -> list[str]:
    """Build one isolated child-process command."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--dataset-directory",
        str(dataset_directory),
        "--run-directory",
        str(run_directory),
        "--num-classes",
        str(num_classes),
        "--seed",
        str(arguments.seed),
        "--batch-size",
        str(arguments.batch_size),
        "--steps",
        str(arguments.steps),
        "--workers",
        str(arguments.workers),
        "--input-size",
        str(arguments.input_size),
    ]
    if arguments.checkpoint is not None:
        command.extend(["--checkpoint", str(arguments.checkpoint.resolve())])
    return command


def compare_runs(run_directories: list[Path]) -> dict[str, Any]:
    """Compare augmentation, metric, tensor, and serialized artifact digests."""
    metadata = []
    for run_directory in run_directories:
        with (run_directory / "run_metadata.json").open(
            encoding="utf-8"
        ) as input_file:
            metadata.append(json.load(input_file))

    keys = [
        "augmentation_trace_sha256",
        "metrics_sha256",
        "initial_model_state_sha256",
        "model_state_sha256",
        "model_artifact_sha256",
    ]
    comparisons = {
        key: {
            "run_1": metadata[0][key],
            "run_2": metadata[1][key],
            "equal": metadata[0][key] == metadata[1][key],
        }
        for key in keys
    }
    weights_changed = all(
        run_metadata["weights_changed"] for run_metadata in metadata
    )
    return {
        "passed": (
            all(item["equal"] for item in comparisons.values())
            and weights_changed
        ),
        "weights_changed": weights_changed,
        "comparisons": comparisons,
    }


def run_parent(arguments: argparse.Namespace) -> None:
    """Stage data, launch two runs, and fail if any deterministic output differs."""
    if arguments.images < 5:
        raise ValueError("Use at least 5 images to exercise mosaic and mixup.")
    if arguments.steps < 1:
        raise ValueError("Training steps must be positive.")
    if arguments.input_size % 32 != 0:
        raise ValueError("Input size must be a multiple of 32.")

    document = load_coco_document(arguments.annotation_file)
    source_images = select_source_images(
        arguments.source_directory,
        document,
        arguments.images,
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    smoke_directory = Path(
        tempfile.mkdtemp(
            prefix="run_",
            dir=arguments.output_directory,
        )
    )
    source_manifest = [
        {"file": str(path.resolve()), "sha256": sha256_file(path)}
        for path in source_images
    ]
    with (smoke_directory / "source_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(source_manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    run_directories = [smoke_directory / "run_1", smoke_directory / "run_2"]
    with tempfile.TemporaryDirectory(
        prefix="yolox_determinism_dataset_"
    ) as temporary_dataset:
        dataset_directory = Path(temporary_dataset)
        staged_document = stage_coco_dataset(
            source_images,
            [document],
            dataset_directory,
        )

        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(arguments.seed)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(YOLOX_ROOT)
            if not python_path
            else os.pathsep.join((str(YOLOX_ROOT), python_path))
        )

        num_classes = len(staged_document["categories"])
        for run_directory in run_directories:
            command = child_command(
                arguments,
                dataset_directory,
                run_directory,
                num_classes,
            )
            print(f"Starting isolated run: {run_directory.name}")
            subprocess.run(command, check=True, env=environment, cwd=YOLOX_ROOT)

    report = compare_runs(run_directories)
    report.update(
        {
            "seed": arguments.seed,
            "selected_images": [path.name for path in source_images],
            "output_directory": str(smoke_directory.resolve()),
        }
    )
    report_path = smoke_directory / "comparison.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Comparison report: {report_path}")
    if not report["passed"]:
        raise SystemExit("Determinism smoke test failed.")
    print("Determinism smoke test passed.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLOX-S twice on an isolated COCO subset and compare exact "
            "augmentation, metric, and model digests."
        )
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=DEFAULT_ANNOTATION_FILE,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset-directory", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument("--run-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--num-classes", type=int, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Run the parent coordinator or one internal training child."""
    arguments = build_parser().parse_args()
    if arguments.child_run:
        required = {
            "dataset_directory": arguments.dataset_directory,
            "run_directory": arguments.run_directory,
            "num_classes": arguments.num_classes,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Missing child-run arguments: " + ", ".join(sorted(missing))
            )
        run_training_child(arguments)
    else:
        try:
            run_parent(arguments)
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()

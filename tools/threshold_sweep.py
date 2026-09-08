#!/usr/bin/env python3
"""Select a YOLOX confidence threshold on a frozen validation split."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


YOLOX_ROOT = Path(__file__).resolve().parents[1]
ROOT = YOLOX_ROOT.parents[1]
PROJECT_TOOLS = ROOT / "tools"
EXPERIMENTS_ROOT = ROOT / "experiments"
VALIDATION_SPLIT = "val"
WORKFLOW_STAGE = "validation_threshold_sweep"
PREDICTIONS_FILE = "validation_predictions.json"
RESULTS_FILE = "threshold_sweep_results.json"
RUN_METADATA_FILE = "run_metadata.json"
LOG_FILE = "val_log.txt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(YOLOX_ROOT))
sys.path.insert(0, str(PROJECT_TOOLS))

from experiment_selection import (  # noqa: E402
    load_experiment_configuration,
    select_experiment_directory,
)
from yolox.evaluators.coco_metrics import coco_recall_at_iou  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import configure_module, get_model_info  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse threshold-sweep arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOX once on validation data and select a confidence threshold."
        )
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment path under experiments/. If omitted, select interactively.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--inference-confidence", type=float, default=0.001)
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def write_json(path: Path, document: Any) -> Path:
    """Write canonical JSON atomically."""
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


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Build one artifact provenance record."""
    resolved_path = path.resolve()
    return {
        "path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(repository: Path) -> str | None:
    """Return the checked-out Git revision when available."""
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require_probability(value: Any, field: str) -> float:
    """Validate and return one probability value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric.")
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1.")
    return probability


def load_selection_policy(experiment: dict[str, Any]) -> dict[str, Any]:
    """Load deployment constraints from experiment.json."""
    deployment_gate = experiment.get("deployment_gate")
    if not isinstance(deployment_gate, dict):
        raise ValueError("deployment_gate must be an object.")

    selection = deployment_gate.get("confidence_selection")
    if not isinstance(selection, dict):
        raise ValueError("deployment_gate.confidence_selection must be an object.")
    if selection.get("dataset") != "validation":
        raise ValueError("Confidence selection dataset must be validation.")

    constraints = selection.get("constraint")
    if not isinstance(constraints, dict):
        raise ValueError("confidence_selection.constraint must be an object.")
    false_positive_limit = require_probability(
        constraints.get("negative_image_false_positive_rate_lte"),
        "negative_image_false_positive_rate_lte",
    )

    recall_fields = [
        field for field in deployment_gate if field.casefold().startswith("recall@iou")
    ]
    if len(recall_fields) != 1:
        raise ValueError("deployment_gate must contain exactly one recall@IoU field.")
    recall_field = recall_fields[0]
    try:
        iou_threshold = float(recall_field.casefold().split("recall@iou", 1)[1])
    except ValueError as error:
        raise ValueError(f"Invalid recall IoU field: {recall_field}") from error

    return {
        "iou_threshold": require_probability(iou_threshold, "recall IoU"),
        "recall_gte": require_probability(
            deployment_gate[recall_field], recall_field
        ),
        "negative_image_false_positive_rate_lte": false_positive_limit,
        "objective": selection.get("objective"),
    }


def flatten_predictions(output_data: dict[int, Any]) -> list[dict[str, Any]]:
    """Convert YOLOX image-wise output into canonical COCO detections."""
    predictions = []
    for image_id in sorted(output_data):
        image_output = output_data[image_id]
        for box, score, category_id in zip(
            image_output["bboxes"],
            image_output["scores"],
            image_output["categories"],
        ):
            x_min, y_min, x_max, y_max = box
            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(x_max - x_min),
                        float(y_max - y_min),
                    ],
                    "score": float(score),
                }
            )
    predictions.sort(
        key=lambda prediction: (
            prediction["image_id"],
            -prediction["score"],
            prediction["category_id"],
            prediction["bbox"],
        )
    )
    return predictions


def load_coco(annotation_path: Path) -> COCO:
    """Load a COCO annotation file without writing console output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(annotation_path))


def validation_counts(coco_ground_truth: COCO) -> dict[str, Any]:
    """Count regular targets and negative validation images."""
    target_category_ids = set(coco_ground_truth.getCatIds())
    regular_annotations = [
        annotation
        for annotation in coco_ground_truth.dataset.get("annotations", [])
        if annotation.get("category_id") in target_category_ids
        and not annotation.get("iscrowd", 0)
        and not annotation.get("ignore", 0)
    ]
    positive_image_ids = {
        int(annotation["image_id"]) for annotation in regular_annotations
    }
    all_image_ids = {int(image_id) for image_id in coco_ground_truth.getImgIds()}
    negative_image_ids = all_image_ids - positive_image_ids
    return {
        "image_count": len(all_image_ids),
        "positive_image_count": len(positive_image_ids),
        "negative_image_count": len(negative_image_ids),
        "regular_ground_truth_count": len(regular_annotations),
        "negative_image_ids": negative_image_ids,
    }


def recall_at_iou(
    coco_ground_truth: COCO,
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> float:
    """Calculate COCO maximum recall at one IoU threshold."""
    if not detections:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_detections = coco_ground_truth.loadRes(detections)
        evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
        evaluator.params.iouThrs = np.array([iou_threshold], dtype=np.float64)
        evaluator.evaluate()
        evaluator.accumulate()
    recall = coco_recall_at_iou(evaluator, iou_threshold)
    return 0.0 if recall is None else recall


def sweep_thresholds(
    coco_ground_truth: COCO,
    predictions: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    """Evaluate all confidence thresholds and select the best feasible row."""
    counts = validation_counts(coco_ground_truth)
    negative_image_ids = counts.pop("negative_image_ids")
    if not negative_image_ids:
        raise ValueError("Validation split contains no negative images.")

    rows = []
    for index in range(1, 100):
        threshold = index / 100
        detections = [
            prediction
            for prediction in predictions
            if prediction["score"] >= threshold
        ]
        detected_negative_ids = {
            prediction["image_id"]
            for prediction in detections
            if prediction["image_id"] in negative_image_ids
        }
        negative_rate = len(detected_negative_ids) / len(negative_image_ids)
        recall = recall_at_iou(
            coco_ground_truth,
            detections,
            policy["iou_threshold"],
        )
        eligible = (
            recall >= policy["recall_gte"]
            and negative_rate
            <= policy["negative_image_false_positive_rate_lte"]
        )
        rows.append(
            {
                "threshold": threshold,
                "recall_at_iou": recall,
                "negative_image_false_positive_rate": negative_rate,
                "negative_images_with_detections": len(detected_negative_ids),
                "detection_count": len(detections),
                "eligible": eligible,
            }
        )

    eligible_rows = [row for row in rows if row["eligible"]]
    selected = (
        max(
            eligible_rows,
            key=lambda row: (row["recall_at_iou"], -row["threshold"]),
        )
        if eligible_rows
        else None
    )
    return rows, selected, counts


def runtime_metadata() -> dict[str, Any]:
    """Collect runtime and accelerator metadata."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "pycocotools": package_version("pycocotools"),
        "mlflow_version": package_version("mlflow"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_models": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def load_mlflow_tags() -> dict[str, Any]:
    """Load optional MLflow tags from the standard environment variable."""
    raw_tags = os.getenv("MLFLOW_TAGS")
    if not raw_tags:
        return {}
    tags = json.loads(raw_tags)
    if not isinstance(tags, dict):
        raise ValueError("MLFLOW_TAGS must contain a JSON object.")
    return tags


def main() -> None:
    """Run validation inference, threshold selection, and MLflow logging."""
    arguments = parse_arguments()
    if os.getenv("MLFLOW_RUN_ID"):
        raise ValueError(
            "MLFLOW_RUN_ID must be unset because threshold selection creates a new run."
        )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is required.")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < arguments.inference_confidence < 0.01:
        raise ValueError("--inference-confidence must be greater than 0 and below 0.01.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for validation inference.")
    if arguments.device < 0 or arguments.device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device does not exist: {arguments.device}")

    checkpoint_path = arguments.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    experiment_directory = select_experiment_directory(
        EXPERIMENTS_ROOT,
        arguments.experiment,
    )
    experiment_path = experiment_directory / "experiment.json"
    experiment_configuration = load_experiment_configuration(experiment_directory)
    experiment_id = experiment_configuration.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(f"Missing experiment_id: {experiment_path}")
    policy = load_selection_policy(experiment_configuration)

    experiment_file = experiment_directory / "yolox_s.py"
    annotation_path = (
        experiment_directory
        / "project-export"
        / "annotations"
        / "instances_val2017.json"
    )
    if not experiment_file.is_file():
        raise FileNotFoundError(f"YOLOX experiment file not found: {experiment_file}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Validation annotation not found: {annotation_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_name = (
        f"{experiment_id}_{WORKFLOW_STAGE}"
        f"__{timestamp}_{os.getpid()}"
    )
    run_directory = experiment_directory / "runs" / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    log_path = run_directory / LOG_FILE
    predictions_path = run_directory / PREDICTIONS_FILE
    results_path = run_directory / RESULTS_FILE
    metadata_path = run_directory / RUN_METADATA_FILE

    configure_module()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.set_device(arguments.device)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or experiment_id
    mlflow.set_experiment(mlflow_experiment_name)
    tags = load_mlflow_tags()
    tags.update(
        {
            "workflow.stage": WORKFLOW_STAGE,
            "model.source_run_id": arguments.training_run_id,
            "dataset.role": VALIDATION_SPLIT,
        }
    )

    file_sink = logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}",
        encoding="utf-8",
    )
    selected = None
    try:
        active_run = mlflow.start_run(run_name=run_name, tags=tags)
        if active_run is not None:
            mlflow_run_id = active_run.info.run_id
            checkpoint = file_record(checkpoint_path)
            validation_split = file_record(annotation_path)
            experiment_record = file_record(experiment_path)
            yolox_experiment_record = file_record(experiment_file)
            logger.info(f"Threshold sweep run: {run_name}")
            logger.info(f"Checkpoint: {checkpoint_path}")
            logger.info(f"Validation annotation: {annotation_path}")

            experiment = get_exp(str(experiment_file), None)
            experiment.test_conf = arguments.inference_confidence
            model = experiment.get_model()
            logger.info(
                f"Model summary: {get_model_info(model, experiment.test_size)}"
            )
            model.cuda(arguments.device)
            model.eval()
            loaded_checkpoint = torch.load(
                checkpoint_path,
                map_location=f"cuda:{arguments.device}",
                weights_only=False,
            )
            model.load_state_dict(loaded_checkpoint["model"])

            evaluator = experiment.get_evaluator(
                arguments.batch_size,
                is_distributed=False,
                testdev=False,
                legacy=arguments.legacy,
            )
            evaluation, output_data = evaluator.evaluate(
                model,
                distributed=False,
                half=arguments.fp16,
                return_outputs=True,
            )
            _, _, summary = evaluation
            logger.info("\n" + summary)
            predictions = flatten_predictions(output_data)
            write_json(predictions_path, predictions)

            coco_ground_truth = load_coco(annotation_path)
            rows, selected, counts = sweep_thresholds(
                coco_ground_truth,
                predictions,
                policy,
            )
            if selected is None:
                logger.error("No confidence threshold satisfies the deployment gate.")
            else:
                logger.info(
                    "Selected threshold: {:.2f}; recall: {:.6f}; negative FP rate: "
                    "{:.6f}".format(
                        selected["threshold"],
                        selected["recall_at_iou"],
                        selected["negative_image_false_positive_rate"],
                    )
                )
    finally:
        logger.remove(file_sink)

    predictions_record = file_record(predictions_path)
    log_record = file_record(log_path)
    results = {
        "schema_version": 1,
        "workflow_stage": WORKFLOW_STAGE,
        "mlflow_run_id": mlflow_run_id,
        "source_training_run_id": arguments.training_run_id,
        "checkpoint": checkpoint,
        "validation_split": validation_split,
        "predictions": {
            **predictions_record,
            "detection_count": len(predictions),
        },
        "inference": {
            "confidence_threshold": arguments.inference_confidence,
            "nms_iou_threshold": experiment.nmsthre,
        },
        "criteria": policy,
        "validation_counts": counts,
        "selected": selected,
        "thresholds": rows,
        "evaluation_log": log_record,
    }
    write_json(results_path, results)

    metadata = {
        "schema_version": 1,
        "workflow_stage": WORKFLOW_STAGE,
        "run_directory": str(run_directory.resolve()),
        "run_name": run_name,
        "mlflow_experiment_name": mlflow_experiment_name,
        "mlflow_run_id": mlflow_run_id,
        "source_training_run_id": arguments.training_run_id,
        "experiment_id": experiment_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "seed": arguments.seed,
        "batch_size": arguments.batch_size,
        "device": arguments.device,
        "fp16": arguments.fp16,
        "inference_confidence_threshold": arguments.inference_confidence,
        "nms_iou_threshold": experiment.nmsthre,
        "checkpoint": checkpoint,
        "validation_split": validation_split,
        "experiment_config": experiment_record,
        "yolox_experiment_config": yolox_experiment_record,
        "sweep_code": file_record(Path(__file__)),
        "repository_revision": git_revision(ROOT),
        "yolox_revision": git_revision(YOLOX_ROOT),
        "artifacts": {
            PREDICTIONS_FILE: predictions_record,
            RESULTS_FILE: file_record(results_path),
            LOG_FILE: log_record,
        },
    }
    metadata.update(runtime_metadata())
    write_json(metadata_path, metadata)

    mlflow.log_params(
        {
            "source_training_run_id": arguments.training_run_id,
            "checkpoint_sha256": checkpoint["sha256"],
            "validation_split_sha256": validation_split["sha256"],
            "experiment_config_sha256": experiment_record["sha256"],
            "yolox_experiment_config_sha256": yolox_experiment_record["sha256"],
            "inference_confidence_threshold": arguments.inference_confidence,
            "nms_iou_threshold": experiment.nmsthre,
            "iou_threshold": policy["iou_threshold"],
            "recall_gte": policy["recall_gte"],
            "negative_image_false_positive_rate_lte": policy[
                "negative_image_false_positive_rate_lte"
            ],
            "threshold_count": len(rows),
        }
    )
    for step, row in enumerate(rows, start=1):
        mlflow.log_metrics(
            {
                "sweep/threshold": row["threshold"],
                "sweep/recall_at_iou": row["recall_at_iou"],
                "sweep/negative_image_false_positive_rate": row[
                    "negative_image_false_positive_rate"
                ],
            },
            step=step,
        )
    if selected is not None:
        mlflow.log_metrics(
            {
                "selected/confidence_threshold": selected["threshold"],
                "selected/recall_at_iou": selected["recall_at_iou"],
                "selected/negative_image_false_positive_rate": selected[
                    "negative_image_false_positive_rate"
                ],
            }
        )
    for artifact_path in (predictions_path, results_path, metadata_path, log_path):
        mlflow.log_artifact(str(artifact_path), artifact_path="threshold_sweep")

    print(f"Threshold sweep output: {run_directory}")
    if selected is None:
        mlflow.end_run(status="FAILED")
        raise RuntimeError("No confidence threshold satisfies the deployment gate.")
    mlflow.end_run(status="FINISHED")
    print(f"Selected confidence threshold: {selected['threshold']:.2f}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise

#!/usr/bin/env python3
"""Run production YOLOX evaluation on four real test images."""

import json
import os
import random
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import torch


YOLOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = YOLOX_ROOT.parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "yolox_wheel" / "v1"
EXPERIMENT_FILE = EXPERIMENT_ROOT / "yolox_s.py"
SOURCE_DATASET = EXPERIMENT_ROOT / "project-export"
SOURCE_ANNOTATION = SOURCE_DATASET / "annotations" / "instances_test2017.json"
CHECKPOINT = (
    EXPERIMENT_ROOT
    / "runs"
    / "yolox_wheel_v1__20260906T220653.269306Z_6444"
    / "best_ckpt.pth"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "evaluation"

sys.path.insert(0, str(YOLOX_ROOT))

from tools.eval import main as production_eval  # noqa: E402
from tools.eval import make_parser  # noqa: E402
from yolox.exp import get_exp  # noqa: E402


def load_json(path):
    """Load a JSON document."""
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def stage_real_subset(run_directory):
    """Stage four random real images from the test split."""
    source = load_json(SOURCE_ANNOTATION)
    images = source["images"]
    if len(images) < 4:
        raise RuntimeError("Four test images are required.")
    selected = random.sample(images, k=4)

    selected_ids = {image["id"] for image in selected}
    staged_images = []
    for image in selected:
        source_path = SOURCE_DATASET / "test2017" / image["file_name"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Test image not found: {source_path}")
        staged_image = dict(image)
        staged_image["file_name"] = str(source_path.resolve())
        staged_images.append(staged_image)

    subset = dict(source)
    subset["images"] = staged_images
    subset["annotations"] = [
        annotation
        for annotation in source["annotations"]
        if annotation["image_id"] in selected_ids
    ]

    dataset = run_directory / "dataset"
    annotation_directory = dataset / "annotations"
    annotation_directory.mkdir(parents=True)

    annotation_path = annotation_directory / "instances_test2017.json"
    with annotation_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(subset, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    with (run_directory / "selected_images.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(
            [image["file_name"] for image in selected],
            output_file,
            indent=2,
        )
        output_file.write("\n")
    return dataset


class TestEvaluationInference(unittest.TestCase):
    """Test the production evaluator with the trained wheel model."""

    def test_production_evaluation(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA is required.")
        self.assertTrue(CHECKPOINT.is_file(), f"Checkpoint not found: {CHECKPOINT}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_name = f"run_{timestamp}_{os.getpid()}"
        run_directory = OUTPUT_ROOT / run_name
        run_directory.mkdir(parents=True)
        dataset = stage_real_subset(run_directory)

        experiment = get_exp(str(EXPERIMENT_FILE), None)
        experiment.data_dir = str(dataset)
        experiment.output_dir = str(OUTPUT_ROOT)
        experiment.data_num_workers = 0
        arguments = make_parser().parse_args(
            [
                "-l", "none",
                "-f", str(EXPERIMENT_FILE),
                "-expn", run_name,
                "-d", "1",
                "-b", "2",
                "--fp16",
                "--seed", "42",
                "--test",
                "-c", str(CHECKPOINT),
            ]
        )

        previous_directory = Path.cwd()
        try:
            os.chdir(run_directory)
            production_eval(experiment, arguments, num_gpu=1)
        finally:
            os.chdir(previous_directory)

        results_path = run_directory / "evaluation_results.json"
        metadata_path = run_directory / "run_metadata.json"
        log_path = run_directory / "val_log.txt"
        self.assertTrue(results_path.is_file())
        self.assertTrue(metadata_path.is_file())
        self.assertTrue(log_path.is_file())

        results = load_json(results_path)
        self.assertEqual("evaluation", results["workflow_stage"])
        self.assertEqual(
            {
                "ap50",
                "ap50_95",
                "ap75",
                "negative_image_false_positive_rate",
                "recall_at_iou_0_75",
            },
            set(results["metrics"]),
        )
        for metric_name in ("ap50", "ap50_95", "ap75"):
            self.assertIsNotNone(results["metrics"][metric_name])
        print(f"Evaluation output: {run_directory}")


if __name__ == "__main__":
    unittest.main()

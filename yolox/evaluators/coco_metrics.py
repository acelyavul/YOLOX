#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

from enum import Enum

import numpy as np


class COCOAPMetric(str, Enum):
    """COCO average-precision metrics supported for checkpoint selection."""

    AP50_95 = "AP@0.50:0.95"
    AP50 = "AP@0.50"
    AP75 = "AP@0.75"


class COCOEvaluationMetric(str, Enum):
    """Additional COCO evaluation metrics required by the project."""

    RECALL_AT_IOU_0_75 = "recall@IoU0.75"
    NEGATIVE_IMAGE_FALSE_POSITIVE_RATE = "negative_image_false_positive_rate"


def resolve_coco_ap_metric(target_metric):
    """Resolve an experiment target metric to a supported COCO AP metric."""
    metric_name = target_metric.rsplit("/", 1)[-1]
    try:
        return COCOAPMetric(metric_name).value
    except ValueError as error:
        raise ValueError(
            f"Unsupported checkpoint selection metric: {metric_name}"
        ) from error


def coco_recall_at_iou(coco_eval, iou_threshold):
    """Return mean maximum recall for one IoU threshold and all object sizes."""
    threshold_indices = np.flatnonzero(
        np.isclose(coco_eval.params.iouThrs, iou_threshold)
    )
    if threshold_indices.size != 1:
        raise ValueError(
            f"COCO evaluation does not contain IoU threshold {iou_threshold}."
        )

    area_index = list(coco_eval.params.areaRngLbl).index("all")
    max_detections_index = list(coco_eval.params.maxDets).index(100)
    recall = coco_eval.eval["recall"][
        threshold_indices[0], :, area_index, max_detections_index
    ]
    valid_recall = recall[recall > -1]
    return float(np.mean(valid_recall)) if valid_recall.size else None


def negative_image_false_positive_rate(coco_ground_truth, detections):
    """Return the share of negative images containing an accepted detection."""
    positive_image_ids = {
        annotation["image_id"]
        for annotation in coco_ground_truth.dataset.get("annotations", [])
        if not annotation.get("iscrowd", 0)
        and not annotation.get("ignore", 0)
    }
    negative_image_ids = set(coco_ground_truth.getImgIds()) - positive_image_ids
    if not negative_image_ids:
        return None

    false_positive_image_ids = {
        detection["image_id"]
        for detection in detections
        if detection["image_id"] in negative_image_ids
    }
    return len(false_positive_image_ids) / len(negative_image_ids)

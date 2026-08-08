#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

from enum import Enum


class COCOAPMetric(str, Enum):
    """COCO average-precision metrics supported for checkpoint selection."""

    AP50_95 = "AP@0.50:0.95"
    AP50 = "AP@0.50"
    AP75 = "AP@0.75"


def resolve_coco_ap_metric(target_metric):
    """Resolve an experiment target metric to a supported COCO AP metric."""
    metric_name = target_metric.rsplit("/", 1)[-1]
    try:
        return COCOAPMetric(metric_name).value
    except ValueError as error:
        raise ValueError(
            f"Unsupported checkpoint selection metric: {metric_name}"
        ) from error

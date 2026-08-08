#!/usr/bin/env python3
"""Regression tests for COCO crowd-region handling in YOLOX."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from yolox.data import TrainTransform
from yolox.exp import Exp as YOLOXExp
from yolox.models import YOLOXHead


class TestIgnoreRegions(unittest.TestCase):
    """Verify ignore flags across preprocessing and detection losses."""

    def test_train_transform_preserves_ignore_flags(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        targets = np.array(
            [
                [8.0, 8.0, 24.0, 24.0, 0.0, 0.0],
                [32.0, 32.0, 48.0, 48.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        transform = TrainTransform(max_labels=4, flip_prob=0.0, hsv_prob=0.0)
        _, transformed_targets = transform(image, targets, (64, 64))

        self.assertEqual((4, 6), transformed_targets.shape)
        np.testing.assert_array_equal(
            transformed_targets[:2, 5],
            np.array([0.0, 1.0], dtype=np.float32),
        )

    def test_objectness_mask_excludes_centers_inside_ignore_box(self):
        head = YOLOXHead(num_classes=1)
        ignore_boxes = torch.tensor([[12.0, 12.0, 16.0, 16.0]])
        expanded_strides = torch.full((1, 4), 8.0)
        x_shifts = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        y_shifts = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        mask = head.get_objectness_loss_mask(
            ignore_boxes,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors=4,
        )

        torch.testing.assert_close(
            mask,
            torch.tensor([False, False, False, True]),
        )

    def test_crowd_ground_truth_is_excluded_and_positive_takes_precedence(self):
        head = YOLOXHead(num_classes=1)
        outputs = torch.tensor(
            [
                [
                    [12.0, 12.0, 8.0, 8.0, 0.0, 0.0],
                    [20.0, 12.0, 8.0, 8.0, 0.0, 0.0],
                ]
            ]
        )
        labels = torch.zeros((1, 4, 6))
        labels[0, 0] = torch.tensor([0.0, 12.0, 12.0, 8.0, 8.0, 0.0])
        labels[0, 1] = torch.tensor([0.0, 16.0, 12.0, 32.0, 32.0, 1.0])
        x_shifts = [torch.tensor([[1.0, 2.0]])]
        y_shifts = [torch.tensor([[1.0, 1.0]])]
        expanded_strides = [torch.full((1, 2), 8.0)]
        assignment = (
            torch.tensor([0.0]),
            torch.tensor([True, False]),
            torch.tensor([1.0]),
            torch.tensor([0], dtype=torch.long),
            1,
        )

        with patch.object(head, "get_assignments", return_value=assignment) as mocked:
            losses = head.get_losses(
                imgs=None,
                x_shifts=x_shifts,
                y_shifts=y_shifts,
                expanded_strides=expanded_strides,
                labels=labels,
                outputs=outputs,
                origin_preds=[],
                dtype=torch.float32,
            )

        assignment_args = mocked.call_args.args
        self.assertEqual(1, assignment_args[1])
        self.assertEqual((1, 4), tuple(assignment_args[2].shape))
        torch.testing.assert_close(
            losses[2],
            torch.tensor(math.log(2.0)),
        )

    def test_multiscale_preprocess_preserves_ignore_flag(self):
        experiment = YOLOXExp()
        inputs = torch.zeros((1, 3, 640, 640))
        targets = torch.zeros((1, 2, 6))
        targets[0, 0] = torch.tensor([0.0, 320.0, 320.0, 64.0, 64.0, 1.0])

        _, scaled_targets = experiment.preprocess(inputs, targets, (480, 480))

        torch.testing.assert_close(
            scaled_targets[0, 0, 1:5],
            torch.tensor([240.0, 240.0, 48.0, 48.0]),
        )
        self.assertEqual(1.0, scaled_targets[0, 0, 5].item())


if __name__ == "__main__":
    unittest.main()

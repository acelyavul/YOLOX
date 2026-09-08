#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.evaluators.coco_metrics import COCOAPMetric, COCOEvaluationMetric
from yolox.exp import get_exp
from yolox.utils import (
    MlflowLogger,
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger,
    write_evaluation_results,
    write_evaluation_run_metadata,
)


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        choices=["none", "mlflow"],
        default="none",
        help="evaluation logger",
    )

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    ckpt_file = None
    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc, weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    mlflow_logger = None
    if rank == 0 and args.logger == "mlflow":
        mlflow_logger = MlflowLogger()
        mlflow_logger.setup(args=args, exp=exp)

    # start evaluate
    ap50_95, ap50, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    evaluator_metrics = getattr(evaluator, "metrics", {})
    ap75 = evaluator_metrics.get(COCOAPMetric.AP75.value)
    recall_at_iou_0_75 = evaluator_metrics.get(
        COCOEvaluationMetric.RECALL_AT_IOU_0_75.value
    )
    negative_image_false_positive_rate = evaluator_metrics.get(
        COCOEvaluationMetric.NEGATIVE_IMAGE_FALSE_POSITIVE_RATE.value
    )
    logger.info("\n" + summary)

    artifact_files = ()
    if rank == 0 and ckpt_file is not None:
        tags = mlflow_logger.tags if mlflow_logger is not None else {}
        evaluation_run_id = (
            mlflow_logger.run_id if mlflow_logger is not None else None
        )
        workflow_stage = tags.get("workflow.stage", "evaluation")
        provenance_mode = tags.get("provenance.mode")
        source_training_run_id = tags.get("model.source_run_id")
        annotation_name = exp.test_ann if args.test else exp.val_ann
        split_path = Path(exp.data_dir) / "annotations" / annotation_name
        split_display_path = str(
            Path(Path(exp.data_dir).name) / "annotations" / annotation_name
        ).replace("\\", "/")
        log_file = Path(file_name) / "val_log.txt"
        result_metrics = {
            "ap50": float(ap50),
            "ap50_95": float(ap50_95),
            "ap75": float(ap75) if ap75 is not None else None,
            "negative_image_false_positive_rate": (
                float(negative_image_false_positive_rate)
                if negative_image_false_positive_rate is not None
                else None
            ),
            "recall_at_iou_0_75": (
                float(recall_at_iou_0_75)
                if recall_at_iou_0_75 is not None
                else None
            ),
        }
        evaluation_results_path = write_evaluation_results(
            run_directory=file_name,
            metrics=result_metrics,
            evaluation_run_id=evaluation_run_id,
            source_training_run_id=source_training_run_id,
            provenance_mode=provenance_mode,
            workflow_stage=workflow_stage,
            tested_checkpoint=ckpt_file,
            test_split=split_path,
            test_split_display_path=split_display_path,
            log_file=log_file,
        )
        run_metadata_path = write_evaluation_run_metadata(
            run_directory=file_name,
            batch_size=args.batch_size,
            devices=args.devices,
            seed=args.seed,
            fp16=args.fp16,
            experiment_id=exp.exp_name,
            tested_checkpoint=ckpt_file,
            test_split=split_path,
            confidence_threshold=exp.test_conf,
            nms_threshold=exp.nmsthre,
            evaluation_completed_at=datetime.now(timezone.utc).isoformat(),
            mlflow_experiment_name=(
                mlflow_logger.experiment_name
                if mlflow_logger is not None
                else None
            ),
            mlflow_run_name=(
                mlflow_logger.run_name if mlflow_logger is not None else None
            ),
            mlflow_run_id=evaluation_run_id,
            source_training_run_id=source_training_run_id,
            provenance_mode=provenance_mode,
            workflow_stage=workflow_stage,
            artifacts={
                "evaluation_results.json": evaluation_results_path,
                "val_log.txt": log_file,
            },
        )
        artifact_files = (
            str(log_file),
            str(evaluation_results_path),
            str(run_metadata_path),
        )

    if mlflow_logger is not None:
        mlflow_metrics = {
            "eval/COCOAP50_95": ap50_95,
            "eval/COCOAP50": ap50,
            "eval/negative_image_false_positive_rate": (
                negative_image_false_positive_rate
            ),
            "eval/recall_at_iou_0_75": recall_at_iou_0_75,
        }
        if ap75 is not None:
            mlflow_metrics["eval/COCOAP75"] = ap75
        mlflow_logger.on_eval_end(
            args=args,
            file_name=file_name,
            metrics=mlflow_metrics,
            artifact_files=artifact_files,
        )


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
        args.experiment_name = (
            mlflow_experiment_name
            if args.logger == "mlflow" and mlflow_experiment_name
            else exp.exp_name
        )

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )

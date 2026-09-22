"""
Utilities for running local YOLOX experiments.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import tempfile
import time
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "object_ctrl_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools.cocoeval import COCOeval
from torch.utils.data import SequentialSampler
from tqdm.auto import tqdm

from yolox.data import (
    COCODataset,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    TrainTransform,
    ValTransform,
    YoloBatchSampler,
    worker_init_reset_seed,
)
from yolox.exp import Exp as YOLOXExp
from yolox.models import YOLOX, YOLOPAFPN
from yolox.models.losses import IOUloss
from yolox.models.network_blocks import SiLU as YOLOXSiLU
import yolox.models.yolo_head as yolox_yolo_head
from yolox.models.yolo_head import YOLOXHead
import yolox.utils.boxes as yolox_boxes
from yolox.utils import LRScheduler, ModelEMA, postprocess, replace_module

from detection_common.utils.filepath import ensure_dir
from detection_common.utils.json_io import read_json
from detection_common.utils.urlhelper import cache_download

from .config import OUTPUT_ROOT

BASKETBALL_CLASSES = ("basketball",)
YOLOX_TINY_WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_tiny.pth"
)
YOLOX_NANO_WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_nano.pth"
)


class RunMode(StrEnum):
    """
    Supported modes for starting a YOLOX training run.
    """

    FRESH = "fresh"
    RESUME = "resume"


@dataclass(frozen=True)
class SplitSpec:
    """
    COCO annotation and image directory names for one dataset split.
    """

    split: str
    annotation_file: str
    image_dir: str


@dataclass(frozen=True)
class TrainingSettings:
    """
    Runtime settings for a YOLOX notebook run.
    """

    epochs: int
    batch_size: int
    train_batch_limit: int | None
    image_size: int
    seed: int
    smoke_run: bool
    show_progress: bool
    verbose_output: bool
    resume_run_dir: Path | None = None

    @property
    def run_mode(self) -> RunMode:
        """
        Derive the run mode from the optional resume directory.
        """
        if self.resume_run_dir is None:
            return RunMode.FRESH
        return RunMode.RESUME

    @property
    def resolved_resume_run_dir(self) -> Path:
        """
        Return the resume directory resolved from the output root.
        """
        if self.resume_run_dir is None:
            raise ValueError("A fresh run does not have a resume directory.")
        if self.resume_run_dir.is_absolute():
            return self.resume_run_dir
        return OUTPUT_ROOT / self.resume_run_dir


@dataclass(frozen=True)
class ResumeState:
    """
    State restored from a recoverable training checkpoint.
    """

    completed_epoch: int
    scheduler_position: int
    best_map: float
    curr_map: float
    history: list[dict[str, float]]
    mosaic_enabled: bool
    l1_enabled: bool
    validation_pending: bool
    pending_train_metrics: dict[str, float] | None


SPLITS = {
    "train": SplitSpec("train", "instances_train.json", "images/train"),
    "val": SplitSpec("val", "instances_val.json", "images/val"),
    "test": SplitSpec("test", "instances_test.json", "images/test"),
}


class YOLOXTinyExp(YOLOXExp):
    """
    YOLOX-Tiny experiment configuration for a COCO-format dataset.
    """

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        max_epoch: int,
        image_size: int,
        project_name: str,
        class_names: tuple[str, ...] = BASKETBALL_CLASSES,
        seed: int = 42,
        workers: int = 0,
    ) -> None:
        """
        Configure YOLOX-Tiny for the configured detection classes.
        """
        super().__init__()
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = len(class_names)
        self.input_size = (image_size, image_size)
        self.test_size = (image_size, image_size)
        self.multiscale_range = 0
        self.random_size = (image_size // 32, image_size // 32)
        self.mosaic_scale = (0.5, 1.5)
        self.data_dir = str(dataset_dir)
        self.output_dir = str(output_dir)
        self.exp_name = project_name

        self.train_ann = SPLITS["train"].annotation_file
        self.val_ann = SPLITS["val"].annotation_file
        self.test_ann = SPLITS["test"].annotation_file
        self.train_name = SPLITS["train"].image_dir
        self.val_name = SPLITS["val"].image_dir
        self.test_name = SPLITS["test"].image_dir

        self.max_epoch = max_epoch
        self.no_aug_epochs = min(15, max(0, max_epoch - 1))
        self.eval_interval = 1
        self.save_history_ckpt = False
        self.data_num_workers = workers
        self.seed = seed
        self.enable_mixup = False

    def random_resize(
        self,
        data_loader: DataLoader,
        epoch: int,
        rank: int,
        is_distributed: bool,
    ) -> tuple[int, int]:
        """
        Keep this experiment single-scale.
        """
        return self.input_size


class YOLOXNanoExp(YOLOXTinyExp):
    """
    YOLOX-Nano experiment configuration for a COCO-format dataset.
    """

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        max_epoch: int,
        image_size: int,
        project_name: str,
        class_names: tuple[str, ...] = BASKETBALL_CLASSES,
        seed: int = 42,
        workers: int = 0,
    ) -> None:
        """
        Configure YOLOX-Nano for the configured detection classes.
        """
        super().__init__(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            max_epoch=max_epoch,
            image_size=image_size,
            project_name=project_name,
            class_names=class_names,
            seed=seed,
            workers=workers,
        )
        self.width = 0.25
        self.depthwise = True

    def get_model(self) -> YOLOX:
        """
        Build the depthwise YOLOX-Nano network.
        """
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                depthwise=True,
                act=self.act,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                depthwise=True,
                act=self.act,
            )
            self.model = YOLOX(backbone, head)

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model


def env_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_optional_int(name: str, default: int | None) -> int | None:
    """
    Read an optional integer setting from the environment.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_optional_path(name: str) -> Path | None:
    """
    Read an optional filesystem path from the environment.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def training_settings_from_env(
    *,
    default_epochs: int = 100,
    default_batch_size: int = 8,
    default_image_size: int = 640,
    default_seed: int = 42,
    env_prefix: str = "YOLOX_TINY",
) -> TrainingSettings:
    """
    Read notebook training settings from environment variables.
    """
    smoke_run = os.environ.get(f"{env_prefix}_SMOKE", "0") == "1"
    return TrainingSettings(
        epochs=env_int(f"{env_prefix}_EPOCHS", 2 if smoke_run else default_epochs),
        batch_size=env_int(f"{env_prefix}_BATCH_SIZE", default_batch_size),
        train_batch_limit=env_optional_int(f"{env_prefix}_TRAIN_BATCH_LIMIT", None),
        image_size=env_int(f"{env_prefix}_IMAGE_SIZE", default_image_size),
        seed=env_int(f"{env_prefix}_SEED", default_seed),
        smoke_run=smoke_run,
        show_progress=os.environ.get(f"{env_prefix}_PROGRESS", "0") == "1",
        verbose_output=os.environ.get(f"{env_prefix}_VERBOSE", "0") == "1",
        resume_run_dir=env_optional_path(f"{env_prefix}_RESUME_RUN"),
    )


def patch_mps_compatibility() -> None:
    """
    Patch YOLOX tensor conversions that fail on Apple MPS.
    """

    def bboxes_iou(
        bboxes_a: torch.Tensor,
        bboxes_b: torch.Tensor,
        xyxy: bool = True,
    ) -> torch.Tensor:
        """
        Compute box IoUs without string-based tensor type conversion.
        """
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2,
                bboxes_b[:, :2] - bboxes_b[:, 2:] / 2,
            )
            br = torch.min(
                bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2,
                bboxes_b[:, :2] + bboxes_b[:, 2:] / 2,
            )
            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)

        en = (tl < br).to(dtype=tl.dtype).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def iou_loss_forward(
        self: IOUloss,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute YOLOX IoU loss without string-based tensor type conversion.
        """
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            pred[:, :2] - pred[:, 2:] / 2,
            target[:, :2] - target[:, 2:] / 2,
        )
        br = torch.min(
            pred[:, :2] + pred[:, 2:] / 2,
            target[:, :2] + target[:, 2:] / 2,
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).to(dtype=tl.dtype).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou**2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                pred[:, :2] - pred[:, 2:] / 2,
                target[:, :2] - target[:, 2:] / 2,
            )
            c_br = torch.max(
                pred[:, :2] + pred[:, 2:] / 2,
                target[:, :2] + target[:, 2:] / 2,
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise ValueError(f"Unsupported YOLOX IoU loss type: {self.loss_type}")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def get_output_and_grid(
        self: YOLOXHead,
        output: torch.Tensor,
        k: int,
        stride: int,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return decoded training output and a cached grid on the output device.
        """
        del dtype

        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        grid_changed = grid.shape[2:4] != output.shape[2:4]
        grid_changed = grid_changed or grid.device != output.device
        grid_changed = grid_changed or grid.dtype != output.dtype

        if grid_changed:
            yv, xv = torch.meshgrid(
                torch.arange(hsize, device=output.device),
                torch.arange(wsize, device=output.device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2)
            grid = grid.view(1, 1, hsize, wsize, 2).to(dtype=output.dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2)
        output = output.reshape(batch_size, hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(
        self: YOLOXHead,
        outputs: torch.Tensor,
        dtype: Any,
    ) -> torch.Tensor:
        """
        Decode inference outputs using tensors on the same device as outputs.
        """
        del dtype

        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides, strict=True):
            yv, xv = torch.meshgrid(
                torch.arange(hsize, device=outputs.device),
                torch.arange(wsize, device=outputs.device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid.to(dtype=outputs.dtype))
            strides.append(
                torch.full(
                    (*grid.shape[:2], 1),
                    stride,
                    device=outputs.device,
                    dtype=outputs.dtype,
                )
            )

        grids = torch.cat(grids, dim=1)
        strides = torch.cat(strides, dim=1)
        return torch.cat(
            [
                (outputs[..., 0:2] + grids) * strides,
                torch.exp(outputs[..., 2:4]) * strides,
                outputs[..., 4:],
            ],
            dim=-1,
        )

    yolox_boxes.bboxes_iou = bboxes_iou
    yolox_yolo_head.bboxes_iou = bboxes_iou
    IOUloss.forward = iou_loss_forward
    YOLOXHead.get_output_and_grid = get_output_and_grid
    YOLOXHead.decode_outputs = decode_outputs


def ensure_pretrained_checkpoint(path: Path, url: str = YOLOX_TINY_WEIGHTS_URL) -> Path:
    """
    Download a pretrained YOLOX checkpoint if it is not cached yet.
    """
    return cache_download(path, url)


def summarize_coco_split(dataset_dir: Path, split: str) -> dict[str, Any]:
    """
    Summarize one COCO split.
    """
    annotation_path = dataset_dir / "annotations" / f"instances_{split}.json"
    coco = read_json(annotation_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    labeled_image_ids = {annotation["image_id"] for annotation in annotations}
    category_names = [category["name"] for category in coco.get("categories", [])]

    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "background_images": len(images) - len(labeled_image_ids),
        "classes": ", ".join(category_names),
    }


def summarize_coco_dataset(
    dataset_dir: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> pd.DataFrame:
    """
    Summarize all requested COCO splits.
    """
    return pd.DataFrame(summarize_coco_split(dataset_dir, split) for split in splits)


def set_reproducibility(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discard_cached_exp_attribute(exp: YOLOXTinyExp, attr: str) -> None:
    """
    Remove a cached YOLOX experiment attribute when it exists.
    """
    try:
        delattr(exp, attr)
    except AttributeError:
        pass


def sync_device(device: torch.device) -> None:
    """
    Synchronize asynchronous accelerators before timing.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def tensor_to_float(value: torch.Tensor | float) -> float:
    """
    Convert a scalar tensor or Python number to a float.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def optional_stdout_suppression(
    verbose: bool,
) -> contextlib.AbstractContextManager[io.StringIO | None]:
    """
    Suppress noisy third-party stdout unless verbose output is requested.
    """
    if verbose:
        return contextlib.nullcontext(None)
    return contextlib.redirect_stdout(io.StringIO())


def build_train_loader(
    exp: YOLOXTinyExp,
    batch_size: int,
    cache_img: str | None = None,
    verbose: bool = False,
) -> DataLoader:
    """
    Build the YOLOX mosaic training loader.
    """
    with optional_stdout_suppression(verbose):
        raw_dataset = COCODataset(
            data_dir=exp.data_dir,
            json_file=exp.train_ann,
            name=exp.train_name,
            img_size=exp.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=exp.flip_prob,
                hsv_prob=exp.hsv_prob,
            ),
            cache=cache_img is not None,
            cache_type=cache_img or "ram",
        )
    dataset = MosaicDetection(
        dataset=raw_dataset,
        mosaic=True,
        img_size=exp.input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=exp.flip_prob,
            hsv_prob=exp.hsv_prob,
        ),
        degrees=exp.degrees,
        translate=exp.translate,
        mosaic_scale=exp.mosaic_scale,
        mixup_scale=exp.mixup_scale,
        shear=exp.shear,
        enable_mixup=exp.enable_mixup,
        mosaic_prob=exp.mosaic_prob,
        mixup_prob=exp.mixup_prob,
    )
    sampler = InfiniteSampler(len(dataset), seed=exp.seed or 0)
    batch_sampler = YoloBatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
        mosaic=True,
    )
    dataloader_kwargs = {
        "num_workers": exp.data_num_workers,
        "pin_memory": False,
        "batch_sampler": batch_sampler,
    }
    if exp.data_num_workers > 0:
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    return DataLoader(dataset, **dataloader_kwargs)


def build_eval_dataset(
    exp: YOLOXTinyExp,
    split: str,
    verbose: bool = False,
) -> COCODataset:
    """
    Build a YOLOX COCO dataset for validation or testing.
    """
    split_spec = SPLITS[split]
    with optional_stdout_suppression(verbose):
        return COCODataset(
            data_dir=exp.data_dir,
            json_file=split_spec.annotation_file,
            name=split_spec.image_dir,
            img_size=exp.test_size,
            preproc=ValTransform(legacy=False),
        )


def build_eval_loader(
    exp: YOLOXTinyExp,
    split: str,
    batch_size: int,
    verbose: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build a sequential COCO evaluation loader.
    """
    dataset = build_eval_dataset(exp, split, verbose=verbose)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=exp.data_num_workers,
        pin_memory=False,
    )


def build_loss_loader(
    exp: YOLOXTinyExp,
    split: str,
    batch_size: int,
    verbose: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Build a label-preserving loader for validation loss computation.
    """
    split_spec = SPLITS[split]
    with optional_stdout_suppression(verbose):
        dataset = COCODataset(
            data_dir=exp.data_dir,
            json_file=split_spec.annotation_file,
            name=split_spec.image_dir,
            img_size=exp.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.0,
                hsv_prob=0.0,
            ),
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=exp.data_num_workers,
        pin_memory=False,
    )


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert `xyxy` boxes to `xywh` boxes.
    """
    converted = boxes.clone()
    converted[:, 2] = boxes[:, 2] - boxes[:, 0]
    converted[:, 3] = boxes[:, 3] - boxes[:, 1]
    return converted


def convert_outputs_to_coco(
    outputs: Sequence[torch.Tensor | None],
    info_imgs: list[torch.Tensor],
    img_ids: torch.Tensor,
    dataset: COCODataset,
    img_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """
    Convert YOLOX postprocessed outputs into COCO detection dictionaries.
    """
    detections = []
    flat_img_ids = img_ids.view(-1)

    for output, img_h, img_w, img_id in zip(
        outputs,
        info_imgs[0],
        info_imgs[1],
        flat_img_ids,
        strict=True,
    ):
        if output is None:
            continue

        output = output.cpu()
        bboxes = output[:, 0:4].clone()
        scale = min(
            img_size[0] / float(img_h),
            img_size[1] / float(img_w),
        )
        bboxes /= scale
        bboxes_xywh = xyxy_to_xywh(bboxes)
        cls = output[:, 6].to(torch.int64)
        scores = output[:, 4] * output[:, 5]

        for index in range(bboxes_xywh.shape[0]):
            category_id = dataset.class_ids[int(cls[index])]
            detections.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": bboxes_xywh[index].numpy().tolist(),
                    "score": float(scores[index]),
                    "segmentation": [],
                }
            )

    return detections


def bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute IoU between two COCO `xywh` boxes.
    """
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_precision_recall(
    coco_gt: Any,
    detections: list[dict[str, Any]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
) -> tuple[float, float]:
    """
    Compute image-level greedy precision and recall.
    """
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco_gt.dataset.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        gt_by_image[int(annotation["image_id"])].append(annotation)

    matched_gt: set[tuple[int, int]] = set()
    tp = 0
    fp = 0
    filtered_detections = sorted(
        (
            detection
            for detection in detections
            if detection["score"] >= conf_threshold
        ),
        key=lambda detection: detection["score"],
        reverse=True,
    )
    for detection in filtered_detections:
        image_id = int(detection["image_id"])
        category_id = int(detection["category_id"])
        best_iou = 0.0
        best_gt_id = None

        for annotation in gt_by_image.get(image_id, []):
            gt_key = (image_id, int(annotation["id"]))
            if gt_key in matched_gt:
                continue
            if int(annotation["category_id"]) != category_id:
                continue
            iou = bbox_iou_xywh(detection["bbox"], annotation["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_id = int(annotation["id"])

        if best_gt_id is not None and best_iou >= iou_threshold:
            matched_gt.add((image_id, best_gt_id))
            tp += 1
        else:
            fp += 1

    total_gt = sum(len(annotations) for annotations in gt_by_image.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return precision, recall


def run_coco_eval(
    coco_gt: Any,
    detections: list[dict[str, Any]],
    verbose: bool = False,
) -> tuple[np.ndarray, str]:
    """
    Run pycocotools COCO bbox evaluation and capture its text summary.
    """
    if not detections:
        return np.zeros(12, dtype=float), "No detections above threshold."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as tmp_file:
            json.dump(detections, tmp_file)
            tmp_path = tmp_file.name

        with optional_stdout_suppression(verbose):
            coco_dt = coco_gt.loadRes(tmp_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
            coco_eval.evaluate()
            coco_eval.accumulate()

        summary_buffer = io.StringIO()
        with contextlib.redirect_stdout(summary_buffer):
            coco_eval.summarize()
        return coco_eval.stats.copy(), summary_buffer.getvalue()
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    exp: YOLOXTinyExp,
    split: str,
    batch_size: int,
    device: torch.device,
    progress: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a YOLOX model on one split and return comparable metrics.
    """
    loader = build_eval_loader(exp, split, batch_size, verbose=verbose)
    dataset = cast(COCODataset, loader.dataset)
    model.eval()

    detections: list[dict[str, Any]] = []
    forward_time = 0.0
    nms_time = 0.0
    image_count = 0
    iterable = (
        tqdm(loader, desc=f"evaluate {split}", leave=False)
        if progress
        else loader
    )

    for imgs, _, info_imgs, img_ids in iterable:
        imgs = imgs.to(device=device, dtype=torch.float32)
        image_count += imgs.shape[0]

        sync_device(device)
        start_time = time.perf_counter()
        outputs = model(imgs)
        sync_device(device)
        forward_time += time.perf_counter() - start_time

        outputs = outputs.detach().cpu()
        start_time = time.perf_counter()
        outputs = postprocess(
            outputs,
            exp.num_classes,
            exp.test_conf,
            exp.nmsthre,
            class_agnostic=False,
        )
        nms_time += time.perf_counter() - start_time
        detections.extend(
            convert_outputs_to_coco(
                outputs,
                info_imgs,
                img_ids,
                dataset,
                exp.test_size,
            )
        )

    coco_stats, coco_summary = run_coco_eval(dataset.coco, detections, verbose)
    precision, recall = compute_precision_recall(dataset.coco, detections)
    speed_denominator = max(image_count, 1)

    return {
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "metrics/mAP50(B)": float(coco_stats[1]),
        "metrics/mAP50-95(B)": float(coco_stats[0]),
        "speed/forward_ms": 1000.0 * forward_time / speed_denominator,
        "speed/nms_ms": 1000.0 * nms_time / speed_denominator,
        "detections": detections,
        "summary": coco_summary,
        "dataset": dataset,
    }


def make_lr_scheduler(
    exp: YOLOXTinyExp,
    base_lr: float,
    iters_per_epoch: int,
) -> LRScheduler:
    """
    Build the YOLOX learning-rate scheduler for the effective run length.
    """
    return LRScheduler(
        exp.scheduler,
        base_lr,
        iters_per_epoch,
        exp.max_epoch,
        warmup_epochs=exp.warmup_epochs,
        warmup_lr_start=exp.warmup_lr,
        no_aug_epochs=exp.no_aug_epochs,
        min_lr_ratio=exp.min_lr_ratio,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_map: float,
    curr_map: float,
    *,
    ema_model: ModelEMA | None,
    scheduler_position: int,
    history: Sequence[Mapping[str, float]],
    mosaic_enabled: bool,
    l1_enabled: bool,
    validation_pending: bool,
    pending_train_metrics: Mapping[str, float] | None,
    iters_per_epoch: int,
    max_epoch: int,
    batch_size: int,
    image_size: int,
) -> None:
    """
    Atomically save all state needed to resume a YOLOX training run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": 1,
        "start_epoch": epoch,
        "completed_epoch": epoch,
        "scheduler_position": scheduler_position,
        "model": model.state_dict(),
        "ema_model": ema_model.ema.state_dict() if ema_model is not None else None,
        "ema_updates": ema_model.updates if ema_model is not None else 0,
        "optimizer": optimizer.state_dict(),
        "best_ap": best_map,
        "curr_ap": curr_map,
        "history": [dict(record) for record in history],
        "mosaic_enabled": mosaic_enabled,
        "l1_enabled": l1_enabled,
        "validation_pending": validation_pending,
        "pending_train_metrics": (
            dict(pending_train_metrics)
            if pending_train_metrics is not None
            else None
        ),
        "training_config": {
            "iters_per_epoch": iters_per_epoch,
            "max_epoch": max_epoch,
            "batch_size": batch_size,
            "image_size": image_size,
        },
    }
    atomic_torch_save(checkpoint, path)


def atomic_torch_save(checkpoint: Mapping[str, Any], path: Path) -> None:
    """
    Write a Torch checkpoint through a same-directory temporary file.
    """
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(dict(checkpoint), temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move restored optimizer tensors from CPU to the training device.
    """

    def move_value(value: Any) -> Any:
        """
        Recursively move tensors contained in optimizer state values.
        """
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        return value

    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = move_value(state)


def restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model: ModelEMA | None,
    lr_scheduler: LRScheduler,
    device: torch.device,
    *,
    iters_per_epoch: int,
    max_epoch: int,
    batch_size: int,
    image_size: int,
) -> ResumeState:
    """
    Restore and validate a recoverable YOLOX training checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError(
            f"Checkpoint is not in the recoverable format: {path}. "
            "Start a fresh run or provide a Phase 1 checkpoint."
        )

    expected_config = {
        "iters_per_epoch": iters_per_epoch,
        "max_epoch": max_epoch,
        "batch_size": batch_size,
        "image_size": image_size,
    }
    saved_config = checkpoint.get("training_config")
    if saved_config != expected_config:
        raise ValueError(
            "Resume settings do not match the checkpoint: "
            f"saved={saved_config}, requested={expected_config}."
        )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _move_optimizer_state_to_device(optimizer, device)

    saved_ema = checkpoint["ema_model"]
    if ema_model is None and saved_ema is not None:
        raise ValueError("Checkpoint uses EMA, but the current experiment does not.")
    if ema_model is not None and saved_ema is None:
        raise ValueError("Checkpoint has no EMA state for this EMA-enabled experiment.")
    if ema_model is not None:
        ema_model.ema.load_state_dict(saved_ema)
        ema_model.updates = int(checkpoint["ema_updates"])

    completed_epoch = int(checkpoint["completed_epoch"])
    scheduler_position = int(checkpoint["scheduler_position"])
    expected_position = completed_epoch * iters_per_epoch
    if scheduler_position != expected_position:
        raise ValueError(
            "Checkpoint scheduler position is inconsistent with its completed epoch: "
            f"{scheduler_position} != {expected_position}."
        )

    restored_lr = lr_scheduler.update_lr(scheduler_position)
    for param_group in optimizer.param_groups:
        param_group["lr"] = restored_lr

    history = [dict(record) for record in checkpoint["history"]]
    validation_pending = bool(checkpoint["validation_pending"])
    expected_history_length = completed_epoch - int(validation_pending)
    if len(history) != expected_history_length:
        raise ValueError(
            "Checkpoint history is inconsistent with its completed epoch: "
            f"{len(history)} rows for epoch {completed_epoch}."
        )

    pending_train_metrics = checkpoint["pending_train_metrics"]
    if validation_pending and pending_train_metrics is None:
        raise ValueError("Pending validation checkpoint has no training metrics.")
    if not validation_pending and pending_train_metrics is not None:
        raise ValueError(
            "Completed checkpoint unexpectedly has pending training metrics."
        )

    return ResumeState(
        completed_epoch=completed_epoch,
        scheduler_position=scheduler_position,
        best_map=float(checkpoint["best_ap"]),
        curr_map=float(checkpoint["curr_ap"]),
        history=history,
        mosaic_enabled=bool(checkpoint["mosaic_enabled"]),
        l1_enabled=bool(checkpoint["l1_enabled"]),
        validation_pending=validation_pending,
        pending_train_metrics=(
            dict(pending_train_metrics)
            if pending_train_metrics is not None
            else None
        ),
    )


def write_run_metadata(
    path: Path,
    exp: YOLOXTinyExp,
    dataset_dir: Path,
    pretrained_path: Path,
    class_names: tuple[str, ...],
    device: torch.device,
    settings: TrainingSettings,
    project_root: Path | None = None,
) -> None:
    """
    Write a compact YAML record of the experiment configuration.
    """
    if project_root is None:
        project_root = dataset_dir.parents[2]

    metadata = {
        "project_name": exp.exp_name,
        "dataset": str(dataset_dir.relative_to(project_root)),
        "pretrained_checkpoint": str(pretrained_path.relative_to(project_root)),
        "classes": list(class_names),
        "epochs": exp.max_epoch,
        "batch_size": settings.batch_size,
        "train_batch_limit": settings.train_batch_limit,
        "image_size": exp.input_size,
        "device": str(device),
        "seed": exp.seed,
        "test_conf": exp.test_conf,
        "nms_threshold": exp.nmsthre,
        "smoke_run": settings.smoke_run,
    }
    path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> torch.nn.Module:
    """
    Load COCO-pretrained YOLOX weights, skipping incompatible head tensors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    model.load_state_dict(compatible_state, strict=False)
    if skipped_keys:
        print(
            "Skipped incompatible pretrained tensors for the dataset-specific head: "
            f"{', '.join(skipped_keys)}"
        )
    return model


@torch.no_grad()
def evaluate_losses(
    model: torch.nn.Module,
    exp: YOLOXTinyExp,
    split: str,
    batch_size: int,
    device: torch.device,
    progress: bool = False,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Compute average YOLOX loss components on one labeled split.
    """
    loader = build_loss_loader(exp, split, batch_size, verbose=verbose)
    loss_columns = {
        "total_loss": "total_loss",
        "iou_loss": "iou_loss",
        "obj_loss": "conf_loss",
        "cls_loss": "cls_loss",
        "l1_loss": "l1_loss",
    }
    loss_totals = {f"{split}/{name}": 0.0 for name in loss_columns}
    module_modes = {module: module.training for module in model.modules()}
    model_head = cast(YOLOXHead, model.head)
    # YOLOX returns losses only through its training branches; keep child
    # modules in eval mode so validation loss does not update BatchNorm stats.
    model.eval()
    model.training = True
    model_head.training = True

    iterable = (
        tqdm(loader, desc=f"{split} loss", leave=False)
        if progress
        else loader
    )
    batch_count = 0
    try:
        for imgs, targets, _, _ in iterable:
            imgs = imgs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs = model(imgs, targets)
            batch_count += 1

            for column_name, output_name in loss_columns.items():
                loss_totals[f"{split}/{column_name}"] += tensor_to_float(
                    outputs[output_name]
                )
    finally:
        for module, training in module_modes.items():
            module.training = training

    denominator = max(batch_count, 1)
    return {key: value / denominator for key, value in loss_totals.items()}


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    lr_scheduler: LRScheduler,
    epoch_index: int,
    iters_per_epoch: int,
    device: torch.device,
    ema_model: ModelEMA | None,
    progress: bool,
) -> dict[str, float]:
    """
    Train for one epoch and return average YOLOX losses.
    """
    model.train()
    loader_iter = iter(train_loader)
    loss_totals = {
        "train/total_loss": 0.0,
        "train/iou_loss": 0.0,
        "train/obj_loss": 0.0,
        "train/cls_loss": 0.0,
        "train/l1_loss": 0.0,
    }
    latest_lr = optimizer.param_groups[0]["lr"]
    iter_range = range(iters_per_epoch)
    progress_bar = None
    iterable = iter_range
    if progress:
        progress_bar = tqdm(
            iter_range,
            desc=f"train epoch {epoch_index + 1}",
            leave=False,
        )
        iterable = progress_bar

    for iter_index in iterable:
        imgs, targets, _, _ = next(loader_iter)
        imgs = imgs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        targets.requires_grad = False

        outputs = model(imgs, targets)
        total_loss = outputs["total_loss"]

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        global_iter = epoch_index * iters_per_epoch + iter_index + 1
        latest_lr = lr_scheduler.update_lr(global_iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = latest_lr

        loss_values = {
            "train/total_loss": outputs["total_loss"],
            "train/iou_loss": outputs["iou_loss"],
            "train/obj_loss": outputs["conf_loss"],
            "train/cls_loss": outputs["cls_loss"],
            "train/l1_loss": outputs["l1_loss"],
        }
        for key, value in loss_values.items():
            loss_totals[key] += tensor_to_float(value)
        if progress_bar is not None:
            progress_bar.set_postfix(
                loss=f"{tensor_to_float(total_loss):.3f}",
                lr=f"{latest_lr:.2e}",
            )

    averaged_losses = {
        key: value / iters_per_epoch for key, value in loss_totals.items()
    }
    averaged_losses["lr"] = latest_lr
    return averaged_losses


def apply_training_phase(
    train_loader: DataLoader,
    model: torch.nn.Module,
    ema_model: ModelEMA | None,
    *,
    mosaic_enabled: bool,
    l1_enabled: bool,
) -> None:
    """
    Apply the saved mosaic and L1 phase to the loader and model copies.
    """
    if not mosaic_enabled:
        train_loader.close_mosaic()

    cast(YOLOXHead, model.head).use_l1 = l1_enabled
    if ema_model is not None:
        cast(YOLOXHead, ema_model.ema.head).use_l1 = l1_enabled


def evaluate_training_epoch(
    eval_model: torch.nn.Module,
    exp: YOLOXTinyExp,
    settings: TrainingSettings,
    device: torch.device,
    epoch: int,
    train_metrics: Mapping[str, float],
) -> dict[str, float]:
    """
    Validate one trained epoch and combine its training and validation metrics.
    """
    val_losses = evaluate_losses(
        eval_model,
        exp,
        "val",
        settings.batch_size,
        device,
        progress=settings.show_progress,
        verbose=settings.verbose_output,
    )
    val_metrics = evaluate_model(
        eval_model,
        exp,
        "val",
        settings.batch_size,
        device,
        verbose=settings.verbose_output,
    )
    return {
        "epoch": epoch,
        **train_metrics,
        **val_losses,
        "metrics/precision(B)": val_metrics["metrics/precision(B)"],
        "metrics/recall(B)": val_metrics["metrics/recall(B)"],
        "metrics/mAP50(B)": val_metrics["metrics/mAP50(B)"],
        "metrics/mAP50-95(B)": val_metrics["metrics/mAP50-95(B)"],
        "speed/forward_ms": val_metrics["speed/forward_ms"],
        "speed/nms_ms": val_metrics["speed/nms_ms"],
    }


def write_training_history(
    path: Path,
    history: Sequence[Mapping[str, float]],
) -> None:
    """
    Atomically replace the CSV representation of the checkpoint history.
    """
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            encoding="utf-8",
            newline="",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            pd.DataFrame(history).to_csv(temporary_file, index=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fit_yolox(
    exp: YOLOXTinyExp,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    settings: TrainingSettings,
    device: torch.device,
    project_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_CLASSES,
    *,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Fine-tune a YOLOX experiment and save history/checkpoint artifacts.
    """
    set_reproducibility(exp.seed or settings.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = ensure_dir(output_dir / "weights")
    history_path = output_dir / "results.csv"
    if not resume:
        write_run_metadata(
            output_dir / "args.yaml",
            exp,
            dataset_dir,
            checkpoint_path,
            class_names,
            device,
            settings,
            project_root,
        )

    for attr in ("model", "optimizer"):
        discard_cached_exp_attribute(exp, attr)

    model = exp.get_model()
    if not resume:
        model = load_pretrained_weights(model, checkpoint_path)
    model.to(device)

    optimizer = exp.get_optimizer(settings.batch_size)
    train_loader = build_train_loader(
        exp,
        settings.batch_size,
        verbose=settings.verbose_output,
    )
    full_iters_per_epoch = len(train_loader)
    iters_per_epoch = (
        min(settings.train_batch_limit, full_iters_per_epoch)
        if settings.train_batch_limit is not None
        else full_iters_per_epoch
    )
    lr_scheduler = make_lr_scheduler(
        exp,
        exp.basic_lr_per_img * settings.batch_size,
        iters_per_epoch,
    )
    ema_model = ModelEMA(model, 0.9998) if exp.ema else None

    history: list[dict[str, float]] = []
    best_map = -1.0
    curr_map = -1.0
    best_path = weights_dir / "best_ckpt.pth"
    last_path = weights_dir / "last_ckpt.pth"
    mosaic_enabled = True
    l1_enabled = False
    start_epoch = 0

    def save_training_state(
        path: Path,
        completed_epoch: int,
        *,
        validation_pending: bool,
        pending_train_metrics: Mapping[str, float] | None,
    ) -> None:
        """
        Save the current trainer state with the run's fixed configuration.
        """
        save_checkpoint(
            path,
            model,
            optimizer,
            completed_epoch,
            best_map,
            curr_map,
            ema_model=ema_model,
            scheduler_position=completed_epoch * iters_per_epoch,
            history=history,
            mosaic_enabled=mosaic_enabled,
            l1_enabled=l1_enabled,
            validation_pending=validation_pending,
            pending_train_metrics=pending_train_metrics,
            iters_per_epoch=iters_per_epoch,
            max_epoch=exp.max_epoch,
            batch_size=settings.batch_size,
            image_size=settings.image_size,
        )

    def finish_validation(
        completed_epoch: int,
        train_metrics: Mapping[str, float],
    ) -> None:
        """
        Validate a trained epoch and commit its best, last, and history artifacts.
        """
        nonlocal best_map, curr_map
        eval_model = ema_model.ema if ema_model is not None else model
        validation_start = time.perf_counter()
        record = evaluate_training_epoch(
            eval_model,
            exp,
            settings,
            device,
            completed_epoch,
            train_metrics,
        )
        validation_seconds = time.perf_counter() - validation_start
        epoch_seconds = train_metrics["time/train_s"] + validation_seconds
        cumulative_seconds = sum(
            previous.get("time/epoch_s", 0.0) for previous in history
        )
        record.update(
            {
                "time/val_s": validation_seconds,
                "time/epoch_s": epoch_seconds,
                "time/cumulative_s": cumulative_seconds + epoch_seconds,
            }
        )
        curr_map = record["metrics/mAP50-95(B)"]
        is_best = curr_map > best_map
        best_map = max(best_map, curr_map)
        history.append(record)

        if is_best:
            save_training_state(
                best_path,
                completed_epoch,
                validation_pending=False,
                pending_train_metrics=None,
            )
        save_training_state(
            last_path,
            completed_epoch,
            validation_pending=False,
            pending_train_metrics=None,
        )
        write_training_history(history_path, history)
        if settings.verbose_output:
            print_epoch_summary(record)

    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")
        restored = restore_checkpoint(
            last_path,
            model,
            optimizer,
            ema_model,
            lr_scheduler,
            device,
            iters_per_epoch=iters_per_epoch,
            max_epoch=exp.max_epoch,
            batch_size=settings.batch_size,
            image_size=settings.image_size,
        )
        history = restored.history
        best_map = restored.best_map
        curr_map = restored.curr_map
        mosaic_enabled = restored.mosaic_enabled
        l1_enabled = restored.l1_enabled
        start_epoch = restored.completed_epoch
        apply_training_phase(
            train_loader,
            model,
            ema_model,
            mosaic_enabled=mosaic_enabled,
            l1_enabled=l1_enabled,
        )
        write_training_history(history_path, history)

        if restored.validation_pending:
            if settings.verbose_output:
                print(f"Completing validation for epoch {start_epoch}.")
            finish_validation(
                start_epoch,
                cast(dict[str, float], restored.pending_train_metrics),
            )

    for epoch_index in range(start_epoch, exp.max_epoch):
        if (
            exp.no_aug_epochs > 0
            and epoch_index >= exp.max_epoch - exp.no_aug_epochs
            and mosaic_enabled
        ):
            mosaic_enabled = False
            l1_enabled = True
            apply_training_phase(
                train_loader,
                model,
                ema_model,
                mosaic_enabled=mosaic_enabled,
                l1_enabled=l1_enabled,
            )
            if settings.verbose_output:
                print(f"Closed mosaic augmentation at epoch {epoch_index + 1}.")

        training_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            lr_scheduler,
            epoch_index,
            iters_per_epoch,
            device,
            ema_model,
            settings.show_progress,
        )
        train_metrics["time/train_s"] = time.perf_counter() - training_start
        completed_epoch = epoch_index + 1
        save_training_state(
            last_path,
            completed_epoch,
            validation_pending=True,
            pending_train_metrics=train_metrics,
        )
        finish_validation(completed_epoch, train_metrics)

    return pd.DataFrame(history)


def fit_yolox_tiny(
    exp: YOLOXTinyExp,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    settings: TrainingSettings,
    device: torch.device,
    project_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_CLASSES,
    *,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Fine-tune YOLOX-Tiny and save history/checkpoint artifacts.
    """
    return fit_yolox(
        exp,
        checkpoint_path,
        output_dir,
        dataset_dir,
        settings,
        device,
        project_root,
        class_names,
        resume=resume,
    )


def fit_yolox_nano(
    exp: YOLOXNanoExp,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_dir: Path,
    settings: TrainingSettings,
    device: torch.device,
    project_root: Path,
    class_names: tuple[str, ...] = BASKETBALL_CLASSES,
    *,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Fine-tune YOLOX-Nano and save history/checkpoint artifacts.
    """
    return fit_yolox(
        exp,
        checkpoint_path,
        output_dir,
        dataset_dir,
        settings,
        device,
        project_root,
        class_names,
        resume=resume,
    )


def print_epoch_summary(record: dict[str, float]) -> None:
    """
    Print a compact epoch summary.
    """
    print(
        {
            "epoch": record["epoch"],
            "total_loss": record["train/total_loss"],
            "val_total_loss": record["val/total_loss"],
            "precision": record["metrics/precision(B)"],
            "recall": record["metrics/recall(B)"],
            "mAP50": record["metrics/mAP50(B)"],
            "mAP50-95": record["metrics/mAP50-95(B)"],
            "epoch_s": record["time/epoch_s"],
        }
    )


def load_trained_model(
    exp: YOLOXTinyExp,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a trained YOLOX checkpoint into a fresh model instance.
    """
    discard_cached_exp_attribute(exp, "model")
    model = exp.get_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("ema_model") or checkpoint["model"]
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def export_trained_model_to_onnx(
    exp: YOLOXTinyExp,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """
    Export a selected YOLOX checkpoint's EMA model to a validated ONNX file.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output path must end with .onnx: {output_path}")

    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_model = load_trained_model(
            exp,
            checkpoint_path,
            torch.device("cpu"),
        )
        export_model = replace_module(
            export_model,
            torch.nn.SiLU,
            YOLOXSiLU,
        )
        export_model.head.decode_in_inference = False
        dummy_input = torch.randn(1, 3, *exp.test_size)

        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            input_names=["images"],
            output_names=["output"],
            opset_version=11,
            dynamo=False,
        )
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    finally:
        discard_cached_exp_attribute(exp, "model")

    return output_path


def read_training_history(results_csv: Path) -> pd.DataFrame:
    """
    Read this notebook's YOLOX training history CSV.
    """
    history = pd.read_csv(results_csv)
    history.columns = history.columns.str.strip()
    return history


def plot_train_losses(history: pd.DataFrame):
    """
    Plot YOLOX train and validation loss components.
    """
    import matplotlib.pyplot as plt

    loss_columns = [
        ("total", "train/total_loss", "val/total_loss"),
        ("iou", "train/iou_loss", "val/iou_loss"),
        ("objectness", "train/obj_loss", "val/obj_loss"),
        ("classification", "train/cls_loss", "val/cls_loss"),
        ("l1", "train/l1_loss", "val/l1_loss"),
    ]
    fig, axes = plt.subplots(1, len(loss_columns), figsize=(18, 4), sharex=True)

    for axis, (loss_name, train_column, val_column) in zip(
        axes,
        loss_columns,
        strict=True,
    ):
        axis.plot(history["epoch"], history[train_column], label="train")
        if val_column in history.columns:
            axis.plot(history["epoch"], history[val_column], label="validation")
            axis.legend()
        axis.set_title(loss_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_detection_metrics(history: pd.DataFrame):
    """
    Plot validation detection metrics.
    """
    import matplotlib.pyplot as plt

    metric_columns = [
        ("precision", "metrics/precision(B)"),
        ("recall", "metrics/recall(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for axis, (metric_name, column) in zip(
        axes.ravel(),
        metric_columns,
        strict=True,
    ):
        axis.plot(history["epoch"], history[column])
        axis.set_title(metric_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("metric")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

    fig.tight_layout()
    return fig, axes


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """
    Draw a filled text label on an image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x, y = origin
    y = max(y, text_h + baseline + 2)
    cv2.rectangle(
        image,
        (x, y - text_h - baseline - 4),
        (x + text_w + 4, y),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Mapping[str, Any]],
    category_names: Mapping[int, str],
    color: tuple[int, int, int],
    include_scores: bool = False,
) -> np.ndarray:
    """
    Draw COCO `xywh` boxes on an image.
    """
    result = image.copy()
    for box_record in boxes:
        x, y, w, h = [int(round(value)) for value in box_record["bbox"]]
        category_id = int(box_record["category_id"])
        label = category_names.get(category_id, str(category_id))
        if include_scores:
            label = f"{label} {box_record['score']:.2f}"

        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness=2)
        draw_label(result, label, (x, y), color)
    return result


def letterbox_for_grid(
    image: np.ndarray,
    tile_size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    """
    Resize an image into a fixed tile without changing its aspect ratio.
    """
    tile_w, tile_h = tile_size
    height, width = image.shape[:2]
    scale = min(tile_w / width, tile_h / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_h, tile_w, 3), 114, dtype=np.uint8)
    x_offset = (tile_w - resized_w) // 2
    y_offset = (tile_h - resized_h) // 2
    tile[y_offset : y_offset + resized_h, x_offset : x_offset + resized_w] = resized
    return tile


def make_image_grid(
    images: list[np.ndarray],
    columns: int = 4,
    tile_size: tuple[int, int] = (320, 320),
) -> np.ndarray:
    """
    Combine images into a fixed-size grid.
    """
    rows = max(1, math.ceil(len(images) / columns))
    tile_w, tile_h = tile_size
    grid = np.full((rows * tile_h, columns * tile_w, 3), 114, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        tile = letterbox_for_grid(image, tile_size)
        y0 = row * tile_h
        x0 = column * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile

    return grid


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    exp: YOLOXTinyExp,
    image: np.ndarray,
    device: torch.device,
    class_ids: Sequence[int],
    conf_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Run YOLOX inference on one raw BGR image.
    """
    preproc = ValTransform(legacy=False)
    processed, _ = preproc(image, None, exp.test_size)
    tensor = torch.from_numpy(processed).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    height, width = image.shape[:2]
    scale = min(exp.test_size[0] / height, exp.test_size[1] / width)

    model.eval()
    outputs = model(tensor)
    outputs = postprocess(
        outputs.detach().cpu(),
        exp.num_classes,
        conf_threshold,
        exp.nmsthre,
        class_agnostic=False,
    )
    output = outputs[0]
    if output is None:
        return []

    bboxes = output[:, 0:4].clone() / scale
    bboxes_xywh = xyxy_to_xywh(bboxes)
    cls = output[:, 6].to(torch.int64)
    scores = output[:, 4] * output[:, 5]

    predictions = []
    for index in range(bboxes_xywh.shape[0]):
        category_id = class_ids[int(cls[index])]
        predictions.append(
            {
                "category_id": int(category_id),
                "bbox": bboxes_xywh[index].numpy().tolist(),
                "score": float(scores[index]),
            }
        )
    return predictions


def save_sample_visualizations(
    model: torch.nn.Module,
    exp: YOLOXTinyExp,
    split: str,
    output_dir: Path,
    device: torch.device,
    max_images: int = 8,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """
    Save ground-truth and prediction grids for one dataset split.
    """
    dataset = build_eval_dataset(exp, split, verbose=verbose)
    category_names = {
        category_id: category["name"]
        for category_id, category in dataset.coco.cats.items()
    }
    label_images = []
    pred_images = []

    for image_id in dataset.ids[:max_images]:
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = Path(dataset.data_dir) / dataset.name / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        annotation_ids = dataset.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        annotations = dataset.coco.loadAnns(annotation_ids)
        predictions = predict_image(model, exp, image, device, dataset.class_ids)

        label_images.append(
            draw_boxes(
                image,
                annotations,
                category_names,
                color=(40, 180, 80),
                include_scores=False,
            )
        )
        pred_images.append(
            draw_boxes(
                image,
                predictions,
                category_names,
                color=(50, 120, 230),
                include_scores=True,
            )
        )

    labels_path = output_dir / f"{split}_batch0_labels.jpg"
    preds_path = output_dir / f"{split}_batch0_pred.jpg"
    cv2.imwrite(str(labels_path), make_image_grid(label_images))
    cv2.imwrite(str(preds_path), make_image_grid(pred_images))
    return labels_path, preds_path


def format_final_interpretation(
    history: pd.DataFrame,
    test_metrics: dict[str, Any],
) -> str:
    """
    Format a Markdown interpretation from the completed run.
    """
    best_row = history.loc[history["metrics/mAP50-95(B)"].idxmax()]
    final_row = history.iloc[-1]

    return f"""
### YOLOX-Tiny training summary

The best validation checkpoint was selected at epoch
**{int(best_row["epoch"])}**, with validation mAP50-95 of
**{best_row["metrics/mAP50-95(B)"]:.3f}** and mAP50 of
**{best_row["metrics/mAP50(B)"]:.3f}**.

At the final recorded epoch, train total loss was
**{final_row["train/total_loss"]:.3f}**. The held-out test split produced
precision **{test_metrics["metrics/precision(B)"]:.3f}**, recall
**{test_metrics["metrics/recall(B)"]:.3f}**, mAP50
**{test_metrics["metrics/mAP50(B)"]:.3f}**, and mAP50-95
**{test_metrics["metrics/mAP50-95(B)"]:.3f}**.
"""

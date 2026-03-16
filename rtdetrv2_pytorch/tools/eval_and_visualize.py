#!/usr/bin/env python3
"""Evaluate an RT-DETRv2 checkpoint on a COCO dataset and optionally save visualizations.

Examples:
  python rtdetrv2_pytorch/tools/eval_and_visualize.py \
    -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml \
    -r /path/to/best.pth \
    --images-dir rtdetrv2_pytorch/dataset/bbox_data/images \
    --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json \
    --device cuda \
    --save-vis-dir /tmp/heron_eval_vis \
    --draw-gt
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFile
from pycocotools.cocoeval import COCOeval

# Match tools/train.py behavior so `src.*` imports work from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, create

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def _override_eval_dataset(cfg: YAMLConfig, images_dir: str | None, coco_json: str | None, num_workers: int) -> None:
    cfg.yaml_cfg.setdefault("val_dataloader", {})
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = int(num_workers)
    cfg.yaml_cfg["val_dataloader"]["batch_size"] = 1
    cfg.yaml_cfg["val_dataloader"].pop("total_batch_size", None)
    cfg.yaml_cfg["val_dataloader"]["shuffle"] = False
    dataset = cfg.yaml_cfg["val_dataloader"].setdefault("dataset", {})
    if images_dir is not None:
        dataset["img_folder"] = images_dir
    if coco_json is not None:
        dataset["ann_file"] = coco_json


def _category_color(category_id: int) -> tuple[int, int, int]:
    hue = (int(category_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _draw_boxes(
    image: Image.Image,
    *,
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    id_to_name: dict[int, str],
    score_threshold: float,
    draw_gt: bool,
    gt_box_width: int,
    pred_box_width: int,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    if draw_gt:
        for ann in gt_boxes:
            x, y, w, h = ann["bbox"]
            xyxy = [x, y, x + w, y + h]
            cid = int(ann["category_id"])
            color = _category_color(cid)
            draw.rectangle(xyxy, outline=color, width=gt_box_width)
            draw.text((x, max(0, y - 14)), f"GT {id_to_name.get(cid, cid)}", fill=color)

    for det in pred_boxes:
        score = float(det["score"])
        if score < score_threshold:
            continue
        x, y, w, h = det["bbox"]
        xyxy = [x, y, x + w, y + h]
        cid = int(det["category_id"])
        color = _category_color(cid)
        draw.rectangle(xyxy, outline=color, width=pred_box_width)
        label = f"{id_to_name.get(cid, cid)} {score:.2f}"
        draw.text((x, max(0, y - 14)), label, fill=color)

    return out


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    cfg = YAMLConfig(args.config, resume=args.resume)
    _override_eval_dataset(cfg, args.images_dir, args.coco_json, args.num_workers)

    device = torch.device(args.device)
    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)
    _load_checkpoint(model, args.resume)
    model.eval()
    postprocessor.eval()

    val_loader = cfg.val_dataloader
    dataset = val_loader.dataset

    # COCO GT API from the dataset itself.
    coco_gt = dataset.coco
    categories = coco_gt.dataset.get("categories", [])
    id_to_name = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in categories}

    detections: list[dict[str, Any]] = []
    vis_dir = Path(args.save_vis_dir).expanduser().resolve() if args.save_vis_dir else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    saved_vis = 0
    processed = 0

    for samples, targets in val_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            labels = result["labels"].detach().cpu().tolist()
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()

            image_dets: list[dict[str, Any]] = []
            for cid, box, score in zip(labels, boxes, scores):
                if float(score) < float(args.eval_score_threshold):
                    continue
                det = {
                    "image_id": image_id,
                    "category_id": int(cid),
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                }
                detections.append(det)
                image_dets.append(det)

            if vis_dir is not None and saved_vis < args.max_vis_images:
                info = dataset.coco.loadImgs([image_id])[0]
                image_path = Path(dataset.img_folder) / info["file_name"]
                gt_anns = copy.deepcopy(dataset.coco.imgToAnns.get(image_id, []))
                image = Image.open(image_path).convert("RGB")
                vis = _draw_boxes(
                    image,
                    gt_boxes=gt_anns,
                    pred_boxes=image_dets,
                    id_to_name=id_to_name,
                    score_threshold=float(args.vis_score_threshold),
                    draw_gt=args.draw_gt,
                    gt_box_width=int(args.gt_box_width),
                    pred_box_width=int(args.pred_box_width),
                )
                vis.save(vis_dir / info["file_name"])
                saved_vis += 1

            processed += 1
            if processed % max(1, int(args.print_freq)) == 0:
                print(f"Processed {processed}/{len(dataset)} images")

    results_json = Path(args.results_json).expanduser().resolve() if args.results_json else None
    if results_json is not None:
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        print(f"Saved detections JSON to {results_json}")

    if not detections:
        raise SystemExit("No detections were produced. Try lowering --eval-score-threshold.")

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "AP@[0.50:0.95]": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR@1": float(evaluator.stats[6]),
        "AR@10": float(evaluator.stats[7]),
        "AR@100": float(evaluator.stats[8]),
        "AR_small": float(evaluator.stats[9]),
        "AR_medium": float(evaluator.stats[10]),
        "AR_large": float(evaluator.stats[11]),
        "num_images": int(len(dataset)),
        "num_detections": int(len(detections)),
    }

    if args.metrics_json:
        metrics_json = Path(args.metrics_json).expanduser().resolve()
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to {metrics_json}")

    print("\nMetrics summary:")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="RT-DETRv2 YAML config.")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--images-dir", type=str, default=None, help="Override val image folder.")
    parser.add_argument("--coco-json", type=str, default=None, help="Override val COCO annotation JSON.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="Evaluation dataloader workers.")
    parser.add_argument("--eval-score-threshold", type=float, default=0.0, help="Threshold for COCO eval detections.")
    parser.add_argument("--save-vis-dir", type=str, default=None, help="Directory to save prediction visualizations.")
    parser.add_argument("--vis-score-threshold", type=float, default=0.4, help="Threshold for drawn predictions.")
    parser.add_argument("--pred-box-width", type=int, default=5, help="Line width for predicted boxes.")
    parser.add_argument("--gt-box-width", type=int, default=4, help="Line width for GT boxes when --draw-gt is used.")
    parser.add_argument("--max-vis-images", type=int, default=50, help="Max images to visualize.")
    parser.add_argument("--draw-gt", action="store_true", help="Also draw GT boxes on the visualization.")
    parser.add_argument("--results-json", type=str, default=None, help="Optional path to save raw detection JSON.")
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save metrics summary.")
    parser.add_argument("--print-freq", type=int, default=10, help="Progress print frequency in images.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

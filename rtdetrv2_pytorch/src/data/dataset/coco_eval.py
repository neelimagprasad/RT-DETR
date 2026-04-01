"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

import copy

from ...core import register
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

@register()
class CocoEvaluator(FasterCocoEvaluator):
    def __init__(self, coco_gt, iou_types, lvis_style=False, ranges=None, class_agnostic=False):
        self.class_agnostic = class_agnostic

        if ranges is None:
            ranges = {
                "small": [0**2, 32**2],
                "medium": [32**2, 96**2],
                "large": [96**2, 1e5**2],
            }

        if class_agnostic:
            coco_gt = self._make_class_agnostic_coco_gt(coco_gt)

        super().__init__(coco_gt=coco_gt, iou_types=iou_types, lvis_style=lvis_style, ranges=ranges)

    @staticmethod
    def _make_class_agnostic_coco_gt(coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        if not hasattr(coco_gt, "dataset"):
            return coco_gt

        dataset = copy.deepcopy(coco_gt.dataset)
        dataset["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]

        for ann in dataset.get("annotations", []):
            ann["category_id"] = 1

        coco_gt.dataset = dataset
        coco_gt.createIndex()
        return coco_gt

    def update(self, predictions):
        if self.class_agnostic:
            predictions = {
                image_id: {
                    **prediction,
                    "labels": prediction["labels"].new_ones(prediction["labels"].shape, dtype=prediction["labels"].dtype),
                }
                for image_id, prediction in predictions.items()
            }

        return super().update(predictions)

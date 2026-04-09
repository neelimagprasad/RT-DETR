# Fine-Tuning Docling

Fine-tuning Heron 101 on construction documents to improve object detection bounding box placement.

## Overview

This repository adapts RT-DETR to fine-tune the Heron 101 model on construction document pages. The main goal is to improve object detection localization so bounding boxes more accurately capture the target regions in architecutre and construction documents. The goal is to capture both drawings and legends, notes and schedules with one unified model. 

## Key Results

High-level summary of the best outcomes from this repo.

Suggested things to include:
- best validation metric
- what metric was used
- whether evaluation is class-aware or class-agnostic
- which checkpoint/config produced the best result

## Repository Structure

Most of the project-specific work lives under `rtdetrv2_pytorch/`.

- `rtdetrv2_pytorch/configs/rtdetrv2/`: Heron 101 training configs
- `rtdetrv2_pytorch/dataset/bbox_data/`: construction document dataset, images, and COCO annotations
- `rtdetrv2_pytorch/dataset/`: dataset preparation, labeling, rendering, and visualization scripts
- `rtdetrv2_pytorch/tools/`: training, evaluation, checkpoint conversion, ONNX export, and analysis utilities
- `rtdetrv2_pytorch/tools/heron-training-script.ipynb`: Modal notebook used to launch Heron 101 fine-tuning
- `output/` or external Modal output storage: checkpoints, logs, and exported artifacts

## Dataset

This project uses a custom COCO-style construction document dataset built from rendered PDF pages. The current annotation files contain `358` train images with `3,206` annotations and `30` test images with `199` annotations across `17` categories.

The images live under `rtdetrv2_pytorch/dataset/bbox_data/images_train` and `rtdetrv2_pytorch/dataset/bbox_data/images_test`, and the COCO JSON files live under `rtdetrv2_pytorch/dataset/bbox_data/annotations`.

For details on how to inspect, edit, or visualize the dataset and its annotations, see the [dataset README](rtdetrv2_pytorch/dataset/bbox_data/README.md).

## Modal Setup

The training notebook for Modal is `rtdetrv2_pytorch/tools/heron-training-script.ipynb`.

This notebook expects two Modal volumes:
- `rtdetr-bbox-data`: dataset volume
- `rtdetr-outputs`: output volume for checkpoints and logs

Create them with:

```bash
modal volume create rtdetr-bbox-data
modal volume create rtdetr-outputs
```

The notebook reads the dataset from the data volume and writes checkpoints, logs, converted weights, and exports to the output volume.

## Training Setup

The main training config is `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml`. This is the base Heron 101 fine-tuning config for the construction document dataset and points to the train and test COCO JSON files under `rtdetrv2_pytorch/dataset/bbox_data/`.

The base config uses:
- `17` classes
- class-agnostic bbox evaluation
- main learning rate: `0.00015`
- backbone learning rate: `0.000015`
- LR milestones: `[40, 80, 120]`
- LR gamma: `0.2`
- warmup: `100`

Training was launched from the Modal notebook at `rtdetrv2_pytorch/tools/heron-training-script.ipynb`. That notebook:
- clones the training branch into the Modal environment
- mounts the dataset volume at `/data` and the output volume at `/outputs`
- copies `/data/bbox_data` into `rtdetrv2_pytorch/dataset/bbox_data`
- installs repo requirements inside the training image
- converts the Hugging Face Heron 101 checkpoint into a repo-compatible `.pth` file if it is not already cached
- launches multi-GPU training with `torchrun`

The main multi-GPU training recipe used:
- `4` GPUs
- batch size `4` per GPU, for total train and validation batch size `16`
- seed `3407`
- `200` epochs
- `aux_loss=True`
- `num_denoising=100`
- `use_amp=False`
- `find_unused_parameters=True`

The best run in this repo is `run8`, with checkpoint `best.pth`. This run used the 4-GPU setup above with auxiliary loss and denoising enabled, and training behavior stabilized after roughly `30` epochs.

## Backbone And Weight Initialization

This project uses the RT-DETRv2 R101 architecture as the training backbone and initializes it from Docling Heron 101 weights hosted on Hugging Face. The Hugging Face source model is `docling-project/docling-layout-heron-101`, and the conversion logic for this repo lives in `rtdetrv2_pytorch/tools/convert_hf_heron101_to_rtdetrv2.py`.

That conversion script builds this repo's RT-DETRv2-R101 model and maps the Hugging Face checkpoint weights into the repo format, including:
- the ResNet backbone stem and residual stages
- encoder and decoder heads
- decoder layers and attention weights
- denoising class embeddings

The converted output is a repo-compatible `.pth` checkpoint that can be passed into training with `-t`. In the Modal notebook workflow, this converted checkpoint is cached in the output volume as `heron101_converted_with_backbone.pth` so it does not need to be rebuilt every run.





## Evaluation

The main manual evaluation script is `rtdetrv2_pytorch/tools/eval_and_visualize.py`. This is the script used to evaluate a saved checkpoint on the validation or test set, optionally apply NMS, and save visualizations plus JSON outputs.

This same evaluation workflow also appears in the Modal notebook at `rtdetrv2_pytorch/tools/heron-training-script.ipynb`, where it is used to run the `eval_vis_test_ca_nms` style evaluation. That is the evaluation variant used most often in this repo because it does three useful things at once:
- computes COCO metrics
- saves visualization images for manual inspection
- applies NMS with `--nms-iou-threshold 0.5`, which makes the output easier to review visually

The notebook evaluation command uses:
- config: `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml`
- checkpoint: `/mnt/rtdetr-outputs/heron101_bbox_data_run5/best.pth` in the example notebook cell
- images: `/mnt/rtdetr-bbox-data/bbox_data/images_test`
- annotations: `/mnt/rtdetr-bbox-data/bbox_data/annotations/instances_test.json`
- class-agnostic evaluation: enabled
- NMS IoU threshold: `0.5`
- evaluation score threshold: `0.05`
- visualization score threshold: `0.5`

Example command:

```bash
python3 rtdetrv2_pytorch/tools/eval_and_visualize.py \
  -c rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml \
  -r /path/to/best.pth \
  --images-dir rtdetrv2_pytorch/dataset/bbox_data/images_test \
  --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_test.json \
  --device cuda \
  --save-vis-dir /path/to/eval_vis_test_ca_nms \
  --results-json /path/to/eval_results_test_ca_nms.json \
  --metrics-json /path/to/eval_metrics_test_ca_nms.json \
  --class-agnostic-eval \
  --nms-iou-threshold 0.5 \
  --eval-score-threshold 0.05 \
  --vis-score-threshold 0.5
```

During training, evaluation is computed separately by the built-in validation loop in `rtdetrv2_pytorch/src/solver/det_engine.py`. That path runs the model on the validation loader, applies the RT-DETR postprocessor, and updates the COCO evaluator. This is the source of the validation AP numbers reported during training.



## Metrics And Interpretation

The main metrics reported in this repo are COCO-style bounding-box metrics.

- `AP`: Average Precision averaged across IoU thresholds from `0.50` to `0.95` in steps of `0.05`. This is the main summary metric and is the strictest overall measure.
- `AP50`: Average Precision at IoU `0.50`. This is more forgiving and tells you whether detections roughly overlap the correct objects.
- `AP75`: Average Precision at IoU `0.75`. This is stricter than `AP50` and is more sensitive to tight box placement.

For this project, `AP75` is often especially useful because the main goal is better bounding box placement, not just coarse detection. Higher `AP75` usually indicates better localization quality.

Most evaluations in this repo are class-agnostic, meaning the evaluation primarily measures whether the predicted box lands on the correct object region rather than whether the class label is correct. In that setup, ground-truth and predicted labels are remapped to a single object class during evaluation.

There are two evaluation contexts to keep in mind:
- training-time evaluation: runs automatically through `rtdetrv2_pytorch/src/solver/det_engine.py` and produces the validation AP values shown during training
- manual evaluation: runs through `rtdetrv2_pytorch/tools/eval_and_visualize.py`, where you can add NMS, save visualizations, and export result JSON files

When comparing runs fairly, use the same dataset split, the same checkpoint type, the same class-agnostic setting, and the same NMS / score-threshold settings.

## Training Curves

The loss and metric plots are generated with `rtdetrv2_pytorch/tools/plot_training_log.py`. This script parses either the streamed `train.log` output or the solver `log.txt` file and saves a summary PNG of the training run.

The generated figure has four panels:
- `Training losses`: shows total loss plus the main loss components such as `loss_bbox`, `loss_giou`, and `loss_vfl`
- `Logged LR`: shows the learning rate schedule across epochs
- `Evaluation AP`: shows `AP`, `AP50`, `AP75`, and optionally the best-so-far AP line
- `Loss vs AP`: overlays total loss and `AP` so it is easier to see whether lower loss is translating into better evaluation performance

How to interpret the main loss curves:
- `loss_bbox`: L1 box regression loss, which tracks coordinate accuracy
- `loss_giou`: generalized IoU loss, which tracks geometric overlap quality
- `loss_vfl`: classification loss
- total `loss`: combined optimization objective used during training

In this project, the most important visual patterns are usually:
- `loss_bbox` and `loss_giou` trending down as localization improves
- `AP75` rising along with improved box tightness
- the total loss stabilizing before the validation metrics fully plateau

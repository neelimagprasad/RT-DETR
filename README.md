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

Describe the important folders and files.

Example format:
- `rtdetrv2_pytorch/`: Main training and inference code
- `rtdetrv2_pytorch/configs/`: Training configs
- `rtdetrv2_pytorch/dataset/bbox_data/`: Custom dataset
- `rtdetrv2_pytorch/tools/`: Training, evaluation, export, and utility scripts
- `output/` or external output location: Checkpoints and logs

## Dataset

This project uses a custom COCO-style construction document dataset built from rendered PDF pages. The current annotation files contain `358` train images with `3,206` annotations and `30` test images with `199` annotations across `17` categories.

The images live under `rtdetrv2_pytorch/dataset/bbox_data/images_train` and `rtdetrv2_pytorch/dataset/bbox_data/images_test`, and the COCO JSON files live under `rtdetrv2_pytorch/dataset/bbox_data/annotations`.

For details on how to inspect, edit, or visualize the dataset and its annotations, see the [dataset README](rtdetrv2_pytorch/dataset/bbox_data/README.md).

## Training Setup

Document how training was run.

Suggested things to include:
- base model
- checkpoint used for tuning
- image size
- batch size
- number of GPUs
- optimizer and learning rate
- scheduler
- whether aux loss was enabled
- whether denoising was enabled
- number of epochs
- best-performing run name

## Important Configs

List the important config files and what each one was used for.

Example entries:
- `...`: main baseline config
- `...`: best-performing config
- `...`: ablation config
- `...`: export or deployment-related config

## How To Train

Put the exact training command(s) here.

Suggested subsections:
- baseline training
- best-performing training recipe
- ablation runs

## How To Evaluate

Put the exact evaluation command(s) here.

Suggested things to clarify:
- validation set path
- checkpoint path
- whether evaluation is class-agnostic
- whether NMS is applied
- score threshold / IoU threshold choices

## How To Export To ONNX

Document the `.pth` to ONNX workflow.

Suggested things to include:
- export command
- required dependencies
- expected ONNX inputs/outputs
- any gotchas such as `orig_target_sizes` ordering

## How To Run ONNX Inference

Document how to test the ONNX model.

Suggested things to include:
- sample command or notebook cell
- required input preprocessing
- output format
- where visualization files are saved

## Metrics And Interpretation

Explain how to interpret the reported numbers in this repo.

Suggested things to include:
- what AP/AR numbers mean here
- whether metrics are class-aware or class-agnostic
- any caveats about dataset size or split size
- how to compare runs fairly

## Known Issues

List current limitations, bugs, or confusing parts of the workflow.

Examples:
- distributed training caveats
- denoising/DDP caveats
- ONNX export quirks
- notebook path/quoting gotchas

## Reproducibility Notes

Document anything needed to reproduce the best run exactly.

Suggested things to include:
- seed
- checkpoint source
- hardware
- package versions
- batch size assumptions
- any deviations from default config

## TODO

List upcoming cleanup or improvement tasks.

## Credits

Credit the upstream project and any additional sources used.

Suggested things to include:
- upstream repo
- papers
- collaborators

## Citation

Add the citation(s) you want users of this repo to use.

## License

State the license for this repo and note whether it inherits from upstream.

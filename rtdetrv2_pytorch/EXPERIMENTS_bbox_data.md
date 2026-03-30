# RT-DETR bbox_data Experiments

This run sheet turns the bbox_data fine-tuning plan into concrete configs you
can launch from `rtdetrv2_pytorch/tools/heron-training-script.ipynb`.

## Baseline full fine-tune runs

- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml`
  - Stronger fine-tuning LR, short warmup.
- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_low_lr.yml`
  - Lower LR baseline with the same train/test split.
- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_strong_lr_warmup100.yml`
  - Stronger LR with the longer warmup used in the stronger multi-GPU recipe.

## Backbone-scope run

- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_backbone_frozen.yml`
  - Freezes the full `PResNet` backbone and trains the encoder/decoder stack.

## Head-scope runs

- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_freeze_classifier.yml`
  - Freezes `enc_score_head`, `dec_score_head`, and `denoising_class_embed`.
- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_freeze_bbox.yml`
  - Freezes `enc_bbox_head` and `dec_bbox_head`.
- `rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_heads_only.yml`
  - Freezes the shared trunk and leaves the box/class heads trainable.

## Modal notebook usage

The notebook launcher now accepts `cfg_path`, `resume_path`, and `extra_updates`.
That means the YAML controls LR, scheduler, and freezing rather than the notebook
hardcoding those choices.

Example baseline launch:

```python
with modal.enable_output():
    with app.run(detach=True):
        fc = train_from_heron101.spawn(
            run_name="bbox_full_ft_low_lr",
            cfg_path="rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_low_lr.yml",
        )
        TRAIN_CALL_ID = fc.object_id
        print("Spawned training call:", TRAIN_CALL_ID)
```

Example bbox-focused launch:

```python
with modal.enable_output():
    with app.run(detach=True):
        fc = train_from_heron101.spawn(
            run_name="bbox_freeze_classifier",
            cfg_path="rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101_freeze_classifier.yml",
        )
        TRAIN_CALL_ID = fc.object_id
        print("Spawned training call:", TRAIN_CALL_ID)
```

Example resume:

```python
with modal.enable_output():
    with app.run(detach=True):
        fc = train_from_heron101.spawn(
            run_name="bbox_full_ft_resume",
            cfg_path="rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_bbox_data_heron101.yml",
            resume_path="/outputs/bbox_full_ft/last.pth",
            extra_updates=("epoches=300",),
        )
        TRAIN_CALL_ID = fc.object_id
        print("Spawned training call:", TRAIN_CALL_ID)
```

## Comparing runs

For curves:

```bash
python rtdetrv2_pytorch/tools/plot_training_log.py /path/to/run/log.txt
```

For a run-to-run summary table:

```bash
python rtdetrv2_pytorch/tools/summarize_experiments.py \
  /path/to/run_a \
  /path/to/run_b \
  /path/to/run_c
```

Optional CSV export:

```bash
python rtdetrv2_pytorch/tools/summarize_experiments.py \
  /path/to/run_a \
  /path/to/run_b \
  --csv /tmp/rtdetr_experiment_summary.csv
```

## How to read the results

- Prefer `best_ap` on the held-out `images_test` split for the main ranking.
- Check `AP50` and `AP75` together. If `AP50` rises but `AP75` stays low, the
  model is finding regions but not localizing tightly.
- Compare visualizations at the same threshold across runs so you can separate
  missing boxes from low-confidence boxes.
- Expect `freeze_classifier` to help only if class semantics are already mostly
  correct and localization is the main problem.
- Expect `freeze_bbox` to help only if boxes are mostly right and labels are the
  main source of error.

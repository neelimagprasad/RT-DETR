#!/usr/bin/env python3
"""Plot RT-DETR training curves from a streamed train.log file.

This parser is tailored to the stdout log format emitted by
`rtdetrv2_pytorch/tools/train.py`, such as:

  Averaged stats: lr: 0.000001  loss: 1.64 (...)  loss_bbox: 0.24 (...)
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454

It also understands the JSONL `log.txt` format written by the solver and will
use that when passed instead of the streamed `train.log`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


AVERAGED_PREFIX = "Averaged stats:"
BEST_STAT_PREFIX = "best_stat:"
AP_ALL_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[ IoU=(?P<iou>0\.50:0\.95|0\.50|0\.75)\s+\| area=\s+all \| maxDets=100 \] = (?P<value>-?\d+\.\d+)"
)
AVG_METRIC_RE = re.compile(r"([a-zA-Z0-9_]+): ([0-9eE+\-.]+)(?: \(([0-9eE+\-.]+)\))?")


def _parse_averaged_stats(line: str) -> dict[str, float]:
    payload = line.split(AVERAGED_PREFIX, 1)[1].strip()
    values: dict[str, float] = {}
    for name, raw_value, raw_global in AVG_METRIC_RE.findall(payload):
        value = raw_global if raw_global else raw_value
        values[name] = float(value)
    return values


def _parse_jsonl(path: Path) -> dict[str, list[float]]:
    epochs: list[int] = []
    loss: list[float] = []
    loss_bbox: list[float] = []
    loss_giou: list[float] = []
    loss_vfl: list[float] = []
    lr: list[float] = []
    ap: list[float] = []
    ap50: list[float] = []
    ap75: list[float] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        epochs.append(int(item["epoch"]))
        loss.append(float(item.get("train_loss", float("nan"))))
        loss_bbox.append(float(item.get("train_loss_bbox", float("nan"))))
        loss_giou.append(float(item.get("train_loss_giou", float("nan"))))
        loss_vfl.append(float(item.get("train_loss_vfl", float("nan"))))
        lr.append(float(item.get("train_lr", float("nan"))))
        bbox = item.get("test_coco_eval_bbox", [])
        ap.append(float(bbox[0]) if len(bbox) > 0 else float("nan"))
        ap50.append(float(bbox[1]) if len(bbox) > 1 else float("nan"))
        ap75.append(float(bbox[2]) if len(bbox) > 2 else float("nan"))

    return {
        "epochs": epochs,
        "loss": loss,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
        "loss_vfl": loss_vfl,
        "lr": lr,
        "ap": ap,
        "ap50": ap50,
        "ap75": ap75,
    }


def _parse_stream_log(path: Path) -> dict[str, list[float]]:
    epochs: list[int] = []
    loss: list[float] = []
    loss_bbox: list[float] = []
    loss_giou: list[float] = []
    loss_vfl: list[float] = []
    lr: list[float] = []
    ap: list[float] = []
    ap50: list[float] = []
    ap75: list[float] = []
    best_ap: list[float] = []

    current_epoch = -1
    current_ap: dict[str, float] = {}
    pending_best_ap: float | None = None

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("Epoch: [") and "Total time:" in line:
            try:
                current_epoch = int(line.split("[", 1)[1].split("]", 1)[0])
            except ValueError:
                continue
            continue

        if AVERAGED_PREFIX in line:
            stats = _parse_averaged_stats(line)
            epochs.append(current_epoch)
            loss.append(stats.get("loss", float("nan")))
            loss_bbox.append(stats.get("loss_bbox", float("nan")))
            loss_giou.append(stats.get("loss_giou", float("nan")))
            loss_vfl.append(stats.get("loss_vfl", float("nan")))
            lr.append(stats.get("lr", float("nan")))
            ap.append(current_ap.get("0.50:0.95", float("nan")))
            ap50.append(current_ap.get("0.50", float("nan")))
            ap75.append(current_ap.get("0.75", float("nan")))
            best_ap.append(pending_best_ap if pending_best_ap is not None else float("nan"))
            current_ap = {}
            pending_best_ap = None
            continue

        ap_match = AP_ALL_RE.search(line)
        if ap_match:
            current_ap[ap_match.group("iou")] = float(ap_match.group("value"))
            continue

        if line.startswith(BEST_STAT_PREFIX):
            try:
                stats_dict = ast.literal_eval(line.split(BEST_STAT_PREFIX, 1)[1].strip())
                pending_best_ap = float(stats_dict.get("coco_eval_bbox", float("nan")))
            except Exception:
                pending_best_ap = None

    return {
        "epochs": epochs,
        "loss": loss,
        "loss_bbox": loss_bbox,
        "loss_giou": loss_giou,
        "loss_vfl": loss_vfl,
        "lr": lr,
        "ap": ap,
        "ap50": ap50,
        "ap75": ap75,
        "best_ap": best_ap,
    }


def parse_log(path: Path) -> dict[str, list[float]]:
    if path.name == "log.txt":
        return _parse_jsonl(path)
    return _parse_stream_log(path)


def plot_curves(data: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = data["epochs"]
    if not epochs:
        raise SystemExit("No epochs were parsed from the log.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(epochs, data["loss"], label="loss")
    ax.plot(epochs, data["loss_bbox"], label="loss_bbox")
    ax.plot(epochs, data["loss_giou"], label="loss_giou")
    ax.plot(epochs, data["loss_vfl"], label="loss_vfl")
    ax.set_title("Training losses")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, data["lr"], color="tab:orange")
    ax.set_title("Logged LR")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, data["ap"], label="AP@[0.50:0.95]")
    ax.plot(epochs, data["ap50"], label="AP50")
    ax.plot(epochs, data["ap75"], label="AP75")
    if "best_ap" in data:
        ax.plot(epochs, data["best_ap"], label="best_so_far", linestyle="--")
    ax.set_title("Evaluation AP")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, data["loss"], label="loss")
    ax2 = ax.twinx()
    ax2.plot(epochs, data["ap"], label="AP@[0.50:0.95]", color="tab:green")
    ax.set_title("Loss vs AP")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Loss")
    ax2.set_ylabel("AP")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Path to train.log or log.txt")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path. Defaults next to the input log.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else log_path.with_suffix(".png")
    title = args.title or f"Training curves: {log_path.name}"

    data = parse_log(log_path)
    plot_curves(data, output, title)

    print(f"Saved plot to {output}")
    if data["epochs"]:
        print(f"Parsed {len(data['epochs'])} epochs from {log_path}")
        print(f"Last epoch: {data['epochs'][-1]}")


if __name__ == "__main__":
    main()

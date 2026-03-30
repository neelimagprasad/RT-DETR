#!/usr/bin/env python3
"""Summarize RT-DETR experiment runs from output directories or log files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from plot_training_log import parse_log


def resolve_log_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = [path / "log.txt", path / "train.log"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No log.txt or train.log found under {path}")


def _best_index(values: list[float]) -> int | None:
    best_idx = None
    best_value = float("-inf")
    for idx, value in enumerate(values):
        if value is None or not math.isfinite(value):
            continue
        if value > best_value:
            best_value = value
            best_idx = idx
    return best_idx


def summarize_run(path: Path) -> dict[str, object]:
    log_path = resolve_log_path(path)
    data = parse_log(log_path)

    epochs = data.get("epochs", [])
    if not epochs:
        raise ValueError(f"No epochs parsed from {log_path}")

    ap = data.get("ap", [])
    ap50 = data.get("ap50", [])
    ap75 = data.get("ap75", [])
    loss = data.get("loss", [])
    lr = data.get("lr", [])

    best_idx = _best_index(ap)
    best_ap = ap[best_idx] if best_idx is not None else float("nan")
    best_epoch = epochs[best_idx] if best_idx is not None else None

    return {
        "run": path.name if path.is_dir() else log_path.parent.name,
        "log_path": str(log_path),
        "last_epoch": epochs[-1],
        "best_epoch": best_epoch,
        "best_ap": best_ap,
        "best_ap50": ap50[best_idx] if best_idx is not None and best_idx < len(ap50) else float("nan"),
        "best_ap75": ap75[best_idx] if best_idx is not None and best_idx < len(ap75) else float("nan"),
        "last_loss": loss[-1] if loss else float("nan"),
        "last_lr": lr[-1] if lr else float("nan"),
    }


def format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return "nan"
    if value is None:
        return "-"
    return str(value)


def print_table(rows: list[dict[str, object]]) -> None:
    headers = ["run", "last_epoch", "best_epoch", "best_ap", "best_ap50", "best_ap75", "last_loss", "last_lr"]
    widths = {
        header: max(len(header), max(len(format_value(row[header])) for row in rows))
        for header in headers
    }

    header_row = "  ".join(header.ljust(widths[header]) for header in headers)
    divider = "  ".join("-" * widths[header] for header in headers)
    print(header_row)
    print(divider)
    for row in rows:
        print("  ".join(format_value(row[header]).ljust(widths[header]) for header in headers))


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["run", "log_path", "last_epoch", "best_epoch", "best_ap", "best_ap50", "best_ap75", "last_loss", "last_lr"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run output directories or specific train.log/log.txt files.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path for the summarized comparison.",
    )
    args = parser.parse_args()

    rows = [summarize_run(Path(path)) for path in args.paths]
    rows.sort(key=lambda row: float(row["best_ap"]) if isinstance(row["best_ap"], float) else float("-inf"), reverse=True)

    print_table(rows)

    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"\nWrote CSV summary to {args.csv}")


if __name__ == "__main__":
    main()

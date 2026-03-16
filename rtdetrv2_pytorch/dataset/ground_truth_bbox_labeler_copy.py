#!/usr/bin/env python3
"""
Local interactive bounding-box labeler for rendered PNG pages.

This writes a COCO detection JSON compatible with the RT-DETRv2 dataset loader in this repo
(`rtdetrv2_pytorch/src/data/dataset/coco_dataset.py`) and with common COCO tooling.

Workflow:
  1) Put PDFs under: bbox_data/pdfs/
  2) Render pages to PNGs at 300 DPI:
       python3 rtdetrv2_pytorch/dataset/render_pdf_pages.py --dpi 300
  3) Label boxes on the PNGs and save COCO JSON:
       python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler_copy.py \
         --images-dir bbox_data/images \
         --coco-json bbox_data/annotations/instances_train.json

Controls:
  - n / right arrow: next image
  - p / left arrow : previous image
  - a             : add a new bbox (draw with mouse, Enter=accept, Esc=cancel)
  - Backspace     : delete last bbox on current image
  - x             : clear ALL bboxes on current image
  - s             : save COCO JSON
  - q / Esc       : quit (auto-saves if modified)

Classes:
  This script does NOT try to classify boxes yet. New annotations are created with
  `category_id=0` by default. You can later edit `category_id` manually in the JSON.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent


WINDOW_NAME = "COCO BBox Labeler"
DISPLAY_MAX_DIM = 1600  # downscale for display if larger
COLOR_BOX = (0, 0, 255)  # BGR: red
COLOR_EDIT = (0, 0, 255)  # BGR: red while dragging
COLOR_NEW = (255, 0, 0)  # BGR: blue for newly committed boxes this session
COLOR_TEXT = (10, 10, 10)  # BGR

HUD_BG = (245, 245, 245)  # BGR
HUD_BORDER = (60, 60, 60)  # BGR
HUD_TEXT = COLOR_TEXT
HUD_FONT_SCALE = 0.42
HUD_THICKNESS = 1
HUD_PAD_X = 14
HUD_PAD_Y = 10
HUD_LINE_GAP = 6


@dataclass
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int


def _normalize(b: BBox) -> BBox:
    return BBox(x0=min(b.x0, b.x1), y0=min(b.y0, b.y1), x1=max(b.x0, b.x1), y1=max(b.y0, b.y1))


def _clip(b: BBox, *, w: int, h: int) -> BBox:
    b = _normalize(b)
    return BBox(
        x0=max(0, min(w - 1, b.x0)),
        y0=max(0, min(h - 1, b.y0)),
        x1=max(0, min(w - 1, b.x1)),
        y1=max(0, min(h - 1, b.y1)),
    )


def _valid(b: BBox) -> bool:
    b = _normalize(b)
    return (b.x1 > b.x0) and (b.y1 > b.y0)


def _resize_for_display(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= DISPLAY_MAX_DIM:
        return img, 1.0
    scale = DISPLAY_MAX_DIM / float(max_dim)
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.7,
    thickness: int = 2,
    color: tuple[int, int, int] = COLOR_TEXT,
) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _with_bottom_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Append a bottom HUD band for UI text so it does not cover the image."""
    if not lines:
        return img
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, HUD_FONT_SCALE, HUD_THICKNESS)[0] for t in lines]
    line_h = (max((s[1] for s in sizes), default=0) or 0) + 2
    hud_h = HUD_PAD_Y * 2 + len(lines) * line_h + (len(lines) - 1) * HUD_LINE_GAP

    canvas = np.empty((h + hud_h, w, 3), dtype=img.dtype)
    canvas[:h, :w] = img
    canvas[h:, :] = HUD_BG
    cv2.rectangle(canvas, (0, h), (w - 1, h + hud_h - 1), HUD_BORDER, thickness=1)

    y = h + HUD_PAD_Y + line_h
    for t in lines:
        _draw_text(
            canvas,
            t,
            HUD_PAD_X,
            y,
            scale=HUD_FONT_SCALE,
            thickness=HUD_THICKNESS,
            color=HUD_TEXT,
        )
        y += line_h + HUD_LINE_GAP
    return canvas


def _draw_bbox(img: np.ndarray, b: BBox, *, color: tuple[int, int, int], thickness: int = 3) -> None:
    b = _normalize(b)
    cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), color, thickness)


def _load_images(images_dir: Path, image_glob: str) -> list[Path]:
    paths = sorted([p for p in images_dir.rglob(image_glob) if p.is_file()])
    if not paths:
        raise SystemExit(f"No images found under {images_dir} (glob={image_glob!r})")
    return paths


def _stable_image_id(rel_path: str) -> int:
    # Stable across runs/machines: 31-bit non-negative CRC32 of the relative path.
    # (Python's built-in hash() is salted per process and not stable.)
    return (zlib.crc32(rel_path.encode("utf-8")) & 0x7FFFFFFF) or 1


def _sync_annotations(payload: dict[str, Any], anns_by_image: dict[int, list[dict[str, Any]]]) -> None:
    flat: list[dict[str, Any]] = []
    for image_id in sorted(anns_by_image.keys()):
        flat.extend(anns_by_image[image_id])
    # Keep the JSON stable/readable: sort by annotation id.
    payload["annotations"] = sorted(flat, key=lambda a: int(a.get("id", 0)))


def _default_categories() -> list[dict[str, Any]]:
    # Docling Layout Heron label set (0-16). Even if you don't assign classes yet,
    # having these in `categories` makes later manual `category_id` edits unambiguous.
    names = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
        "Document Index",
        "Code",
        "Checkbox-Selected",
        "Checkbox-Unselected",
        "Form",
        "Key-Value Region",
    ]
    return [{"id": i, "name": n} for i, n in enumerate(names)]


def _load_or_init_coco(coco_json: Path) -> dict[str, Any]:
    if coco_json.exists():
        payload = json.loads(coco_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid COCO JSON (not a dict): {coco_json}")
        payload.setdefault("images", [])
        payload.setdefault("annotations", [])
        payload.setdefault("categories", _default_categories())
        payload.setdefault("info", {})
        return payload
    return {
        "info": {"description": "bbox_data", "version": "1.0", "year": datetime.now().year},
        "images": [],
        "annotations": [],
        "categories": _default_categories(),
    }


def _index_existing(payload: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], int]:
    images_by_id: dict[int, dict[str, Any]] = {int(im["id"]): im for im in payload.get("images", []) if "id" in im}
    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    max_ann_id = 0
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        if "id" in ann:
            max_ann_id = max(max_ann_id, int(ann["id"]))
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        ann_by_image.setdefault(int(image_id), []).append(ann)
    return images_by_id, ann_by_image, max_ann_id


def _ensure_image_record(
    *,
    payload: dict[str, Any],
    images_by_id: dict[int, dict[str, Any]],
    image_path: Path,
    images_dir: Path,
    width: int,
    height: int,
) -> int:
    rel = image_path.relative_to(images_dir).as_posix()
    image_id = _stable_image_id(rel)
    if image_id in images_by_id:
        return image_id
    rec = {"id": image_id, "file_name": rel, "width": int(width), "height": int(height)}
    payload["images"].append(rec)
    images_by_id[image_id] = rec
    return image_id


def _add_boxes_in_window(
    window_name: str, img_full: np.ndarray, *, existing_boxes: list[BBox]
) -> tuple[list[BBox], int]:
    h, w = img_full.shape[:2]
    committed: list[BBox] = []
    state: dict[str, Any] = {"drag": False, "start": (0, 0), "bbox": None, "scale": 1.0}

    def to_full(x_disp: int, y_disp: int) -> tuple[int, int]:
        scale = float(state["scale"]) if float(state["scale"]) > 0 else 1.0
        inv = 1.0 / scale
        return int(round(x_disp * inv)), int(round(y_disp * inv))

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = True
            sx, sy = to_full(x, y)
            state["start"] = (sx, sy)
            state["bbox"] = BBox(x0=sx, y0=sy, x1=sx, y1=sy)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            if state["bbox"] is None:
                return
            fx, fy = to_full(x, y)
            sx, sy = state["start"]
            state["bbox"] = BBox(x0=sx, y0=sy, x1=fx, y1=fy)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
            if state["bbox"] is not None:
                state["bbox"] = _clip(state["bbox"], w=w, h=h)

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        overlay = img_full.copy()
        # Always show any already-saved boxes on this page.
        for b0 in existing_boxes:
            _draw_bbox(overlay, b0, color=COLOR_BOX, thickness=3)
        # Show newly committed boxes in a distinct color.
        for b0 in committed:
            _draw_bbox(overlay, b0, color=COLOR_NEW, thickness=3)
        b = state.get("bbox")
        if isinstance(b, BBox):
            _draw_bbox(overlay, b, color=COLOR_EDIT, thickness=4)
        disp, scale = _resize_for_display(overlay)
        state["scale"] = scale
        hud_lines = [
            "ADD MODE: click-drag to draw | Enter=commit | Esc=done | n/→ next | p/← prev | Backspace=undo | x=clear",
            f"Existing: {len(existing_boxes)}  New (this add): {len(committed)}",
        ]
        cv2.imshow(window_name, _with_bottom_hud(disp, hud_lines))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):  # Esc / q
            return committed, 0
        if key in (ord("n"), 83):  # 'n' or right arrow
            return committed, +1
        if key in (ord("p"), 81):  # 'p' or left arrow
            return committed, -1
        if key == ord("x"):
            committed.clear()
            state["bbox"] = None
            state["drag"] = False
            continue
        if key in (8, 127):  # Backspace
            if isinstance(state.get("bbox"), BBox):
                state["bbox"] = None
                state["drag"] = False
            elif committed:
                committed.pop()
            continue
        if key in (10, 13):  # Enter
            if not isinstance(b, BBox):
                continue
            b = _clip(_normalize(b), w=w, h=h)
            if not _valid(b):
                continue
            committed.append(b)
            state["bbox"] = None
            state["drag"] = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        type=str,
        default=str(DATASET_DIR / "bbox_data" / "images"),
        help="Directory containing PNG pages.",
    )
    p.add_argument("--image-glob", type=str, default="*.png", help="Glob used recursively under images-dir.")
    p.add_argument(
        "--coco-json",
        type=str,
        default=str(DATASET_DIR / "bbox_data" / "annotations" / "instances_train.json"),
    )
    p.add_argument("--default-category-id", type=int, default=0, help="Used for new boxes (placeholder).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    coco_json = Path(args.coco_json).expanduser().resolve()
    coco_json.parent.mkdir(parents=True, exist_ok=True)

    paths = _load_images(images_dir, args.image_glob)
    payload = _load_or_init_coco(coco_json)
    images_by_id, anns_by_image, next_ann_id = _index_existing(payload)
    modified = False

    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(paths):
        img_path = paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {img_path}")
        h, w = img.shape[:2]

        image_id = _ensure_image_record(
            payload=payload,
            images_by_id=images_by_id,
            image_path=img_path,
            images_dir=images_dir,
            width=w,
            height=h,
        )
        anns = anns_by_image.setdefault(image_id, [])

        overlay = img.copy()
        for ann in anns:
            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, bw, bh = [int(round(float(v))) for v in bbox]
                _draw_bbox(overlay, BBox(x0=x, y0=y, x1=x + bw, y1=y + bh), color=COLOR_BOX, thickness=3)

        disp, _scale = _resize_for_display(overlay)
        hud_lines = [
            f"{idx+1}/{len(paths)}  {img_path.name}",
            "n/→ next | p/← prev | a add mode | Backspace del last | x clear | s save | q/Esc quit",
        ]
        cv2.imshow(WINDOW_NAME, _with_bottom_hud(disp, hud_lines))
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q")):  # ESC or q
            break
        if key in (ord("n"), 83):  # 'n' or right arrow
            idx = min(len(paths) - 1, idx + 1)
            continue
        if key in (ord("p"), 81):  # 'p' or left arrow
            idx = max(0, idx - 1)
            continue
        if key == ord("s"):
            coco_json.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
            print(f"[saved] {coco_json}")
            modified = False
            continue
        if key == ord("x"):
            if anns:
                anns.clear()
                modified = True
                _sync_annotations(payload, anns_by_image)
            continue
        if key in (8, 127):  # Backspace (varies by platform)
            if anns:
                anns.pop()
                modified = True
                _sync_annotations(payload, anns_by_image)
            continue
        if key == ord("a"):
            existing_boxes: list[BBox] = []
            for ann in anns:
                bbox = ann.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x, y, bw, bh = [int(round(float(v))) for v in bbox]
                    existing_boxes.append(BBox(x0=x, y0=y, x1=x + bw, y1=y + bh))

            new_boxes, nav = _add_boxes_in_window(WINDOW_NAME, img, existing_boxes=existing_boxes)
            if not new_boxes and nav == 0:
                continue

            for b in new_boxes:
                b = _clip(_normalize(b), w=w, h=h)
                if not _valid(b):
                    continue
                next_ann_id += 1
                anns.append(
                    {
                        "id": int(next_ann_id),
                        "image_id": int(image_id),
                        "category_id": int(args.default_category_id),
                        "bbox": [int(b.x0), int(b.y0), int(b.x1 - b.x0), int(b.y1 - b.y0)],
                        "area": int((b.x1 - b.x0) * (b.y1 - b.y0)),
                        "iscrowd": 0,
                    }
                )
            _sync_annotations(payload, anns_by_image)
            if new_boxes:
                modified = True
            if nav != 0:
                if nav > 0:
                    idx = min(len(paths) - 1, idx + 1)
                else:
                    idx = max(0, idx - 1)
            continue

    cv2.destroyAllWindows()
    if modified:
        coco_json.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        print(f"[saved] {coco_json}")


if __name__ == "__main__":
    main()

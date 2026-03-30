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
         --images-dir rtdetrv2_pytorch/dataset/bbox_data/images \
         --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json

Run from the repo root with:
  python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler_copy.py

Or specify paths explicitly:
  python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler_copy.py \
    --images-dir rtdetrv2_pytorch/dataset/bbox_data/images \
    --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json

Jump to a specific page by zero-based index:
  python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler_copy.py \
    --start-index 120

Controls:
  - n / right arrow: next image
  - p / left arrow : previous image
  - a             : edit boxes on current image (draw new, select, resize, delete)
  - Backspace     : delete last bbox on current image (main view)
  - x             : clear ALL bboxes on current image
  - s             : save COCO JSON
  - q / Esc       : quit (auto-saves if modified)

Edit mode controls (after pressing `a`):
  - click box     : select box
  - drag corner   : resize selected box
  - click-drag    : draw a new box in empty space
  - Enter         : commit new box
  - Backspace     : delete selected box
  - Esc / q       : exit edit mode

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
COLOR_SELECTED = (30, 136, 229)  # BGR: orange/blue highlight for selected box
COLOR_TEXT = (10, 10, 10)  # BGR
HANDLE_RADIUS = 8

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


def _bbox_from_ann(ann: dict[str, Any]) -> BBox | None:
    bbox = ann.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x, y, bw, bh = [int(round(float(v))) for v in bbox]
    return BBox(x0=x, y0=y, x1=x + bw, y1=y + bh)


def _update_ann_bbox(ann: dict[str, Any], b: BBox) -> None:
    b = _normalize(b)
    ann["bbox"] = [int(b.x0), int(b.y0), int(b.x1 - b.x0), int(b.y1 - b.y0)]
    ann["area"] = int((b.x1 - b.x0) * (b.y1 - b.y0))


def _copy_anns(anns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for ann in anns:
        cloned = dict(ann)
        if isinstance(cloned.get("bbox"), list):
            cloned["bbox"] = list(cloned["bbox"])
        copied.append(cloned)
    return copied


def _hit_test_ann(anns: list[dict[str, Any]], x: int, y: int) -> int | None:
    for idx in range(len(anns) - 1, -1, -1):
        bbox = _bbox_from_ann(anns[idx])
        if bbox is None:
            continue
        bbox = _normalize(bbox)
        if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1:
            return idx
    return None


def _corner_points(b: BBox) -> dict[str, tuple[int, int]]:
    b = _normalize(b)
    return {
        "tl": (b.x0, b.y0),
        "tr": (b.x1, b.y0),
        "bl": (b.x0, b.y1),
        "br": (b.x1, b.y1),
    }


def _hit_test_handle(b: BBox, x: int, y: int, tol: int) -> str | None:
    for name, (cx, cy) in _corner_points(b).items():
        if abs(cx - x) <= tol and abs(cy - y) <= tol:
            return name
    return None


def _resize_from_handle(b: BBox, handle: str, x: int, y: int) -> BBox:
    b = _normalize(b)
    if handle == "tl":
        return BBox(x0=x, y0=y, x1=b.x1, y1=b.y1)
    if handle == "tr":
        return BBox(x0=b.x0, y0=y, x1=x, y1=b.y1)
    if handle == "bl":
        return BBox(x0=x, y0=b.y0, x1=b.x1, y1=y)
    return BBox(x0=b.x0, y0=b.y0, x1=x, y1=y)


def _draw_handles(img: np.ndarray, b: BBox) -> None:
    for cx, cy in _corner_points(b).values():
        cv2.circle(img, (cx, cy), HANDLE_RADIUS, COLOR_SELECTED, thickness=-1)
        cv2.circle(img, (cx, cy), HANDLE_RADIUS, (255, 255, 255), thickness=2)


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


def _edit_boxes_in_window(
    window_name: str,
    img_full: np.ndarray,
    *,
    anns: list[dict[str, Any]],
    image_id: int,
    default_category_id: int,
    next_ann_id: int,
) -> tuple[list[dict[str, Any]], int, int]:
    h, w = img_full.shape[:2]
    working = _copy_anns(anns)
    selected_idx: int | None = 0 if working else None
    state: dict[str, Any] = {
        "drag": False,
        "mode": None,
        "start": (0, 0),
        "bbox": None,
        "scale": 1.0,
        "resize_idx": None,
        "resize_handle": None,
    }

    def to_full(x_disp: int, y_disp: int) -> tuple[int, int]:
        scale = float(state["scale"]) if float(state["scale"]) > 0 else 1.0
        inv = 1.0 / scale
        return int(round(x_disp * inv)), int(round(y_disp * inv))

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal selected_idx
        fx, fy = to_full(x, y)
        tol = max(8, int(round(12 / max(float(state["scale"]), 1e-6))))
        if event == cv2.EVENT_LBUTTONDOWN:
            if selected_idx is not None and 0 <= selected_idx < len(working):
                selected_bbox = _bbox_from_ann(working[selected_idx])
                if selected_bbox is not None:
                    handle = _hit_test_handle(selected_bbox, fx, fy, tol)
                    if handle is not None:
                        state["drag"] = True
                        state["mode"] = "resize"
                        state["resize_idx"] = selected_idx
                        state["resize_handle"] = handle
                        return

            hit_idx = _hit_test_ann(working, fx, fy)
            if hit_idx is not None:
                selected_idx = hit_idx
                state["bbox"] = None
                state["drag"] = False
                state["mode"] = None
                return

            state["drag"] = True
            state["mode"] = "draw"
            state["start"] = (fx, fy)
            state["bbox"] = BBox(x0=fx, y0=fy, x1=fx, y1=fy)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            if state["mode"] == "draw":
                if state["bbox"] is None:
                    return
                sx, sy = state["start"]
                state["bbox"] = BBox(x0=sx, y0=sy, x1=fx, y1=fy)
            elif state["mode"] == "resize":
                resize_idx = state.get("resize_idx")
                resize_handle = state.get("resize_handle")
                if resize_idx is None or resize_handle is None or not (0 <= resize_idx < len(working)):
                    return
                bbox = _bbox_from_ann(working[resize_idx])
                if bbox is None:
                    return
                resized = _clip(_resize_from_handle(bbox, str(resize_handle), fx, fy), w=w, h=h)
                if _valid(resized):
                    _update_ann_bbox(working[resize_idx], resized)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drag"] = False
            if state["mode"] == "draw" and state["bbox"] is not None:
                state["bbox"] = _clip(state["bbox"], w=w, h=h)
            state["mode"] = None
            state["resize_idx"] = None
            state["resize_handle"] = None

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        overlay = img_full.copy()
        for idx, ann in enumerate(working):
            b0 = _bbox_from_ann(ann)
            if b0 is None:
                continue
            is_selected = selected_idx is not None and idx == selected_idx
            _draw_bbox(overlay, b0, color=COLOR_SELECTED if is_selected else COLOR_BOX, thickness=5 if is_selected else 3)
            if is_selected:
                _draw_handles(overlay, b0)
        b = state.get("bbox")
        if isinstance(b, BBox):
            _draw_bbox(overlay, b, color=COLOR_EDIT, thickness=4)
        disp, scale = _resize_for_display(overlay)
        state["scale"] = scale
        hud_lines = [
            "EDIT MODE: click empty area to draw | click box to select | drag selected corner to resize",
            "Enter=commit new box | Backspace/Delete=delete selected | n/→ next | p/← prev | x=clear | Esc=done",
            f"Boxes on page: {len(working)}",
        ]
        if selected_idx is not None and 0 <= selected_idx < len(working):
            selected_ann = working[selected_idx]
            hud_lines.append(
                f"Selected ann_id={selected_ann.get('id', 'new')} category_id={selected_ann.get('category_id', default_category_id)}"
            )
        cv2.imshow(window_name, _with_bottom_hud(disp, hud_lines))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):  # Esc / q
            return working, next_ann_id, 0
        if key in (ord("n"), 83):  # 'n' or right arrow
            return working, next_ann_id, +1
        if key in (ord("p"), 81):  # 'p' or left arrow
            return working, next_ann_id, -1
        if key == ord("x"):
            working.clear()
            selected_idx = None
            state["bbox"] = None
            state["drag"] = False
            continue
        if key in (8, 127):  # Backspace
            if isinstance(state.get("bbox"), BBox):
                state["bbox"] = None
                state["drag"] = False
            elif selected_idx is not None and 0 <= selected_idx < len(working):
                working.pop(selected_idx)
                if working:
                    selected_idx = min(selected_idx, len(working) - 1)
                else:
                    selected_idx = None
            continue
        if key in (10, 13):  # Enter
            if not isinstance(b, BBox):
                continue
            b = _clip(_normalize(b), w=w, h=h)
            if not _valid(b):
                continue
            next_ann_id += 1
            working.append(
                {
                    "id": int(next_ann_id),
                    "image_id": int(image_id),
                    "category_id": int(default_category_id),
                    "bbox": [int(b.x0), int(b.y0), int(b.x1 - b.x0), int(b.y1 - b.y0)],
                    "area": int((b.x1 - b.x0) * (b.y1 - b.y0)),
                    "iscrowd": 0,
                }
            )
            selected_idx = len(working) - 1
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
    p.add_argument("--start-index", type=int, default=0, help="Zero-based starting image index.")
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

    idx = max(0, min(len(paths) - 1, int(args.start_index)))
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
            "n/→ next | p/← prev | a edit boxes | Backspace del last | x clear | s save | q/Esc quit",
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
            edited_anns, next_ann_id, nav = _edit_boxes_in_window(
                WINDOW_NAME,
                img,
                anns=anns,
                image_id=image_id,
                default_category_id=int(args.default_category_id),
                next_ann_id=next_ann_id,
            )
            changed = edited_anns != anns
            if not changed and nav == 0:
                continue

            anns_by_image[image_id] = edited_anns
            anns = anns_by_image[image_id]
            _sync_annotations(payload, anns_by_image)
            if changed:
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

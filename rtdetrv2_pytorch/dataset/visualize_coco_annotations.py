#!/usr/bin/env python3
"""
COCO annotation visualizer + lightweight editor (keyboard-driven).

Features:
  - Draw bbox overlays with `annotation_id + category_name (category_id)`
  - Arrow keys cycle images and boxes
  - `c` edits the selected box's `category_id`
  - Backspace/Delete deletes the currently selected box
  - s saves JSON
  - q/Esc quits (auto-saves if modified)

Default paths are relative to this script:
  images:      rtdetrv2_pytorch/dataset/bbox_data/images/
  annotations: rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json

Run from the repo root with:
  python3 rtdetrv2_pytorch/dataset/visualize_coco_annotations.py

Or specify paths explicitly:
  python3 rtdetrv2_pytorch/dataset/visualize_coco_annotations.py \
    --images-dir rtdetrv2_pytorch/dataset/bbox_data/images \
    --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import colorsys

try:
    # Optional: nicer categorical colormaps if available.
    import matplotlib.cm as _mpl_cm  # type: ignore
except Exception:  # pragma: no cover
    _mpl_cm = None


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = DATASET_DIR / "bbox_data"

WINDOW_NAME = "COCO Annotation Visualizer"
DISPLAY_MAX_DIM = 1600

COLOR_BOX = (0, 184, 212)  # BGR
COLOR_SELECTED = (30, 136, 229)  # BGR
COLOR_TEXT = (10, 10, 10)  # BGR
SELECT_FILL_ALPHA = 0.25

LABEL_FONT_SCALE = 1.2  # base; actual per-image label scale is computed dynamically
LABEL_THICKNESS = 3

HUD_BG = (245, 245, 245)  # BGR
HUD_BORDER = (60, 60, 60)  # BGR
HUD_TEXT = COLOR_TEXT
HUD_FONT_SCALE = 0.42
HUD_THICKNESS = 1
HUD_PAD_X = 14
HUD_PAD_Y = 10
HUD_LINE_GAP = 6

RELOAD_POLL_MS = 120


@dataclass(frozen=True)
class Hit:
    ann_id: int
    image_id: int


def _set_category_id(*, anns: list[dict[str, Any]], ann_id: int, category_id: int) -> bool:
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        if int(ann.get("id", -1)) != ann_id:
            continue
        ann["category_id"] = int(category_id)
        return True
    return False


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
    scale: float = 0.65,
    thickness: int = 2,
    color: tuple[int, int, int] = COLOR_TEXT,
) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _with_bottom_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Append a bottom HUD band for viewer UI text (avoid covering the image)."""
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


def _build_category_palette(
    cat_ids: list[int],
    *,
    cmap_name: str = "hsv",
) -> dict[int, tuple[int, int, int]]:
    """
    Build a stable color palette for category IDs.

    We intentionally prefer a "rainbow-ish" palette (default: hsv) because tab20-style
    palettes can look like "mostly blue/orange/brown" at a glance.
    """
    ids = sorted({int(c) for c in cat_ids if int(c) >= 0})
    if not ids:
        return {}

    n = len(ids)
    palette: dict[int, tuple[int, int, int]] = {}

    if _mpl_cm is not None:
        try:
            cmap = _mpl_cm.get_cmap(cmap_name)
        except Exception:
            cmap = _mpl_cm.get_cmap("hsv")
        for i, cid in enumerate(ids):
            t = 0.0 if n <= 1 else (i / float(n - 1))
            r, g, b, _a = cmap(t)
            palette[cid] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        return palette

    # Fallback: deterministic HSV rainbow using index order.
    for i, cid in enumerate(ids):
        t = 0.0 if n <= 1 else (i / float(n - 1))
        r, g, b = colorsys.hsv_to_rgb(t, 0.90, 0.98)
        palette[cid] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    return palette

def _label_scale_for_image(*, h: int, w: int) -> tuple[float, int]:
    """
    Compute a readable font scale for 300-DPI-ish pages.
    Aim: keep category_id readable on large pages (5k-10k px).
    """
    max_dim = float(max(h, w))
    # 2000px -> ~1.2, 6000px -> ~2.6, 10000px -> ~3.5 (clamped)
    scale = 0.6 + (max_dim / 2500.0)
    scale = max(1.2, min(3.6, scale))
    thickness = int(round(scale * 1.6))
    thickness = max(2, min(6, thickness))
    return float(scale), int(thickness)


def _draw_label_bg(
    img: np.ndarray,
    *,
    text: str,
    x: int,
    y: int,
    scale: float,
    thickness: int,
    pad: int = 6,
    bg_color: tuple[int, int, int] = (245, 245, 245),
    border_color: tuple[int, int, int] = (60, 60, 60),
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x0 = int(x)
    y0 = int(y - th - pad)
    x1 = int(x + tw + 2 * pad)
    y1 = int(y + baseline + pad)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1] - 1, x1)
    y1 = min(img.shape[0] - 1, y1)
    cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, thickness=-1)
    cv2.rectangle(img, (x0, y0), (x1, y1), border_color, thickness=2)
    cv2.putText(
        img,
        text,
        (int(x + pad), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def _bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))
    x1 = int(round(float(x + w)))
    y1 = int(round(float(y + h)))
    return x0, y0, x1, y1


def _delete_annotation(
    *,
    payload: dict[str, Any],
    anns_by_image: dict[int, list[dict[str, Any]]],
    hit: Hit,
) -> bool:
    anns = anns_by_image.get(hit.image_id, [])
    if not anns:
        return False
    kept: list[dict[str, Any]] = []
    removed = False
    for ann in anns:
        if not isinstance(ann, dict):
            continue
        if int(ann.get("id", -1)) == hit.ann_id:
            removed = True
            continue
        kept.append(ann)
    if not removed:
        return False
    anns_by_image[hit.image_id] = kept
    _sync_annotations(payload, anns_by_image)
    return True


def _sorted_anns(anns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key_fn(a: dict[str, Any]) -> tuple[int, int]:
        # Prefer stable ordering by (id, area)
        ann_id = int(a.get("id", 0)) if isinstance(a.get("id"), (int, float, str)) else 0
        bbox = a.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                area = float(bbox[2]) * float(bbox[3])
            except Exception:
                area = 0.0
        else:
            area = 0.0
        return (ann_id, int(area))

    return sorted([a for a in anns if isinstance(a, dict)], key=key_fn)


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"COCO JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid COCO JSON (not a dict): {path}")
    payload.setdefault("images", [])
    payload.setdefault("annotations", [])
    payload.setdefault("categories", [])
    return payload


def _safe_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return -1


def _index(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    images = [im for im in payload.get("images", []) if isinstance(im, dict) and "id" in im and "file_name" in im]
    images = sorted(images, key=lambda im: str(im.get("file_name")))

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    cat_name_by_id: dict[int, str] = {}
    for cat in payload.get("categories", []):
        if not isinstance(cat, dict) or "id" not in cat:
            continue
        cid = int(cat["id"])
        name = str(cat.get("name", f"cat_{cid}"))
        cat_name_by_id[cid] = name

    return images, anns_by_image, cat_name_by_id


def _sync_annotations(payload: dict[str, Any], anns_by_image: dict[int, list[dict[str, Any]]]) -> None:
    flat: list[dict[str, Any]] = []
    for image_id in sorted(anns_by_image.keys()):
        flat.extend(anns_by_image[image_id])
    # Keep the JSON stable/readable: sort by annotation id.
    # (Otherwise it ends up grouped by image_id and looks "random" in the file.)
    payload["annotations"] = sorted(flat, key=lambda a: int(a.get("id", 0)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR / "images"),
        help="Directory that `images[*].file_name` is relative to.",
    )
    p.add_argument(
        "--coco-json",
        type=str,
        default=str(DEFAULT_DATA_DIR / "annotations" / "instances_train.json"),
        help="COCO instances JSON to visualize/edit.",
    )
    p.add_argument(
        "--cmap",
        type=str,
        default="hsv",
        help="Matplotlib colormap name used for category colors (e.g. hsv, gist_rainbow, turbo, tab20).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    coco_json = Path(args.coco_json).expanduser().resolve()

    payload = _load_coco(coco_json)
    images, anns_by_image, cat_name_by_id = _index(payload)
    category_palette = _build_category_palette(list(cat_name_by_id.keys()), cmap_name=str(args.cmap))
    if not images:
        raise SystemExit(f"No images found in COCO JSON: {coco_json}")

    modified = False
    selected: Hit | None = None
    last_mtime_ns = _safe_mtime_ns(coco_json)
    category_edit_active = False
    category_edit_buffer = ""

    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # Remember selection index per image for a nicer workflow.
    sel_idx_by_image: dict[int, int] = {}

    def try_reload(*, keep_rel: str | None) -> None:
        nonlocal payload, images, anns_by_image, cat_name_by_id, category_palette, idx, last_mtime_ns, selected
        nonlocal category_edit_active
        nonlocal category_edit_buffer
        # Avoid clobbering unsaved edits made via this tool.
        if modified:
            return
        try:
            new_payload = _load_coco(coco_json)
            new_images, new_anns_by_image, new_cat = _index(new_payload)
            if not new_images:
                return
        except Exception:
            return

        payload = new_payload
        images = new_images
        anns_by_image = new_anns_by_image
        cat_name_by_id = new_cat
        category_palette = _build_category_palette(list(cat_name_by_id.keys()), cmap_name=str(args.cmap))
        last_mtime_ns = _safe_mtime_ns(coco_json)

        if keep_rel is not None:
            for i, im in enumerate(images):
                if str(im.get("file_name")) == keep_rel:
                    idx = i
                    break
            idx = max(0, min(idx, len(images) - 1))
        selected = None
        category_edit_active = False
        category_edit_buffer = ""

    while 0 <= idx < len(images):
        im = images[idx]
        image_id = int(im["id"])
        rel = str(im["file_name"])
        img_path = images_dir / rel

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {img_path}")

        label_scale, label_thickness = _label_scale_for_image(h=int(img.shape[0]), w=int(img.shape[1]))

        anns_raw = anns_by_image.get(image_id, [])
        anns = _sorted_anns(anns_raw)
        anns_by_image[image_id] = anns

        if anns:
            sel_i = sel_idx_by_image.get(image_id, 0)
            sel_i = max(0, min(len(anns) - 1, int(sel_i)))
            sel_idx_by_image[image_id] = sel_i
            selected = Hit(ann_id=int(anns[sel_i].get("id", -1)), image_id=image_id)
        else:
            sel_idx_by_image[image_id] = 0
            selected = None

        def render_and_show() -> None:
            overlay = img.copy()
            # Fill selection first (so borders/text render above it).
            if selected is not None and selected.image_id == image_id:
                sel_ann = next(
                    (a for a in anns if isinstance(a, dict) and int(a.get("id", -1)) == selected.ann_id), None
                )
                if isinstance(sel_ann, dict):
                    bbox = sel_ann.get("bbox")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x0, y0, x1, y1 = _bbox_xywh_to_xyxy([float(v) for v in bbox])
                        sel_cat_id = int(sel_ann.get("category_id", -1))
                        sel_fill_color = category_palette.get(sel_cat_id, COLOR_SELECTED)
                        fill = overlay.copy()
                        cv2.rectangle(fill, (x0, y0), (x1, y1), sel_fill_color, thickness=-1)
                        overlay[:] = cv2.addWeighted(fill, SELECT_FILL_ALPHA, overlay, 1.0 - SELECT_FILL_ALPHA, 0.0)

            for ann in anns:
                bbox = ann.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                x0, y0, x1, y1 = _bbox_xywh_to_xyxy([float(v) for v in bbox])
                ann_id = int(ann.get("id", -1))
                cat_id = int(ann.get("category_id", -1))
                cat_name = cat_name_by_id.get(cat_id, f"cat_{cat_id}")

                is_sel = selected is not None and selected.image_id == image_id and selected.ann_id == ann_id
                base_color = category_palette.get(cat_id, COLOR_BOX)
                if is_sel:
                    # Selection is indicated by a thick highlight outline,
                    # but the inner outline stays category-colored (cmap-based).
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOR_SELECTED, 7)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), base_color, 3)
                else:
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), base_color, 3)
                # Put category_id first since that's what you'll edit most.
                label = f"cid={cat_id}  {cat_name}   ann={ann_id}"
                _draw_label_bg(
                    overlay,
                    text=label,
                    x=x0 + 10,
                    y=max(40, y0 + 40),
                    scale=label_scale,
                    thickness=label_thickness,
                )

            disp, _scale = _resize_for_display(overlay)
            hud_lines: list[str] = [
                f"{idx+1}/{len(images)}  {rel}",
                "Box: ←/→ (or a/d) | Page: ↑/↓ (or p/n or w) | c edit category_id | Backspace=delete | s save | q/Esc quit",
            ]
            if selected is not None and selected.image_id == image_id:
                sel_i = sel_idx_by_image.get(image_id, 0)
                hud_lines.append(f"Selected box {sel_i+1}/{len(anns)}  ann_id={selected.ann_id}")
            if category_edit_active:
                hud_lines.append(
                    f"Edit category_id: {category_edit_buffer or '_'}  | Enter=apply | Backspace=erase | Esc=cancel"
                )
            elif selected is not None and selected.image_id == image_id:
                sel_ann = next(
                    (a for a in anns if isinstance(a, dict) and int(a.get("id", -1)) == selected.ann_id),
                    None,
                )
                if isinstance(sel_ann, dict):
                    hud_lines.append(
                        f"Press c to enter a new category_id (current: {int(sel_ann.get('category_id', -1))})"
                    )
            if modified:
                hud_lines.append("MODIFIED (press 's' to save)")
            disp_hud = _with_bottom_hud(disp, hud_lines)
            cv2.imshow(WINDOW_NAME, disp_hud)

        render_and_show()

        # Small event loop so we can live-reload JSON changes.
        while True:
            # Auto-reload if JSON changed on disk.
            if not modified:
                now_mtime = _safe_mtime_ns(coco_json)
                if now_mtime > 0 and now_mtime != last_mtime_ns:
                    keep = rel
                    try_reload(keep_rel=keep)
                    break

            key = int(cv2.waitKeyEx(RELOAD_POLL_MS))
            if key == -1:
                continue

            # handle key below; if it changes page, break to outer loop
            break

        if key in (27, ord("q")):
            if category_edit_active:
                category_edit_active = False
                category_edit_buffer = ""
                continue
            break

        # Arrow key codes vary by platform / backend.
        LEFT_KEYS = {81, 2424832, 63234}
        RIGHT_KEYS = {83, 2555904, 63235}
        UP_KEYS = {82, 2490368, 63232}
        DOWN_KEYS = {84, 2621440, 63233}

        if category_edit_active:
            if key in (10, 13):  # Enter
                if category_edit_buffer and selected is not None and selected.image_id == image_id:
                    if _set_category_id(
                        anns=anns,
                        ann_id=selected.ann_id,
                        category_id=int(category_edit_buffer),
                    ):
                        modified = True
                category_edit_active = False
                category_edit_buffer = ""
                continue
            if key in (8, 127):
                category_edit_buffer = category_edit_buffer[:-1]
                continue
            if key == 255:
                continue
            if key == 27:
                category_edit_active = False
                category_edit_buffer = ""
                continue
            if ord("0") <= key <= ord("9"):
                category_edit_buffer += chr(key)
                continue
            continue

        # Page nav: Up/Down arrows + fallbacks.
        if key in UP_KEYS or key in (ord("p"), ord("w")):  # prev page
            idx = max(0, idx - 1)
            continue
        if key in DOWN_KEYS or key == ord("n"):  # next page
            idx = min(len(images) - 1, idx + 1)
            continue

        # Box selection: Left/Right arrows + fallbacks.
        if key in LEFT_KEYS or key == ord("a"):  # previous box
            if anns:
                sel_i = sel_idx_by_image.get(image_id, 0)
                sel_i = (sel_i - 1) % len(anns)
                sel_idx_by_image[image_id] = sel_i
                selected = Hit(ann_id=int(anns[sel_i].get("id", -1)), image_id=image_id)
            continue
        if key in RIGHT_KEYS or key == ord("d"):  # next box
            if anns:
                sel_i = sel_idx_by_image.get(image_id, 0)
                sel_i = (sel_i + 1) % len(anns)
                sel_idx_by_image[image_id] = sel_i
                selected = Hit(ann_id=int(anns[sel_i].get("id", -1)), image_id=image_id)
            continue

        if key == ord("c"):
            if selected is None or selected.image_id != image_id:
                continue
            category_edit_active = True
            category_edit_buffer = ""
            continue

        # Delete selected box.
        if key in (8, 127):  # Backspace (macOS delete key often maps to 127)
            if selected is None:
                continue
            if selected.image_id != image_id:
                continue
            if _delete_annotation(payload=payload, anns_by_image=anns_by_image, hit=selected):
                modified = True
                # Refresh selection on this page after deletion.
                anns2 = _sorted_anns(anns_by_image.get(image_id, []))
                anns_by_image[image_id] = anns2
                if anns2:
                    sel_i = min(sel_idx_by_image.get(image_id, 0), len(anns2) - 1)
                    sel_idx_by_image[image_id] = sel_i
                    selected = Hit(ann_id=int(anns2[sel_i].get("id", -1)), image_id=image_id)
                else:
                    sel_idx_by_image[image_id] = 0
                    selected = None
            continue
        # Some platforms emit 255 for the forward-delete key in OpenCV
        if key == 255:
            if selected is None:
                continue
            if selected.image_id != image_id:
                continue
            if _delete_annotation(payload=payload, anns_by_image=anns_by_image, hit=selected):
                modified = True
                anns2 = _sorted_anns(anns_by_image.get(image_id, []))
                anns_by_image[image_id] = anns2
                if anns2:
                    sel_i = min(sel_idx_by_image.get(image_id, 0), len(anns2) - 1)
                    sel_idx_by_image[image_id] = sel_i
                    selected = Hit(ann_id=int(anns2[sel_i].get("id", -1)), image_id=image_id)
                else:
                    sel_idx_by_image[image_id] = 0
                    selected = None
            continue
        if key == ord("s"):
            coco_json.parent.mkdir(parents=True, exist_ok=True)
            coco_json.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
            print(f"[saved] {coco_json}")
            modified = False
            last_mtime_ns = _safe_mtime_ns(coco_json)
            continue

    cv2.destroyAllWindows()
    if modified:
        coco_json.parent.mkdir(parents=True, exist_ok=True)
        coco_json.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        print(f"[saved] {coco_json}")


if __name__ == "__main__":
    main()


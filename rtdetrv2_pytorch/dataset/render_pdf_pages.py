#!/usr/bin/env python3
"""
Render every page of every PDF into PNG images (default: 300 DPI).

This is intended for building a small COCO-style detection dataset from PDFs:
  1) Render PDFs -> PNG pages
  2) Label boxes on PNGs -> instances_*.json (see ground_truth_bbox_labeler_copy.py)

Example:
  python rtdetrv2_pytorch/dataset/render_pdf_pages.py \
    --input-dir bbox_data/pdfs \
    --output-dir bbox_data/images \
    --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent


def _iter_pdfs(input_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p)
    return sorted(out)


def render_pdf_to_pngs(*, pdf_path: Path, output_dir: Path, dpi: int) -> list[Path]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency PyMuPDF.\n"
            "Install with: pip install pymupdf\n"
            f"Original import error: {e}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    scale = float(dpi) / 72.0  # PDF points are 72 DPI
    mat = fitz.Matrix(scale, scale)

    written: list[Path] = []
    stem = pdf_path.stem
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = output_dir / f"{stem}_p{page_idx + 1:04d}.png"
        pix.save(str(out_path))
        written.append(out_path)
    doc.close()
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        type=str,
        default=str(DATASET_DIR / "bbox_data" / "pdfs"),
        help="Directory containing PDFs (recursive).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASET_DIR / "bbox_data" / "images"),
        help="Directory to write PNG pages.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    pdfs = _iter_pdfs(input_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {input_dir}")

    total_pages = 0
    for pdf in pdfs:
        # Skip work if outputs already exist and overwrite not requested.
        if not args.overwrite:
            # cheap heuristic: if page 1 exists, assume rendered
            maybe = output_dir / f"{pdf.stem}_p0001.png"
            if maybe.exists():
                print(f"[skip] {pdf.name} (outputs already exist)")
                continue

        written = render_pdf_to_pngs(pdf_path=pdf, output_dir=output_dir, dpi=int(args.dpi))
        total_pages += len(written)
        print(f"[ok] {pdf.name}: wrote {len(written)} pages -> {output_dir}")

    print(f"Done. Rendered {len(pdfs)} PDFs into {total_pages} PNG pages.")


if __name__ == "__main__":
    main()


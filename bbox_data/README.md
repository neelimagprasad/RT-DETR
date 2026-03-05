## `bbox_data/` quickstart

This folder is for building a small COCO-style detection dataset from your architecture documents.

### Suggested layout

- `bbox_data/pdfs/` : put your source PDFs here
- `bbox_data/images/` : rendered page PNGs (300 DPI)
- `bbox_data/annotations/` : COCO JSON annotations

### 1) Render PDFs to PNG pages (300 DPI)

Install the PDF renderer dependency:

```bash
pip install pymupdf
```

Render:

```bash
python rtdetrv2_pytorch/dataset/render_pdf_pages.py --input-dir bbox_data/pdfs --output-dir bbox_data/images --dpi 300
```

### 2) Draw ground-truth bounding boxes and save COCO JSON

If you don't already have OpenCV:

```bash
pip install opencv-python
```

```bash
python rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler_copy.py \
  --images-dir bbox_data/images \
  --coco-json bbox_data/annotations/instances_train.json
```

This will create a COCO JSON with:
- `images`: page image metadata
- `annotations`: `bbox` in `[x, y, width, height]` pixels
- `categories`: Docling Heron’s 17 class names with IDs `0..16`

You can manually edit `category_id` later.


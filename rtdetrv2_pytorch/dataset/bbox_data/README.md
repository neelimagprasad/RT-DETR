## `rtdetrv2_pytorch/dataset/bbox_data/`

This folder contains the construction document dataset used for Heron 101 fine-tuning. It stores rendered page images plus COCO-format annotation files for training and evaluation.

Current dataset contents in the COCO JSON files:
- train: `358` images, `3,206` annotations
- test: `30` images, `199` annotations
- categories: `17`

## Layout

- `pdfs/`: source PDF files used to create page images
- `images_train/`: training page PNGs
- `images_test/`: test page PNGs
- `annotations/instances_train.json`: COCO training annotations
- `annotations/instances_test.json`: COCO test annotations

## `ground_truth_bbox_labeler.py`

`rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler.py` is the interactive box-drawing tool used to create or update bounding boxes on rendered page images. It opens PNG pages in an OpenCV window and writes a COCO detection JSON that can be used directly by the training pipeline.

What it is for:
- drawing new bounding boxes
- resizing or deleting existing boxes
- saving box coordinates back into a COCO JSON file
- stepping through document pages while labeling

Important note:
- this tool is for box geometry only: drawing, resizing, and deleting bounding boxes
- new boxes are created with `category_id=0` by default
- category labels should be assigned later in `visualize_coco_annotations.py`

Run it on the training split:

```bash
python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler.py \
  --images-dir rtdetrv2_pytorch/dataset/bbox_data/images_train \
  --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json
```

Run it on the test split:

```bash
python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler.py \
  --images-dir rtdetrv2_pytorch/dataset/bbox_data/images_test \
  --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_test.json
```

Useful options:

```bash
python3 rtdetrv2_pytorch/dataset/ground_truth_bbox_labeler.py --start-index 120
```

This starts the labeling session at a specific zero-based image index.

## `visualize_coco_annotations.py`

`rtdetrv2_pytorch/dataset/visualize_coco_annotations.py` is a COCO annotation viewer and lightweight editor. It overlays the saved bounding boxes on top of the page images so you can inspect annotation quality and assign category labels to each annotation.

What it is for:
- visually checking whether boxes match the page content
- browsing annotations image by image
- changing `category_id` values
- deleting incorrect annotations
- saving the updated JSON back to disk

Run it on the training split:

```bash
python3 rtdetrv2_pytorch/dataset/visualize_coco_annotations.py \
  --images-dir rtdetrv2_pytorch/dataset/bbox_data/images_train \
  --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_train.json
```

Run it on the test split:

```bash
python3 rtdetrv2_pytorch/dataset/visualize_coco_annotations.py \
  --images-dir rtdetrv2_pytorch/dataset/bbox_data/images_test \
  --coco-json rtdetrv2_pytorch/dataset/bbox_data/annotations/instances_test.json
```

## Rendering PDFs to PNGs

If you need to create new page images from PDFs, render them first:

```bash
pip install pymupdf
python3 rtdetrv2_pytorch/dataset/render_pdf_pages.py --dpi 300
```

That workflow is typically:
1. place PDFs in `rtdetrv2_pytorch/dataset/bbox_data/pdfs/`
2. render page images
3. draw, resize, and delete bounding boxes with `ground_truth_bbox_labeler.py`
4. review annotations and assign category labels with `visualize_coco_annotations.py`


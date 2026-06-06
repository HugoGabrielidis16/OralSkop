"""
OralSkop — evaluation / inference on a folder of images.

Runs the fine-tuned YOLO model on every image in a directory,
draws coloured masks with a legend (no per-detection text labels),
and saves annotated images to an output folder.

Usage
-----
    python eval.py --weights models/weights/best.pt \
                   --images  ai/dataset/AlphaDent/test/images \
                   --output  eval_results/

    # adjust confidence threshold
    python eval.py --weights best.pt --images ai/dataset/AlphaDent/test/images --conf 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# One vivid colour per class index (BGR for OpenCV)
PALETTE_BGR: list[tuple[int, int, int]] = [
    (  0, 200,   0),   # 0 green
    (  0,  80, 255),   # 1 red-orange
    (  0, 200, 255),   # 2 yellow
    (200,   0, 255),   # 3 purple
    (255, 160,   0),   # 4 blue
    (  0, 210, 210),   # 5 cyan
    (180,  80, 255),   # 6 pink
    ( 80, 255,  80),   # 7 lime
    ( 50, 220, 255),   # 8 gold
]


def color_for(cls_id: int) -> tuple[int, int, int]:
    return PALETTE_BGR[cls_id % len(PALETTE_BGR)]


def draw_legend(image: np.ndarray, names: dict[int, str], present_ids: set[int]) -> np.ndarray:
    """Draw a colour legend in the top-left corner for classes present in the image."""
    if not present_ids:
        return image

    box_size  = 18
    pad       = 8
    line_h    = box_size + 6
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness  = 1

    sorted_ids = sorted(present_ids)
    legend_h   = pad + len(sorted_ids) * line_h + pad
    legend_w   = pad + box_size + 6 + max(
        cv2.getTextSize(names.get(i, f"class_{i}"), font, font_scale, thickness)[0][0]
        for i in sorted_ids
    ) + pad

    # semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (4, 4), (4 + legend_w, 4 + legend_h), (30, 30, 30), -1)
    image = cv2.addWeighted(overlay, 0.65, image, 0.35, 0)

    for row, cls_id in enumerate(sorted_ids):
        y = 4 + pad + row * line_h
        x = 4 + pad
        color = color_for(cls_id)
        cv2.rectangle(image, (x, y), (x + box_size, y + box_size), color, -1)
        cv2.putText(image,
                    names.get(cls_id, f"class_{cls_id}"),
                    (x + box_size + 6, y + box_size - 4),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return image


def annotate_image(image_bgr: np.ndarray, results, alpha: float = 0.45) -> np.ndarray:
    """Overlay coloured masks (no text labels) + legend."""
    overlay = image_bgr.copy()
    present_ids: set[int] = set()
    names: dict[int, str] = results.names

    boxes = results.boxes
    masks = results.masks

    if boxes is None or len(boxes) == 0:
        return image_bgr.copy()

    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        color  = color_for(cls_id)
        present_ids.add(cls_id)

        # filled mask
        if masks is not None and i < len(masks.data):
            mask = masks.data[i].cpu().numpy().astype(np.uint8)
            if mask.shape != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            bool_mask = mask.astype(bool)
            overlay[bool_mask] = (
                overlay[bool_mask] * 0.4 + np.array(color, dtype=np.float32) * 0.6
            ).astype(np.uint8)

            # contour outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        else:
            # fallback: draw box only (no mask)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    annotated = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    annotated = draw_legend(annotated, names, present_ids)
    return annotated


def run(
    weights: str,
    images_dir: str,
    output_dir: str,
    conf: float = 0.25,
    alpha: float = 0.45,
) -> None:
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in images_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}")

    print(f"Loading weights: {weights}")
    model = YOLO(weights)
    print(f"Running inference on {len(image_files)} images → {output_path}/\n")

    for idx, img_path in enumerate(image_files, 1):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue

        results = model.predict(image_bgr, conf=conf, verbose=False)[0]
        n_det   = len(results.boxes) if results.boxes else 0

        annotated = annotate_image(image_bgr, results, alpha=alpha)
        out_file  = output_path / img_path.name
        cv2.imwrite(str(out_file), annotated)

        print(f"  [{idx:>4}/{len(image_files)}] {img_path.name}  — {n_det} detection(s)")

    print(f"\nDone. Annotated images saved to: {output_path}/")


def main() -> None:
    p = argparse.ArgumentParser(description="OralSkop — batch inference with legend.")
    p.add_argument("--weights", required=True,
                   help="Path to fine-tuned YOLO weights (best.pt)")
    p.add_argument("--images",  default="ai/dataset/AlphaDent/test/images",
                   help="Folder of images to run inference on")
    p.add_argument("--output",  default="eval_results",
                   help="Output folder for annotated images")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default 0.25)")
    p.add_argument("--alpha",   type=float, default=0.45,
                   help="Mask opacity 0-1 (default 0.45)")
    args = p.parse_args()

    run(args.weights, args.images, args.output, args.conf, args.alpha)


if __name__ == "__main__":
    main()

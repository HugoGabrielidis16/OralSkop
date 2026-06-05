"""SAM2 inference pipeline with bounding-box prompts for OralSkop.

Wraps SAM2ImagePredictor from Meta's segment-anything-2.
Accepts one or more xyxy bounding boxes, returns binary masks per box.

CLI demo:
    python models/sam2/pipeline.py --image path/to/image.jpg --boxes "100,80,400,350"
    python models/sam2/pipeline.py --image path/to/image.jpg \\
        --boxes "100,80,400,350" "200,100,500,420" --save runs/sam2/result.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "weights" / "sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"   # path inside the sam2 package


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SAM2Pipeline:
    """Load SAM2 once, run inference on multiple images with bounding-box prompts."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        config: str = SAM2_CFG,
        device: str | None = None,
    ):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint}\n"
                f"Download it:\n"
                f"  wget -P models/weights/ "
                f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            )

        sam2_model = build_sam2(config, str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print(f"[sam2] Loaded {checkpoint.name} on {device}")

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Encode an RGB image (H×W×3 uint8) for subsequent predict() calls."""
        self.predictor.set_image(image_rgb)

    def predict(
        self,
        boxes: np.ndarray | list,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run SAM2 with bounding-box prompt(s).

        Args:
            boxes: (N, 4) or (4,) array in xyxy pixel coords.
            multimask_output: Return 3 candidate masks per box if True.

        Returns:
            masks:  bool array (N, H, W)   [best mask per box]
            scores: float array (N,)
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis]

        masks_list, scores_list = [], []
        for box in boxes:
            m, s, _ = self.predictor.predict(box=box, multimask_output=multimask_output)
            # m: (1 or 3, H, W);  pick the best (highest score)
            best = int(np.argmax(s))
            masks_list.append(m[best])
            scores_list.append(float(s[best]))

        return np.stack(masks_list), np.array(scores_list)

    def run(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray | list,
        multimask_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper: set_image + predict in one call."""
        self.set_image(image_rgb)
        return self.predict(boxes, multimask_output=multimask_output)


# ── Visualisation ─────────────────────────────────────────────────────────────

_COLOURS = [
    (255,  80,  80), ( 80, 200,  80), ( 80, 120, 255),
    (255, 180,  50), (200,  80, 200), ( 50, 220, 220),
]

DENTAL_LABELS = {0: "sain", 1: "carie", 2: "lesion_benigne", 3: "lesion_maligne"}


def overlay_masks(
    bgr: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray | None = None,
    labels: list[str] | None = None,
    scores: np.ndarray | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Return BGR image with colored mask overlays and optional box outlines."""
    vis = bgr.copy()
    for i, mask in enumerate(masks):
        color = _COLOURS[i % len(_COLOURS)]
        fill  = vis.copy()
        fill[mask.astype(bool)] = color
        vis = cv2.addWeighted(fill, alpha, vis, 1 - alpha, 0)

        if boxes is not None:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            caption = labels[i] if labels else f"box {i}"
            if scores is not None:
                caption += f" {scores[i]:.2f}"
            cv2.putText(vis, caption, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM2 dental inference demo.")
    p.add_argument("--image",      required=True,    help="Path to input image.")
    p.add_argument("--boxes",      nargs="+",         metavar="X1,Y1,X2,Y2",
                   help="Bounding boxes in xyxy pixel format.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device",     default=None,      help="cuda / cpu (auto-detected if omitted).")
    p.add_argument("--save",       default=None,      help="Save overlay to this path.")
    p.add_argument("--show",       action="store_true", help="Display result in a window.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    boxes: list[list[float]] = []
    if args.boxes:
        for b in args.boxes:
            coords = [float(v) for v in b.split(",")]
            if len(coords) != 4:
                raise ValueError(f"Box must be x1,y1,x2,y2 — got: {b}")
            boxes.append(coords)
    else:
        h, w = rgb.shape[:2]
        boxes = [[0.0, 0.0, float(w), float(h)]]
        print("[sam2] No --boxes given — using full image as prompt.")

    boxes_np = np.array(boxes, dtype=np.float32)

    pipeline = SAM2Pipeline(checkpoint=args.checkpoint, device=args.device)
    masks, scores = pipeline.run(rgb, boxes_np)

    for i, (score, box) in enumerate(zip(scores, boxes_np)):
        area = int(masks[i].sum())
        print(f"[sam2] box {i}  {box.astype(int).tolist()}  score {score:.3f}  mask_area {area}px")

    vis = overlay_masks(bgr, masks, boxes_np, scores=scores)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"[sam2] Overlay saved → {out}")

    if args.show:
        cv2.imshow("SAM2 — OralSkop", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.save and not args.show:
        print("[sam2] Done. Pass --save PATH or --show to view results.")


if __name__ == "__main__":
    main()

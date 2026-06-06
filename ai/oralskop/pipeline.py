"""YOLO11m-seg + SAM2 inference pipeline for dental pathology detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASSES = {0: "sain", 1: "carie", 2: "lesion_benigne", 3: "lesion_maligne"}

# BGR colors per class
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 200, 0),      # green  — sain
    1: (0, 80, 255),     # red    — carie
    2: (0, 200, 255),    # yellow — lesion_benigne
    3: (160, 0, 200),    # purple — lesion_maligne
}

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_SAM2_CKPT    = _PROJECT_ROOT / "models" / "weights" / "sam2.1_hiera_small.pt"
_SAM2_CFG     = "configs/sam2.1/sam2.1_hiera_s.yaml"


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: list[float]       # [x1, y1, x2, y2] in pixels
    mask: Optional[np.ndarray]  # H×W bool array, or None if SAM2 skipped


class OralScopPipeline:
    """
    Run YOLO11m-seg detection then refine each box with SAM2.

    Parameters
    ----------
    yolo_weights:
        Path to fine-tuned weights, or ``"yolo11m-seg.pt"`` for the COCO
        pretrained baseline (before fine-tuning).
    sam2_ckpt:
        Path to SAM2 checkpoint.  Defaults to ``models/weights/sam2.1_hiera_small.pt``.
    device:
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    conf_threshold:
        Minimum YOLO confidence to keep a detection.
    mask_alpha:
        Opacity of the overlay masks (0 = invisible, 1 = fully opaque).
    """

    def __init__(
        self,
        yolo_weights: str | Path = "yolo11m-seg.pt",
        sam2_ckpt: str | Path = _SAM2_CKPT,
        device: str = "auto",
        conf_threshold: float = 0.25,
        mask_alpha: float = 0.45,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf_threshold = conf_threshold
        self.mask_alpha = mask_alpha

        self.yolo = YOLO(str(yolo_weights))

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(_SAM2_CFG, str(sam2_ckpt), device=device)
        self.sam2 = SAM2ImagePredictor(sam2_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self, image_path: str | Path
    ) -> tuple[np.ndarray, list[Detection]]:
        """
        Run the full detection + segmentation pipeline on one image.

        Returns
        -------
        annotated_image : np.ndarray
            BGR array with coloured masks and box labels drawn.
        detections : list[Detection]
            One entry per kept YOLO box.
        """
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        yolo_results = self.yolo.predict(image_bgr, conf=self.conf_threshold, verbose=False)[0]
        boxes = yolo_results.boxes

        if boxes is None or len(boxes) == 0:
            return image_bgr.copy(), []

        self.sam2.set_image(image_rgb)

        detections: list[Detection] = []
        overlay = image_bgr.copy()

        for box in boxes:
            class_id = int(box.cls.item())
            conf     = float(box.conf.item())
            xyxy     = box.xyxy[0].cpu().numpy()

            mask = self._sam2_segment(xyxy)
            if mask is not None:
                color = CLASS_COLORS.get(class_id, (128, 128, 128))
                overlay = self._draw_mask(overlay, mask, color)

            detections.append(Detection(
                class_id=class_id,
                class_name=CLASSES.get(class_id, f"class_{class_id}"),
                confidence=conf,
                box_xyxy=xyxy.tolist(),
                mask=mask,
            ))

        annotated = cv2.addWeighted(overlay, self.mask_alpha, image_bgr, 1 - self.mask_alpha, 0)
        annotated = self._draw_boxes(annotated, detections)
        return annotated, detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sam2_segment(self, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
        try:
            masks, _, _ = self.sam2.predict(box=box_xyxy, multimask_output=False)
            return masks[0].astype(bool)
        except Exception:
            return None

    @staticmethod
    def _draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
        image = image.copy()
        image[mask] = (image[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)
        return image

    @staticmethod
    def _draw_boxes(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        for det in detections:
            color = CLASS_COLORS.get(det.class_id, (128, 128, 128))
            x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return image

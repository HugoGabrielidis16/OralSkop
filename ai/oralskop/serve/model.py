"""Load an OralSkop torchseg checkpoint and run inference on arbitrary RGB photos.

This is the YOLO-free (custom semantic-seg) serving path. The model produces a
per-pixel class map (``0=background``; canonical taxonomy class ``c`` -> ``c+1``,
matching ``torchseg.dataset``). We turn that into:

* **detections** — one record per connected foreground component (class name,
  mean-softmax confidence, pixel area, area fraction, bbox in the *original* image
  coordinates), and
* an **overlay** image (colored masks blended onto the original photo).

The checkpoint carries its own ``arch`` / ``num_classes`` / ``class_names``, so a
served model needs nothing but the ``.pt`` file — no built dataset, no config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from oralskop.torchseg.dataset import AI_ROOT, _MEAN, _STD
from oralskop.torchseg.model import build_model
from oralskop.viz.visualize import color_for  # shared BGR palette (idx = taxonomy id)


@dataclass
class Detection:
    class_id: int          # canonical taxonomy id (0..N-1; background excluded)
    class_name: str
    confidence: float      # mean softmax prob of the class over the component
    area_px: int           # component area in ORIGINAL-image pixels
    area_fraction: float   # area_px / (orig_w * orig_h)
    bbox: list[int]        # [x1, y1, x2, y2] in ORIGINAL-image coordinates

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "area_px": self.area_px,
            "area_fraction": round(self.area_fraction, 6),
            "bbox": self.bbox,
        }


class SegModel:
    """A loaded torchseg model ready to predict on raw image bytes."""

    def __init__(
        self,
        weights: str | Path,
        *,
        arch: str = "deeplabv3_resnet50",
        imgsz: int = 512,
        device: str = "cpu",
        conf: float = 0.5,
        min_area_frac: float = 0.0005,
    ):
        path = Path(weights)
        if not path.is_absolute():
            path = (AI_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        want_cuda = device != "cpu"
        self.device = torch.device("cuda" if want_cuda and torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.min_area_frac = min_area_frac
        self.weights_path = path

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            arch = ckpt.get("arch", arch)
            self.num_seg_classes = ckpt.get("num_classes")
            self.class_names = ckpt.get("class_names")  # {taxonomy_id: name} or None
            self.epoch = ckpt.get("epoch")
            self.miou = ckpt.get("miou", (ckpt.get("metrics") or {}).get("miou"))
        else:
            state = ckpt  # bare state_dict
            self.num_seg_classes = None
            self.class_names = None
            self.epoch = self.miou = None

        if self.num_seg_classes is None:
            # Infer class count from the final classifier conv's output channels.
            out_channels = [t.shape[0] for k, t in state.items()
                            if k.endswith("weight") and t.dim() == 4]
            if not out_channels:
                raise ValueError(
                    "Bare state_dict without metadata; cannot infer num_classes. "
                    "Re-train with save_model (writes arch/num_classes/class_names)."
                )
            self.num_seg_classes = out_channels[-1]
        if not self.class_names:
            self.class_names = {i: f"class_{i}" for i in range(self.num_seg_classes - 1)}

        self.arch = arch
        model = build_model(self.num_seg_classes, arch=arch, pretrained=False)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------ inference
    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 HxWx3 -> normalized [1,3,S,S] tensor (matches training)."""
        resized = cv2.resize(rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
        t = (t - _MEAN) / _STD
        return t.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _raw_predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (pred_mask[S,S] int, prob_map[S,S] float) at model resolution."""
        logits = self.model(self._preprocess(rgb))["out"]  # [1,C,S,S]
        probs = F.softmax(logits, dim=1)[0]                 # [C,S,S]
        conf_map, pred = probs.max(0)                       # [S,S], [S,S]
        return pred.cpu().numpy().astype(np.int32), conf_map.cpu().numpy()

    def predict(self, image_bytes: bytes) -> dict:
        """Run the model on raw image bytes; return JSON-serializable results."""
        rgb, (h, w) = self._decode(image_bytes)
        pred_small, conf_small = self._raw_predict(rgb)

        # Upsample the class map (nearest) and confidence (linear) back to original size.
        pred = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf_small, (w, h), interpolation=cv2.INTER_LINEAR)

        detections = self._components(pred, conf, w, h)
        return {
            "image": {"width": w, "height": h},
            "imgsz": self.imgsz,
            "weights": self.weights_path.name,
            "detections": [d.as_dict() for d in detections],
            # per-class summary (coverage even where no single component clears `conf`)
            "class_coverage": self._coverage(pred, w, h),
        }

    def predict_overlay(self, image_bytes: bytes, *, alpha: float = 0.5) -> bytes:
        """Run the model and return an annotated PNG (colored masks + labels)."""
        rgb, (h, w) = self._decode(image_bytes)
        pred_small, conf_small = self._raw_predict(rgb)
        pred = cv2.resize(pred_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf_small, (w, h), interpolation=cv2.INTER_LINEAR)

        overlay = self._render(rgb, pred, conf, w, h, alpha)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG encoding failed.")
        return buf.tobytes()

    # ------------------------------------------------------------------- helpers
    @staticmethod
    def _decode(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image (unsupported or corrupt file).")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb, (h, w)

    def _components(self, pred: np.ndarray, conf: np.ndarray, w: int, h: int) -> list[Detection]:
        """One Detection per connected component, filtered by conf + min area."""
        min_area = self.min_area_frac * w * h
        dets: list[Detection] = []
        for class_id in range(self.num_seg_classes - 1):  # taxonomy ids; skip bg (0)
            class_mask = (pred == class_id + 1).astype(np.uint8)
            if not class_mask.any():
                continue
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, 8)
            for comp in range(1, n_labels):
                area = int(stats[comp, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                comp_mask = labels == comp
                comp_conf = float(conf[comp_mask].mean())
                if comp_conf < self.conf:
                    continue
                x = int(stats[comp, cv2.CC_STAT_LEFT])
                y = int(stats[comp, cv2.CC_STAT_TOP])
                bw = int(stats[comp, cv2.CC_STAT_WIDTH])
                bh = int(stats[comp, cv2.CC_STAT_HEIGHT])
                dets.append(Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=comp_conf,
                    area_px=area,
                    area_fraction=area / (w * h),
                    bbox=[x, y, x + bw, y + bh],
                ))
        dets.sort(key=lambda d: d.area_px, reverse=True)
        return dets

    def _coverage(self, pred: np.ndarray, w: int, h: int) -> list[dict]:
        total = w * h
        out = []
        for class_id in range(self.num_seg_classes - 1):
            px = int((pred == class_id + 1).sum())
            if px == 0:
                continue
            out.append({
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                "area_px": px,
                "area_fraction": round(px / total, 6),
            })
        out.sort(key=lambda c: c["area_px"], reverse=True)
        return out

    def _render(self, rgb: np.ndarray, pred: np.ndarray, conf: np.ndarray,
                w: int, h: int, alpha: float) -> np.ndarray:
        """Blend colored masks onto the photo and label each kept component."""
        out = rgb.astype(np.float32)
        color = np.zeros_like(rgb)
        for class_id in range(self.num_seg_classes - 1):
            b, g, r = color_for(class_id)  # palette is BGR
            color[pred == class_id + 1] = (r, g, b)
        fg = pred > 0
        out[fg] = (1.0 - alpha) * rgb[fg] + alpha * color[fg]
        out = out.astype(np.uint8)

        for det in self._components(pred, conf, w, h):
            b, g, r = color_for(det.class_id)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (r, g, b), max(1, w // 400))
            label = f"{det.class_name} {det.confidence:.2f}"
            scale = max(0.4, w / 1600)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), (r, g, b), -1)
            cv2.putText(out, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def info(self) -> dict:
        return {
            "weights": self.weights_path.name,
            "arch": self.arch,
            "num_seg_classes": self.num_seg_classes,
            "classes": self.class_names,
            "imgsz": self.imgsz,
            "device": str(self.device),
            "conf": self.conf,
            "min_area_frac": self.min_area_frac,
            "epoch": self.epoch,
            "val_miou": self.miou,
        }

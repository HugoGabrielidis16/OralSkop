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
import os
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


@dataclass
class PreprocessResult:
    """Image actually sent to the model plus how it maps to the original image."""

    rgb: np.ndarray
    crop_box: tuple[int, int, int, int] | None = None
    fallback_reason: str | None = None

    @property
    def crop_applied(self) -> bool:
        return self.crop_box is not None

    def as_dict(self) -> dict:
        out = {"crop_applied": self.crop_applied}
        if self.crop_box is not None:
            out["method"] = "rekognition_mouth"
            out["crop_box"] = list(self.crop_box)
        if self.fallback_reason:
            out["fallback_reason"] = self.fallback_reason
        return out


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
        self.mouth_crop_enabled = True
        self.rekognition_region = (
            os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
        )
        self._rekognition_client = None
        self._rekognition_init_error: str | None = None
        self._rekognition_init_error_reason: str | None = None

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
        rgb, pred, conf, preprocess, (h, w) = self._predict_arrays(image_bytes)

        detections = self._components(pred, conf, w, h)
        return {
            "image": {"width": w, "height": h},
            "imgsz": self.imgsz,
            "weights": self.weights_path.name,
            "preprocess": preprocess.as_dict(),
            "detections": [d.as_dict() for d in detections],
            # per-class summary (coverage even where no single component clears `conf`)
            "class_coverage": self._coverage(pred, w, h),
        }

    def predict_overlay(self, image_bytes: bytes, *, alpha: float = 0.5) -> bytes:
        """Run the model and return an annotated PNG (colored masks + labels)."""
        rgb, pred, conf, _, (h, w) = self._predict_arrays(image_bytes)

        overlay = self._render(rgb, pred, conf, w, h, alpha)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG encoding failed.")
        return buf.tobytes()

    # ------------------------------------------------------------------- helpers
    def _predict_arrays(
        self,
        image_bytes: bytes,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, PreprocessResult, tuple[int, int]]:
        """Return original RGB, original-size pred/conf maps, preprocess metadata."""
        rgb, (h, w) = self._decode(image_bytes)
        preprocess = self._mouth_preprocess(image_bytes, rgb, w, h)
        crop_h, crop_w = preprocess.rgb.shape[:2]

        pred_small, conf_small = self._raw_predict(preprocess.rgb)
        crop_pred = cv2.resize(
            pred_small.astype(np.uint8),
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST,
        )
        crop_conf = cv2.resize(conf_small, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        if preprocess.crop_box is None:
            return rgb, crop_pred, crop_conf, preprocess, (h, w)

        pred = np.zeros((h, w), dtype=np.uint8)
        conf = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = preprocess.crop_box
        pred[y1:y2, x1:x2] = crop_pred
        conf[y1:y2, x1:x2] = crop_conf
        return rgb, pred, conf, preprocess, (h, w)

    @staticmethod
    def _decode(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image (unsupported or corrupt file).")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb, (h, w)

    def _get_rekognition(self):
        if self._rekognition_client is not None:
            return self._rekognition_client
        if self._rekognition_init_error:
            if self._rekognition_init_error_reason == "boto3_unavailable":
                raise ImportError(self._rekognition_init_error)
            raise RuntimeError(self._rekognition_init_error)
        try:
            import boto3

            self._rekognition_client = boto3.client(
                "rekognition",
                region_name=self.rekognition_region,
            )
            return self._rekognition_client
        except ImportError as exc:
            self._rekognition_init_error = str(exc)
            self._rekognition_init_error_reason = "boto3_unavailable"
            raise
        except Exception as exc:  # noqa: BLE001 - prediction must degrade to full image
            self._rekognition_init_error = str(exc)
            self._rekognition_init_error_reason = "rekognition_error"
            raise

    def _mouth_preprocess(self, image_bytes: bytes, rgb: np.ndarray, w: int, h: int) -> PreprocessResult:
        """Crop to Rekognition mouth landmarks, falling back to the full image."""
        if not self.mouth_crop_enabled:
            return PreprocessResult(rgb=rgb, fallback_reason="disabled")
        crop_box, reason = self._rekognition_mouth_box(image_bytes, w, h)
        if crop_box is None:
            return PreprocessResult(rgb=rgb, fallback_reason=reason)
        x1, y1, x2, y2 = crop_box
        return PreprocessResult(rgb=np.ascontiguousarray(rgb[y1:y2, x1:x2]), crop_box=crop_box)

    def _rekognition_mouth_box(
        self,
        image_bytes: bytes,
        w: int,
        h: int,
    ) -> tuple[tuple[int, int, int, int] | None, str | None]:
        """Return a clamped original-image crop box from Rekognition landmarks."""
        try:
            response = self._get_rekognition().detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"],
            )
        except ImportError:
            return None, "boto3_unavailable"
        except Exception:
            return None, "rekognition_error"

        faces = response.get("FaceDetails") or []
        if not faces:
            return None, "no_face"

        landmarks = {p.get("Type"): p for p in faces[0].get("Landmarks", [])}
        needed = ("mouthLeft", "mouthRight", "mouthUp", "mouthDown")
        if any(k not in landmarks for k in needed):
            return None, "missing_mouth_landmarks"

        mouth_points = [landmarks[k] for k in needed]
        try:
            xs = [float(p["X"]) * w for p in mouth_points]
            ys = [float(p["Y"]) * h for p in mouth_points]
        except (KeyError, TypeError, ValueError):
            return None, "invalid_crop"

        mouth_left, mouth_right = min(xs), max(xs)
        mouth_top, mouth_bottom = min(ys), max(ys)
        mouth_width = mouth_right - mouth_left
        if mouth_width <= 0 or mouth_bottom <= mouth_top:
            return None, "invalid_crop"

        pad_x = mouth_width
        pad_y_top = mouth_width
        pad_y_bottom = mouth_width
        x1 = max(0, int(mouth_left - pad_x))
        y1 = max(0, int(mouth_top - pad_y_top))
        x2 = min(w, int(mouth_right + pad_x))
        y2 = min(h, int(mouth_bottom + pad_y_bottom))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, "invalid_crop"
        return (x1, y1, x2, y2), None

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
            "preprocess": {
                "mouth_crop_enabled": self.mouth_crop_enabled,
                "method": "rekognition_mouth",
                "rekognition_region": self.rekognition_region,
            },
            "epoch": self.epoch,
            "val_miou": self.miou,
        }

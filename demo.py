"""
OralSkop — web demo.
Drop a dental photo, get back the annotated image with detected lesions.

Usage
-----
    pip install gradio ultralytics
    python demo.py
    # then open http://localhost:7860

    # custom weights after fine-tuning:
    python demo.py --weights runs/segment/oralskop_v1/weights/best.pt

    # make it reachable from outside (SageMaker / remote server):
    python demo.py --share
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── class colours (BGR → RGB for display) ────────────────────────────────────
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0:  (220,  50,  50),   # red-ish
    1:  ( 50, 200,  50),   # green
    2:  ( 50, 130, 255),   # blue
    3:  (255, 180,   0),   # orange
    4:  (180,   0, 255),   # purple
    5:  (  0, 210, 210),   # cyan
    6:  (255,  80, 180),   # pink
    7:  (120, 255,  80),   # lime
    8:  (255, 220,  50),   # yellow
}


def annotate(image_rgb: np.ndarray, results) -> tuple[np.ndarray, list[dict]]:
    """Draw masks + boxes on the image, return annotated image and detections."""
    annotated = image_rgb.copy()
    overlay   = image_rgb.copy()
    detections = []

    boxes = results.boxes
    masks = results.masks

    names: dict[int, str] = results.names  # {0: "abrasion", ...}

    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        xyxy   = box.xyxy[0].cpu().numpy().astype(int)
        color  = CLASS_COLORS_RGB.get(cls_id % len(CLASS_COLORS_RGB), (200, 200, 200))
        label  = f"{names.get(cls_id, cls_id)} {conf:.0%}"

        # filled mask
        if masks is not None and i < len(masks.data):
            mask = masks.data[i].cpu().numpy().astype(bool)
            if mask.shape != image_rgb.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                ) .astype(bool)
            overlay[mask] = (
                overlay[mask] * 0.45 + np.array(color) * 0.55
            ).astype(np.uint8)

        # bounding box
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({"class": names.get(cls_id, str(cls_id)),
                            "confidence": f"{conf:.1%}",
                            "box": xyxy.tolist()})

    annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)
    return annotated, detections


def build_predict_fn(weights: str, conf: float):
    model = YOLO(weights)

    def predict(image: np.ndarray, confidence: float = conf):
        if image is None:
            return None, "No image provided."

        results = model.predict(image, conf=confidence, verbose=False)[0]
        annotated, detections = annotate(image, results)

        if not detections:
            summary = "No lesions detected."
        else:
            lines = [f"**{d['class']}** — {d['confidence']}" for d in detections]
            summary = f"Found **{len(detections)}** lesion(s):\n\n" + "\n\n".join(lines)

        return annotated, summary

    return predict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11m-seg.pt",
                   help="Path to YOLO weights (default: yolo11m-seg.pt)")
    p.add_argument("--conf",  type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--port",  type=int,   default=7860)
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio link (useful on SageMaker)")
    args = p.parse_args()

    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Run:  pip install gradio")

    predict_fn = build_predict_fn(args.weights, args.conf)

    with gr.Blocks(title="OralSkop — Dental AI", theme=gr.themes.Soft()) as app:
        gr.Markdown("# OralSkop — Dental Lesion Detection\nDrop a dental photo to detect and segment lesions.")

        with gr.Row():
            with gr.Column():
                image_in  = gr.Image(label="Input photo", type="numpy")
                conf_sl   = gr.Slider(0.05, 0.95, value=args.conf, step=0.05,
                                      label="Confidence threshold")
                run_btn   = gr.Button("Analyse", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Annotated result", type="numpy")
                text_out  = gr.Markdown(label="Detections")

        run_btn.click(fn=predict_fn,
                      inputs=[image_in, conf_sl],
                      outputs=[image_out, text_out])
        image_in.change(fn=predict_fn,
                        inputs=[image_in, conf_sl],
                        outputs=[image_out, text_out])

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

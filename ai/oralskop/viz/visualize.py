"""Visualize images with their YOLO-seg segmentation masks overlaid.

Works on either a built ``data.yaml`` (canonical or merged) or on raw image/label
directories. Masks are colored by class id, so when you point it at the canonical
data the colors are consistent across datasets (class 4 == calculus everywhere).

Opens a window and shows the sampled images ONE BY ONE with a color legend.
Keys:  →/space = next   ←/a = previous   s = save current   q/esc = quit

Examples
--------
    # Random 12 images from the built AlphaDent dataset (any split)
    python -m oralskop.viz.visualize --dataset alphadent --num_imgs 12

    # Only the val split, only calculus + gingivitis
    python -m oralskop.viz.visualize --dataset merged --num_imgs 20 --split val --classes 4,5

    # A raw (not-yet-built) dataset, names read from its own data.yaml
    python -m oralskop.viz.visualize --images-dir datasets/CariesRoboflow/train/images \
        --labels-dir datasets/CariesRoboflow/train/labels \
        --names-yaml datasets/CariesRoboflow/data.yaml --num_imgs 8

    # Headless (no display): write overlays to a folder instead
    python -m oralskop.viz.visualize --dataset alphadent --num_imgs 12 --save runs/viz/alphadent

The dataset must be built first:  python -m oralskop.data.prepare --datasets <name>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from oralskop.data.verify import IMAGE_EXTS

# ai/ directory (this file is ai/oralskop/viz/visualize.py -> parents[2] == ai/).
AI_ROOT = Path(__file__).resolve().parents[2]

# Distinct BGR colors, indexed by class id (cycled if more classes than colors).
# The first 7 are maximally distinct so adjacent taxonomy classes (e.g. gingivitis vs
# plaque) never look alike: red, orange, blue, magenta, yellow, green, cyan.
_PALETTE = [
    (0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 0, 255),
    (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255), (88, 88, 88),
]


def color_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def _names_from_yaml(path: Path) -> dict[int, str]:
    names = yaml.safe_load(Path(path).read_text()).get("names")
    if isinstance(names, list):
        return dict(enumerate(names))
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {}


def _pairs_from_data_yaml(data_yaml: Path, split: str) -> list[tuple[Path, Path]]:
    """(images_dir, labels_dir) pairs for the requested split(s) of a data.yaml.

    split="all" includes every split key present among train/val/test.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_keys = ["train", "val", "test"] if split == "all" else [split]
    pairs: list[tuple[Path, Path]] = []
    for key in split_keys:
        if key not in cfg:
            continue
        images_dir = (root / cfg[key]).resolve()
        # YOLO convention: labels live alongside images with /images/ -> /labels/.
        labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
        if images_dir.is_dir():
            pairs.append((images_dir, labels_dir))
    return pairs


def resolve_source(args: argparse.Namespace) -> tuple[list[tuple[Path, Path]], dict[int, str]]:
    """Return ([(images_dir, labels_dir), ...], names) from the chosen source mode."""
    # 1) Built dataset by name -> data/<dataset>/data.yaml
    if args.dataset:
        data_yaml = AI_ROOT / "data" / args.dataset / "data.yaml"
        if not data_yaml.exists():
            raise SystemExit(
                f"Dataset '{args.dataset}' is not built: {data_yaml} not found.\n"
                f"Build it first, e.g.:  "
                f"uv run python -m oralskop.data.prepare --datasets {args.dataset}"
            )
        return _pairs_from_data_yaml(data_yaml, args.split), _names_from_yaml(data_yaml)

    # 2) Explicit data.yaml
    if args.data:
        return _pairs_from_data_yaml(Path(args.data), args.split), _names_from_yaml(Path(args.data))

    # 3) Raw image/label dirs
    if args.images_dir and args.labels_dir:
        names = _names_from_yaml(Path(args.names_yaml)) if args.names_yaml else {}
        return [(Path(args.images_dir).resolve(), Path(args.labels_dir).resolve())], names

    raise SystemExit(
        "Provide a source: --dataset NAME (recommended), or --data data.yaml, "
        "or --images-dir + --labels-dir."
    )


def parse_polygons(label_text: str) -> list[tuple[int, list[float]]]:
    """Parse YOLO-seg lines into (class_id, normalized [x,y,...]) tuples."""
    out: list[tuple[int, list[float]]] = []
    for line in label_text.strip().splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        try:
            out.append((int(parts[0]), [float(v) for v in parts[1:]]))
        except ValueError:
            continue
    return out


def draw_overlay(
    image: np.ndarray,
    polygons: list[tuple[int, list[float]]],
    alpha: float = 0.4,
    classes: set[int] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Return (overlay, {class_id: instance_count}) — masks filled + outlined per color."""
    h, w = image.shape[:2]
    fill = image.copy()
    present: dict[int, int] = {}  # class_id -> instance count (preserves first-seen order)

    for class_id, coords in polygons:
        if classes is not None and class_id not in classes:
            continue
        pts = np.array(
            [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords) - 1, 2)],
            dtype=np.int32,
        )
        if len(pts) < 3:
            continue
        color = color_for(class_id)
        cv2.fillPoly(fill, [pts], color)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        present[class_id] = present.get(class_id, 0) + 1

    out = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
    return out, present


def draw_legend(
    image: np.ndarray,
    class_ids: list[int],
    names: dict[int, str],
    counts: dict[int, int] | None = None,
) -> np.ndarray:
    """Draw a color-swatch legend box (top-left) mapping colors -> class names."""
    if not class_ids:
        return image
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    pad, sw, row_h = 10, 22, 28
    rows = [
        f"{names.get(cid, cid)}" + (f"  ({counts[cid]})" if counts else "")
        for cid in class_ids
    ]
    text_w = max(cv2.getTextSize(r, font, scale, thick)[0][0] for r in rows)
    box_w = pad + sw + 8 + text_w + pad
    box_h = pad + row_h * len(class_ids)

    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (box_w, box_h), (32, 32, 32), -1)
    image = cv2.addWeighted(panel, 0.65, image, 0.35, 0)

    y = pad
    for cid, label in zip(class_ids, rows):
        cv2.rectangle(image, (pad, y + 2), (pad + sw, y + row_h - 8), color_for(cid), -1)
        cv2.rectangle(image, (pad, y + 2), (pad + sw, y + row_h - 8), (255, 255, 255), 1)
        cv2.putText(image, label, (pad + sw + 8, y + row_h - 10),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += row_h
    return image


def _fit_to_screen(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    s = max_dim / max(h, w)
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def render(img_path: Path, label_path: Path, names: dict[int, str],
           alpha: float, classes: set[int] | None,
           max_dim: int) -> tuple[np.ndarray | None, dict[int, int]]:
    """Read an image -> screen-fitted mask overlay with a readable legend on top."""
    image = cv2.imread(str(img_path))
    if image is None:
        return None, {}
    polys = parse_polygons(label_path.read_text()) if label_path.exists() else []
    overlay, present = draw_overlay(image, polys, alpha=alpha, classes=classes)
    # Fit to display size FIRST, then draw the legend so it stays a readable size.
    overlay = _fit_to_screen(overlay, max_dim)
    overlay = draw_legend(overlay, sorted(present), names, counts=present)
    return overlay, present


def view_interactive(sample, names, alpha, classes, max_dim: int) -> None:
    """Show overlays one-by-one in a window. Keys: →/space=next, ←=prev, q/esc=quit."""
    win = "OralSkop — masks  (->/space: next | <-/a: prev | s: save | q: quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i, n = 0, len(sample)
    while True:
        img_path, label_path = sample[i]
        overlay, present = render(img_path, label_path, names, alpha, classes, max_dim)
        if overlay is None:
            print(f"  skip (unreadable): {img_path.name}")
            i = (i + 1) % n
            continue
        cv2.setWindowTitle(win, f"[{i + 1}/{n}] {img_path.name}  -  "
                                f"{sum(present.values())} masks")
        cv2.imshow(win, overlay)
        print(f"  [{i + 1}/{n}] {img_path.name}: "
              f"{ {names.get(k, k): v for k, v in present.items()} }")

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):                      # q / esc
            break
        elif key in (ord("a"), 81, 2):                 # left arrow / a
            i = (i - 1) % n
        elif key == ord("s"):                          # save current frame
            dst = AI_ROOT / "runs/viz" / f"{img_path.stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), overlay)
            print(f"    saved -> {dst}")
        else:                                          # right arrow / space / any other
            i = (i + 1) % n
    cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Randomly sample images from a built dataset and show their "
        "segmentation masks one-by-one in a window. Requires the dataset to be built "
        "first with `python -m oralskop.data.prepare --datasets <name>`.",
    )
    # Primary interface: pick a built dataset by name.
    p.add_argument("--dataset", help="Built dataset name (-> data/<dataset>/data.yaml).")
    p.add_argument("--num_imgs", "--num", dest="num_imgs", type=int, default=12,
                   help="Number of images to sample at random.")
    p.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                   help="Which split(s) to sample from (default: all).")

    # Alternative sources.
    alt = p.add_argument_group("alternative sources")
    alt.add_argument("--data", help="Path to a data.yaml (instead of --dataset).")
    alt.add_argument("--images-dir", help="Raw images dir (with --labels-dir).")
    alt.add_argument("--labels-dir", help="Raw labels dir.")
    alt.add_argument("--names-yaml", help="data.yaml to read class names from (raw mode).")

    # Rendering / display.
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (set for reproducibility).")
    p.add_argument("--alpha", type=float, default=0.4, help="Mask fill opacity.")
    p.add_argument("--classes", help="Comma-separated class ids to show (default: all).")
    p.add_argument("--max-dim", type=int, default=1100, help="Max on-screen image size (px).")
    p.add_argument("--save", metavar="DIR", help="Headless: save overlays to DIR instead "
                   "of opening a window.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs, names = resolve_source(args)
    classes = {int(c) for c in args.classes.split(",")} if args.classes else None

    # Collect every (image, label) candidate across the resolved split dirs.
    candidates: list[tuple[Path, Path]] = []
    for images_dir, labels_dir in pairs:
        if not images_dir.is_dir():
            continue
        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                candidates.append((img, labels_dir / f"{img.stem}.txt"))
    if not candidates:
        raise SystemExit(f"No images found in: {[str(p[0]) for p in pairs]}")
    if args.num_imgs <= 0:
        raise SystemExit("--num_imgs must be a positive integer.")

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.num_imgs, len(candidates)))

    label = args.dataset or (Path(args.data).parent.name if args.data else "dataset")
    print(f"Dataset '{label}' split={args.split}: "
          f"showing {len(sample)} of {len(candidates)} images (seed={args.seed})")
    print(f"Classes: {names or '(none — showing raw ids)'}")

    # Headless save mode.
    if args.save:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path, label_path in sample:
            overlay, present = render(img_path, label_path, names, args.alpha,
                                      classes, args.max_dim)
            if overlay is None:
                print(f"  skip (unreadable): {img_path.name}")
                continue
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), overlay)
            print(f"  {img_path.name}: {sum(present.values())} masks")
        print(f"Saved overlays -> {out_dir}")
        return

    # Interactive window (default).
    try:
        view_interactive(sample, names, args.alpha, classes, args.max_dim)
    except cv2.error as exc:
        raise SystemExit(
            f"Could not open a display window ({exc}).\n"
            f"On a headless machine use --save DIR to write the overlays instead."
        )


if __name__ == "__main__":
    main()

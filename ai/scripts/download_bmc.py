#!/usr/bin/env python3
"""Download the BMC Oral Health 2024 dataset (Google Drive) into datasets/BMCOralHealth.

Usage (from ai/):
    uv run --with gdown python scripts/download_bmc.py

Layout of the Drive folder (verified):
    images/(N).jpg   x3375     grayscale-indexed masks in
    lables/(N).png   x3365     (note the author's misspelling "lables")

Heads-up: this is ~6,700 LOOSE files. Google Drive rate-limits per-file downloads, so a
`gdown` folder pull often throttles partway ("too many accesses"). This script uses
`resume=True` — just re-run it and it continues where it stopped. If it keeps stalling,
the most reliable alternatives are:
  * Browser: open the folder, right-click -> Download (Google zips it server-side),
    then unzip into datasets/BMCOralHealth/.
  * rclone with a Google Drive remote (robust retries for large folders).

After download, the masks use values {1,2,3,4} (background=0). Confirm which value maps to
which condition (see configs/data/bmc_oral_health.yaml), then:
    uv run python -m oralskop.data.prepare --datasets bmc_oral_health
"""

from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1vo_qv3EF9eG4Q2dPvtb_rXb4ttBkYJq_"
AI_ROOT = Path(__file__).resolve().parents[1]
OUT = AI_ROOT / "datasets" / "BMCOralHealth"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BMC Oral Health 2024 -> {OUT}")
    gdown.download_folder(FOLDER_URL, output=str(OUT), quiet=False,
                          use_cookies=False, resume=True)
    print(f"\nDone. Inspect the layout under {OUT} and update "
          f"configs/data/bmc_oral_health.yaml (mask pixel values + image/mask dirs).")


if __name__ == "__main__":
    main()

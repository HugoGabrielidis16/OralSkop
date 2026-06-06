"""Multi-label image classification on the curated manifest dataset.

This package is an INDEPENDENT training path, parallel to the YOLO-seg
(`oralskop.train`) and semantic-seg (`oralskop.torchseg`) paths. It does not use
the converter/`build_dataset`/`data.yaml` pipeline. Instead it reads the curated
`manifest_03_master_FINAL.csv` directly (local or `s3://`), builds multi-hot
targets from `canonical_coarse` / `canonical_fine`, and trains a torchvision
backbone with `BCEWithLogitsLoss`.

See `ai/PASSATION_DATA_OralSkop.md` for the dataset contract and
`configs/clf/manifest_clf.yaml` for the run config.
"""

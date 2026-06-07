# OralSkop — Master Dataset Manifest

OralSkop targets dental-issue analysis from **phone / intraoral RGB photos**. All
source datasets below are unified into a single master manifest,
[`manifest_03_master_FINAL.csv`](manifest_03_master_FINAL.csv), which maps every
image to a common canonical taxonomy regardless of its original label format.

**Totals:** 103,878 images across **26 source datasets** —
48,341 unlabeled (pretraining) + 55,537 labeled (train / valid / test).

> All figures on this page are derived directly from `manifest_03_master_FINAL.csv`.

---

## Manifest schema

One row per image. Columns:

| Column | Description |
|---|---|
| `dataset` | Source dataset key (see table below). |
| `image_path` | Path to the RGB image. |
| `label_path` | Path to the native annotation (empty for unlabeled / classification-only). |
| `annotation_format` | Native label format(s), pipe-joined when an image carries several. |
| `split` | `pretrain`, `train`, `valid`, or `test`. |
| `labels_bruts` | Raw/native class label(s) as provided by the source. |
| `canonical_fine` | Mapped fine-grained canonical label(s), pipe-joined (36 classes). |
| `canonical_coarse` | Mapped coarse canonical label(s), pipe-joined (15 classes). |
| `extra_note` | Free-text provenance / caveats. |
| `patient_id` | Patient/image grouping id used for leak-free splits. |

`canonical_fine` and `canonical_coarse` are multi-label: a single image can carry
several pipe-separated labels (e.g. `anatomie|carie`).

---

## Composition

### By annotation format

| Format | Images |
|---|---:|
| `unlabeled-pretraining` | 48,341 |
| `classif` | 31,646 |
| `yolo-bbox` | 19,342 |
| `classif\|segmentation-mask` | 1,629 |
| `captioning\|yolo-bbox` | 1,011 |
| `classif\|yolo-bbox` | 935 |
| `classif\|segmentation-mask\|yolo-bbox` | 931 |
| `captioning` | 26 |
| `segmentation-mask` | 17 |

### By split

| Split | Images |
|---|---:|
| `pretrain` | 48,341 |
| `train` | 44,423 |
| `valid` | 5,508 |
| `test` | 5,606 |

---

## Source datasets

Ordered by image count. Splits shown as train / valid / test.

| Dataset | Images | Annotation format(s) | Splits (tr / va / te) |
|---|---:|---|---|
| `metadent` | 48,341 | unlabeled-pretraining | pretrain (48,341) |
| `code_ragas_icmr` | 10,976 | classif | 8,926 / 934 / 1,116 |
| `sitol2_oral_14k` | 9,720 | classif (+1,007 seg-mask) | 7,739 / 987 / 994 |
| `mithi_ahmed_2025` | 6,206 | yolo-bbox | 4,960 / 625 / 621 |
| `smart_om` | 5,723 | classif | 4,428 / 690 / 605 |
| `roboflow_dental_data2_87ne6` | 3,843 | yolo-bbox (+classif/seg) | 3,105 / 380 / 358 |
| `dentalmate` | 3,067 | yolo-bbox (+classif/seg) | 2,444 / 302 / 321 |
| `yolo_dental_obj_ddt` | 3,033 | yolo-bbox (+captioning/seg/classif) | 2,405 / 308 / 320 |
| `sajid_oral_diseases` | 2,611 | classif (+622 seg-mask) | 2,103 / 271 / 237 |
| `mouth_detection` | 1,999 | yolo-bbox | 1,599 / 200 / 200 |
| `caries_spectra` | 1,849 | classif | 1,477 / 176 / 196 |
| `thtthada` | 1,704 | yolo-bbox | 1,372 / 164 / 168 |
| `skripsi_yolo11` | 1,611 | yolo-bbox (+classif/seg) | 1,308 / 147 / 156 |
| `tooth_marked_tongue` | 1,250 | classif | 1,000 / 125 / 125 |
| `pranta_dental_segms` | 459 | yolo-bbox (+classif/seg) | 356 / 46 / 57 |
| `mod_rashid` | 410 | classif | 331 / 49 / 30 |
| `oral_images_chandrashekar` | 299 | classif | 241 / 28 / 30 |
| `pknu_calculus` | 205 | classif | 161 / 20 / 24 |
| `digital_health_bg_t98xc` | 134 | classif+yolo-bbox (+seg) | 107 / 12 / 15 |
| `kaggle_oral_cancer_smahmedhassan` | 129 | classif | 105 / 15 / 9 |
| `Kaggle_Shivam_Barot` | 103 | classif | 84 / 12 / 7 |
| `digital_health_bg_teeth_discoured` | 84 | classif+yolo-bbox | 69 / 8 / 7 |
| `decolor_clinician_anno` | 47 | classif+yolo-bbox (+seg) | 40 / 4 / 3 |
| `imageseg_fdams` | 32 | seg-mask+yolo-bbox (+classif) | 27 / 2 / 3 |
| `gingivitis_mgi_hanoi` | 26 | captioning | 21 / 3 / 2 |
| `oral_mamba_liu_2024` | 17 | segmentation-mask | 15 / — / 2 |

`metadent` is the large **unlabeled pretraining** pool; the remaining 25 datasets
are labeled and split into train / valid / test.

---

## Canonical taxonomy

Labels are multi-label, so per-class counts overlap and sum to more than the image
total.

### Coarse (15 classes)

| Coarse class | Images |
|---|---:|
| `image_contexte_non_labellisee` | 48,341 |
| `sain` | 19,596 |
| `carie` | 12,563 |
| `lesion_muqueuse_benigne` | 4,842 |
| `anatomie` | 4,618 |
| `maladie_parodontale` | 3,724 |
| `anomalie_de_couleur` | 2,332 |
| `lesion_muqueuse_suspecte` | 2,124 |
| `depots_dentaires` | 1,744 |
| `anomalie_dentaire` | 1,569 |
| `signe_fonctionnel` | 546 |
| `ortho_structure` | 305 |
| `pathologie_urgente` | 210 |
| `restauration` | 14 |
| `usure_dentaire` | 2 |

### Fine (36 classes)

| Fine class | Images | | Fine class | Images |
|---|---:|---|---|---:|
| `image_contexte_non_labellisee` | 48,341 | | `langue_normale` | 704 |
| `muqueuse_saine` | 16,416 | | `plaque` | 572 |
| `caries_moderee` | 7,888 | | `langue_marquee` | 546 |
| `caries_cavitaire` | 3,613 | | `aphte` | 505 |
| `ulcere` | 2,931 | | `herpes` | 470 |
| `gingivite` | 2,667 | | `candidose` | 426 |
| `dent_objet` | 2,619 | | `gingivostomatite` | 386 |
| `decoloration` | 2,141 | | `caries_dent_permanente` | 326 |
| `caries_dent_temporaire` | 2,025 | | `coloration_externe` | 191 |
| `bouche` | 1,999 | | `malocclusion` | 175 |
| `hypodontie` | 1,569 | | `espacement` | 153 |
| `sain_generique` | 1,294 | | `fracture_fissure` | 148 |
| `caries_initiale` | 1,279 | | `tumeur_benigne_ou_kyste` | 141 |
| `dent_saine` | 1,211 | | `parodontite` | 104 |
| `tartre` | 1,192 | | `gencive_saine` | 77 |
| `cancer_oral` | 1,187 | | `abces` | 62 |
| `gingivite_gradee_MGI` | 1,037 | | `restauration` | 14 |
| `opmd` | 937 | | `erosion_usure` | 2 |

---

## Notes & risks

- **Label-definition mismatch** across sources (one set's "caries" ≠ another's) is
  reconciled by the `labels_bruts` → `canonical_fine` / `canonical_coarse` mapping.
  Treat the canonical columns — not `labels_bruts` — as ground truth.
- **Class imbalance** is severe: the long tail (`usure_dentaire`, `restauration`,
  `parodontite`, …) has very few examples; weight/sample accordingly.
- **Mixed annotation formats** — most labeled images are classification or
  YOLO bounding boxes; only a small subset (~2,600) carries segmentation masks.
- **Licensing** — verify commercial terms per source dataset before anything ships.

> To add or regenerate datasets, follow the workflow in [COMMANDS.md](COMMANDS.md).

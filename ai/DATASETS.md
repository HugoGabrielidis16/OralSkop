# OralSkop — Candidate Datasets (RGB intraoral photos only)

OralSkop targets dental-issue segmentation from **phone / intraoral RGB photos**.
We therefore only consider datasets in the **same modality** — X-ray, panoramic, and
CBCT datasets are intentionally excluded (they do not transfer to phone photos).

Dataset #1 (in use): **AlphaDent** — caries, abrasion, filling, crown.

---

## Shortlist (RGB intraoral photos)

| Dataset | Conditions covered | Label type | Why it fits / notes | Link |
|---|---|---|---|---|
| **Roboflow Universe** (dental / caries / tooth) | caries, plaque, calculus, gingivitis, teeth (varies per set) | Often **YOLO-seg / COCO** | Most practical — exports drop straight into our converter. Quality & **license vary per set**; vet each one. | [universe.roboflow.com](https://universe.roboflow.com/search?q=class%3Adental) |
| **BMC Oral Health 2024** — oral screening segmentation **[INTEGRATED]** | **gingivitis, calculus, plaque, caries** | Pixel masks (PNG) | 3,365 oral-endoscopic images; built & verified. Converter `mask_semantic` + `configs/data/bmc_oral_health.yaml`; mask value→class mapping confirmed (v1 gingivitis, v2 calculus, v3 plaque, v4 caries). Download: `scripts/download_bmc.py`. | [bmcoralhealth article](https://bmcoralhealth.biomedcentral.com/articles/10.1186/s12903-024-05072-1) |
| **Gingivitis intraoral dataset** (image captioning) | gingivitis (anterior teeth + gingiva) | 1,096 high-res images | Controlled-condition photos; good dedicated source for the gingivitis class. | [PubMed 39386321](https://pubmed.ncbi.nlm.nih.gov/39386321/) |
| **Kaggle "Oral Diseases"** (salmansajid05) | caries, gingivitis, discoloration, ulcers, etc. | Mostly **classification** | Large, varied phone-quality photos; useful for mining subsets / relabeling, even if not seg-ready. | [Kaggle](https://www.kaggle.com/datasets/salmansajid05/oral-diseases) |
| **6-class dental dataset for object detection** (Data in Brief 2024) | 6 dental classes | Detection (boxes) | Verify it is intraoral photos (not radiographs) before committing; boxes need conversion to masks or a detection branch. | [PMC 11470401](https://pmc.ncbi.nlm.nih.gov/articles/PMC11470401/) |

Survey of available dental datasets (for finding more):
[Publicly Available Dental Image Datasets for AI (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11633071/).

---

## Selection criteria (apply in order)

1. **Modality** — RGB intraoral / phone photo only (no X-ray, CBCT, panoramic).
2. **Label type** — instance-segmentation polygons (YOLO-seg / COCO) reuse the existing
   converter directly; boxes-only or classification require extra work.
3. **License** — must permit **commercial use** (this becomes a shipped app). Many
   Roboflow / Kaggle sets are research-only — confirm before integrating.
4. **Class overlap** — prefer datasets that add conditions AlphaDent lacks (calculus,
   gingivitis, plaque) to broaden the canonical taxonomy.

## Recommended order to add

1. **A Roboflow YOLO-seg set** — zero-friction with the pipeline; validates the
   multi-dataset merge end-to-end.
2. **BMC calculus / gingivitis / caries** — directly extends the taxonomy into gum and
   calculus disease, the biggest gap vs AlphaDent.

## Risks to budget for

- **Label-definition mismatch** across datasets (one set's "caries" ≠ another's). Reconcile
  annotation guidelines when mapping native classes → canonical taxonomy.
- **Licensing** — verify commercial terms before anything ships.

> To add any of these, follow the "Adding a NEW dataset" section in
> [COMMANDS.md](COMMANDS.md).

# DentalOPG Human–AI Comparison — reproducibility resources

This public repository provides code, fixed-threshold detection metrics, and numerical audit outputs for the manuscript comparing two dentists and three AI detectors on panoramic radiographs. It does **not** establish clinical diagnostic validity.

## Data and scope

- **Internal:** DentalOPG-1550 Version 3 (https://doi.org/10.17632/rtt726b26d.3), 154 evaluation radiographs and 588 target reference boxes; six classes; five evaluators (three detectors and two dentists).
- **External AI-only:** MOPG-7 Version 2 (https://doi.org/10.17632/t66ytdrhf2.2), 210 test radiographs; 1,088 reference boxes across seven original categories, 876 eligible boxes in the four mapped categories.
- The external results **do not** constitute an external five-evaluator human–AI comparison.

## External evaluation — what was reproduced

Using the archived prediction JSON files and official MOPG-7 v2 test annotations, the authors reran class-consistent, maximum-cardinality one-to-one bounding-box matching (then maximum total IoU). The rerun reported `PASS`, with no TP/FP/FN mismatches at IoU 0.50 (primary) and 0.25 (exploratory).

| Model | Confidence | F1 at IoU 0.50 | F1 at IoU 0.25 |
|---|---:|---:|---:|
| RT-DETR-L | 0.50 | 41.47% | 58.42% |
| YOLOv8s | 0.40 | 37.89% | 52.20% |
| YOLO11s | 0.30 | 31.76% | 44.44% |

The official COCO file uses placeholder category labels (`class_0` to `class_6`). **The script validates class IDs against the publisher's `metadata/class_definitions.csv`** before scoring. External category mapping: 0 Missing Teeth, 1 Dental Crown, 2 Root Canal, 3 Caries. Model IDs map 0→3, 1→0, 2→2, 4→1; model IDs 3 and 5 are excluded. Predictions on images without a mapped reference remain eligible to count as false positives.

Run the external self-test:

```bash
python -m pip install -r requirements.txt
python code/reproduce_external_mopg7.py --selftest
```

Then obtain the **official** MOPG-7 Version 2 files `annotations/instances_test.json` and `metadata/class_definitions.csv` from the data publisher, and obtain the original prediction JSONs from the authors (see `docs/INPUTS_TO_OBTAIN.txt`):

```bash
python code/reproduce_external_mopg7.py \
  --coco /path/to/annotations/instances_test.json \
  --class-definitions /path/to/metadata/class_definitions.csv \
  --predictions /path/to/prediction_jsons \
  --out external_reproduced_results
```

Required prediction filenames: `YOLOv8s_predictions.json`, `YOLO11s_predictions.json`, `RT_DETR_L_predictions.json`.

**Public-data limitation (current release):** The three model prediction JSON files are NOT included in the public repository pending the authors' release check. Consequently, the published code and numerical outputs support method inspection and comparison, but the external recomputation cannot be rerun from this repository **alone**. Model training/inference from checkpoints is also not reproduced. The archived run's file SHA-256 values appear in `results/external_reproduction_audit.json`.

## Internal paired bootstrap — revised analysis

The primary internal comparison uses 154 radiographs, all five evaluators and six AI-versus-dentist contrasts, with the same image resample drawn for each evaluator. The revised analysis performed **20,000 paired image-level bootstrap replicates**, seed `20261112`, and **Bonferroni-adjusted 99.17% percentile confidence intervals**. This is a **new analysis undertaken during manuscript revision**; it is **not** described as the unknown-seed historical bootstrap.

The input counts are already available in `data/derived/per_image_detection_counts.csv`. To repeat the updated analysis:

```bash
python code/recalculate_internal_bootstrap.py
```

Outputs:
- `results/main_tables/Table_4_Paired_Bootstrap.csv`
- `results/statistical_audit/Bonferroni_Adjusted_Bootstrap_CIs.csv`

The inputs are per-image detection counts, **not the raw bounding-box annotations**. No patient identifiers are available for patient-level cluster bootstrap. See `results/statistical_audit/internal_bootstrap_new_run_protocol.json`.

## Ethics, reproducibility and licensing

The manuscript reuses publicly released datasets; it involved no new recruitment or imaging. The authors did **not** obtain a separate study-specific written ethics approval or exemption, and do not claim one. This statement does not amount to an institutional determination that ethics review was unnecessary. Follow applicable institutional and journal requirements.

The repository's source code is distributed under the existing **MIT License** in `LICENSE`, subject to confirmation that contributors have authority to license their contributions. This code license does **not** replace the original dataset licenses or grant permission to republish source radiographs. See the publishers' dataset pages for version-specific rights, and verify rights before publishing derived prediction files.

`SHA256_MANIFEST.csv` lists integrity hashes of selected published files. For the current code and result files, refer to the Git commit history for the exact version used.

# MOPG-7 external evaluation — reproducibility package

This is a computational reproduction of the four-class *external AI-only evaluation* in the DentalOPG human–AI manuscript. It does **not** re-run model training or inference from weights, nor independently establish clinical diagnostic validity.

## What has been checked

The manuscript's archived prediction files were scored against the official MOPG-7 v2 COCO test reference: 210 images; 1,088 reference boxes across all seven categories, of which 876 are eligible for the four mapped categories. The uploaded run reported `PASS`, with zero mismatched class-specific TP/FP/FN counts at IoU 0.50 and 0.25. Calculated micro-F1 is 41.46868%/58.42333% (RT-DETR-L), 37.89210%/52.19573% (YOLOv8s), and 31.76265%/44.44444% (YOLO11s) for IoU 0.50/0.25.

The auditor verified internal consistency of the output CSV files and reviewed the run's hashes. Repeating the run requires obtaining the original inputs (not included here). See `docs/INPUTS_TO_OBTAIN.txt`.

## Locked protocol

- Internal training ontology: 0 Caries, 1 Missing Teeth, 2 Root Canal Treatment, 3 Filling, 4 Crown, 5 Unerupted Wisdom Teeth.
- External MOPG-7 official category mapping, **validated against the official class_definitions.csv**: 0 Missing Teeth, 1 Dental Crown, 2 Root Canal, 3 Caries. Other external categories excluded.
- Model → external class mapping: 0→3, 1→0, 2→2, 4→1; model classes 3 and 5 excluded.
- Frozen operating confidence thresholds: YOLOv8s 0.40; YOLO11s 0.30; RT-DETR-L 0.50.
- Bounding boxes matched within class and image, one-to-one, maximizing number of accepted matches then sum of IoUs. Two thresholds: 0.50 primary; 0.25 exploratory.
- Predictions on test images without mapped reference labels are **not** ignored. Unmatched eligible predictions count as FPs; unmatched eligible reference labels as FNs.

## Requirements and command

```bash
python -m pip install -r requirements.txt
python code/reproduce_external_mopg7.py --selftest
python code/reproduce_external_mopg7.py \
    --coco /path/to/annotations/instances_test.json \
    --class-definitions /path/to/metadata/class_definitions.csv \
    --predictions /path/to/private_prediction_jsons \
    --out /path/to/output
```

Expected prediction JSON filenames: `YOLOv8s_predictions.json`, `YOLO11s_predictions.json`, `RT_DETR_L_predictions.json`.

Results include `external_per_image_class_recalculated.csv`, `external_per_image_recalculated.csv`, `external_micro_f1_recalculated.csv`, `external_reproduction_audit.json`. If the recomputation does not match archived class counts, the script fails closed by default (do not alter thresholds to force a match).

## Availability, licensing, and provenance

- MOPG-7 Version 2: https://doi.org/10.17632/t66ytdrhf2.2 — download and check license terms directly with the data publisher.
- Internal DentalOPG-1550 Version 3: https://doi.org/10.17632/rtt726b26d.3
- This package includes only code and non-image derived numerical reports. It intentionally excludes images, patient data, trained weights, and the three raw prediction JSONs pending the authors' release review.
- **No software license has been chosen for these authors' code yet.** Do not assert a particular code license until the authors select and add one.
- Report `reference_file_sha256` and model `prediction_json_sha256` fields correspond to files in the authors' run and enable identity checks. They do not independently verify training provenance.
- The audit JSON field `category_id_mapping_from_COCO_names` in the archived run should be read as *metadata-verified numeric ID mapping*, since the COCO category names are placeholders (`class_0` etc.) and the official `class_definitions.csv` supplies the semantics. This release script uses the corrected audit field name.

## Publication requirement

Before claiming public availability in a reviewer response, upload an approved version to the journal-submitted repository, verify the public URL/commit SHA, check all included data and filenames for confidentiality, and add the actual repository link in manuscript and response. The public status should be assessed from the actual GitHub repository or release; this README does not certify archival.

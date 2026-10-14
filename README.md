# DentalOPG Human–AI Comparison

**Research title:** Influence of Bounding-Box Matching Criteria on Human–AI Performance Comparisons for Six Dental Findings in Panoramic Radiographs: A Five-Evaluator Study

This repository contains research code, methodological documentation, and derived evaluation tables for a comparative, annotation-based object-detection study involving three deep-learning detectors and two independent dentists. It is intended to document the study's analysis and support reproducibility; it is **not** a clinically validated diagnostic tool.

## Study overview

- **Source dataset:** DentalOPG-1550, Mendeley Data, Version 3 ([DOI: 10.17632/rtt726b26d.3](https://doi.org/10.17632/rtt726b26d.3)).
- **Prepared dataset:** 1,522 panoramic radiographs: 1,065 training, 303 validation, and 154 internal test images.
- **Reference annotations in the test set:** 588 bounding boxes for six target categories: caries, missing teeth, root canal treatment, filling, crown, and unerupted wisdom teeth. The source annotation category *Cap* was mapped to *Crown*.
- **AI models:** YOLOv8s, YOLO11s, and RT-DETR-L.
- **Human observers:** Two dentists independently annotated all 154 test radiographs.
- **Evaluation:** Class-consistent, one-to-one bounding-box matching, with a primary IoU threshold of 0.50 and a post-hoc exploratory IoU threshold of 0.25.
- **Locked model confidence thresholds (selected on validation data):** YOLOv8s 0.40; YOLO11s 0.30; RT-DETR-L 0.50.
- **Statistical analysis:** Paired image-level bootstrap; the multiple-comparison analysis used 20,000 resamples and Bonferroni adjustment for six AI–dentist contrasts.

### Overall micro-F1 on the internal test set

| Evaluator | IoU ≥ 0.50 | IoU ≥ 0.25 (exploratory) |
|---|---:|---:|
| Dentist 01 | 55.68% | 86.38% |
| Dentist 02 | 60.97% | 91.12% |
| YOLOv8s | 64.81% | 74.44% |
| YOLO11s | 60.47% | 67.72% |
| RT-DETR-L | 65.48% | 76.90% |

These figures quantify agreement with existing dataset reference annotations. They **must not** be interpreted as definitive clinical diagnostic accuracy, clinical superiority, or external validation.

## Repository structure

The release is organized as follows. Some components are included only after the final release audit and may be absent from preliminary repository drafts.

```text
DentalOPG-Human-AI-Comparison/
├── README.md
├── requirements.txt
├── code/
│   ├── notebooks/
│   │   └── DentalOPG_Research_Pipeline_Clean.ipynb
│   ├── 05_Human_Observer_Study/
│   │   └── Final_Comparative_Analysis/
│   │       └── DentalOPG_Reader_Analysis_Engine.py
│   └── reproduce_metrics.py                  # after derived-data validation
├── data/
│   └── derived/
│       ├── per_image_detection_counts.csv    # after validation
│       └── README.md                         # after validation
├── results/
│   ├── main_tables/
│   │   ├── Table_1_Test_Dataset.csv
│   │   ├── Table_2_Combined_Performance.csv
│   │   ├── Table_3_Per_Class_F1.csv
│   │   └── Table_4_Paired_Bootstrap.csv
│   ├── supplementary_tables/
│   │   ├── Supplementary_Table_S1_Training.csv
│   │   ├── Supplementary_Table_S2_Primary_Per_Class.csv
│   │   ├── Supplementary_Table_S3_IoU_Sensitivity.csv
│   │   └── Supplementary_Table_S4_Methodology_Audit.csv
│   └── statistical_audit/
│       └── Bonferroni_Adjusted_Bootstrap_CIs.csv
└── documentation/
    └── Training_Configuration_Extract.csv
```

## Access to the original dataset

The original panoramic radiographs and source annotations are **not redistributed here**. Obtain the dataset from its original publisher using the Mendeley Data DOI above and follow the source licensing conditions. The evaluation used a prepared six-class subset with three confirmed duplicate images removed from the training partition. The absence of patient identifiers prevents verification of full patient-level independence.

## Reproducing the analysis

1. Obtain the source data from Mendeley Data and review the study's dataset preparation procedure before attempting image-level inference or model training.
2. Install the software dependencies listed in `requirements.txt`, preferably in an isolated Python environment. The experiments used Ultralytics version 8.4.175 and a Google Colab NVIDIA T4 environment; exact package/runtime versions and pretrained weights matter for full retraining.
3. Inspect `code/notebooks/DentalOPG_Research_Pipeline_Clean.ipynb`. It documents the original workflow but contains historical Colab/Google Drive paths and may require path configuration and access to original CVAT exports, cached predictions, and trained checkpoints. **Do not assume that running all cells sequentially is safe or sufficient to reproduce the paper.**
4. The CSV files under `results/` are the published summary outputs. Once `data/derived/per_image_detection_counts.csv` and `code/reproduce_metrics.py` have passed the release audit, run:

   ```bash
   python code/reproduce_metrics.py
   ```

   This command recalculates aggregate precision, recall, and micro-F1 from cached per-image TP/FP/FN counts. It does not retrain detectors or recompute bounding-box assignments from raw images.
5. Exact replication of bootstrap confidence intervals requires the relevant image-level inputs, resampling procedure, random-state configuration, and matching implementation. Summary tables alone do not reproduce those intervals.

## Interpretation and limitations

- This is an **internal, annotation-based benchmark**, not an independent external validation study.
- The reference boxes were not comprehensively adjudicated by an independent clinical panel across all 154 test images.
- The IoU ≥ 0.25 comparison was exploratory and post hoc; alternative localization conventions can materially change evaluator rankings.
- The two human observers had one and two years of clinical experience; results cannot be generalized to all dental specialists.
- Patient-level independence could not be established from the available source data.
- The targeted specialist review examined selected reference-negative images only and did not replace the operational reference standard for the full test set.

## Research integrity and data handling

The public release must not include API keys, access tokens, private Colab outputs, identifiable patient information, unrevised CVAT exports, private audit reports, proprietary material, or redistributed source radiographs. The final files should be reviewed manually before publication. Research artifacts and software code may be subject to different licenses; select and publish an appropriate license only after confirming ownership and reuse permissions.

## Citation and archived version

Please cite the associated journal article once published and the source dataset linked above. The permanent Zenodo DOI for this repository will be added **only after a public, versioned GitHub release has been successfully archived**. No repository DOI has been assigned in this README.

## Status

**Release candidate — pending final privacy, dependency, and reproducibility checks.** The GitHub repository and Zenodo record should not be advertised as a completed public reproducibility package until these checks pass.

# DentalOPG Human–AI Comparison

**Study title:** Influence of Bounding-Box Matching Criteria on Human–AI Performance Comparisons for Six Dental Findings in Panoramic Radiographs: A Five-Evaluator Study

Public code and derived results for a retrospective, **annotation-based internal comparison**, not a validated clinical diagnosis system. Three object detectors (YOLOv8s, YOLO11s, RT-DETR-L) and two independently annotating restorative dentists were compared with existing DentalOPG-1550 reference annotations. Full test-set adjudication and independent external validation were **not** performed.

## Dataset and methods

- **Source:** [DentalOPG-1550, Mendeley Data, version 3](https://doi.org/10.17632/rtt726b26d.3). Original radiographs are **not** redistributed in this repository.
- **Frozen analytical set:** 1,522 images; train 1,065 / validation 303 / internal test 154; three confirmed duplicates removed from the training partition.
- **Test reference:** 588 boxes from six classes: Caries, Missing Teeth, Root Canal Treatment, Filling, Crown (source: Cap), and Unerupted Wisdom Teeth.
- **Locked model operating thresholds:** YOLOv8s 0.40, YOLO11s 0.30, RT-DETR-L 0.50; determined from validation images.
- **Matching:** class-consistent, cardinality-first, one-to-one, using a primary IoU >= 0.50 and an exploratory post-hoc IoU >= 0.25.
- **Statistical comparisons:** paired image-level bootstrap, 20,000 replicates (`numpy.random.default_rng(20261014)`), Bonferroni adjustment across six AI-vs-dentist comparisons.

### Micro-averaged F1 (%) on the internal test set

| Evaluator | IoU >= 0.50 | Exploratory IoU >= 0.25 |
|---|---:|---:|
| Dentist 01 | 55.68 | 86.38 |
| Dentist 02 | 60.97 | 91.12 |
| YOLOv8s | 64.81 | 74.44 |
| YOLO11s | 60.47 | 67.72 |
| RT-DETR-L | 65.48 | 76.90 |

Values indicate **agreement with dataset boxes**, not independently adjudicated diagnostic sensitivity or clinical superiority.

## Repository structure

```text
README.md
LICENSE
CITATION.cff
.github/workflows/reproducibility.yml
.gitignore
requirements.txt
code/
  notebooks/DentalOPG_Research_Pipeline_Clean.ipynb
  DentalOPG_Reader_Analysis_Engine.py
  reproduce_metrics.py
  reproduce_bootstrap.py
  run_reproducibility_checks.py
data/derived/
  README.md
  per_image_detection_counts.csv
  per_image_class_detection_counts.csv
documentation/
  Training_Configuration_Extract.csv
  RELEASE_PRIVACY_REVIEW.md
  RELEASE_CHECKLIST_AR.md
  REPRODUCIBILITY_VERIFICATION.md
results/
  main_tables/Table_1_Test_Dataset.csv
  main_tables/Table_2_Combined_Performance.csv
  main_tables/Table_3_Per_Class_F1.csv
  main_tables/Table_4_Paired_Bootstrap.csv
  supplementary_tables/Supplementary_Table_S1_Training.csv
  supplementary_tables/Supplementary_Table_S2_Primary_Per_Class.csv
  supplementary_tables/Supplementary_Table_S3_IoU_Sensitivity.csv
  supplementary_tables/Supplementary_Table_S4_Methodology_Audit.csv
  statistical_audit/Bonferroni_Adjusted_Bootstrap_CIs.csv
```

## Reproduce the published summary metrics and adjusted bootstrap intervals

Clone the repository, install dependencies in a Python environment, then run:

```bash
python -m pip install -r requirements.txt
python code/reproduce_metrics.py
python code/reproduce_bootstrap.py --check
# Or use the combined checker:
python code/run_reproducibility_checks.py
```

The GitHub Actions workflow (`.github/workflows/reproducibility.yml`) runs both checks on each push to `main`, pull request, and release. **Check the green Actions result before archiving.**

`code/reproduce_metrics.py` recomputes 10 evaluator/IoU groups from the public per-image TP/FP/FN counts and checks the main Table 2. `code/reproduce_bootstrap.py` implements the primary IoU >= 0.50 paired 20,000-replicate bootstrap with the **same random seed, case ordering and shared resamples** documented in the research notebook, and checks the published Bonferroni-adjusted Table 4 and audit CSV. The computations do not retrain models or repeat the bounding-box matching against source images. The supplied notebook has its saved execution outputs removed, but retains historical Colab and Google Drive paths and is **not** an independently runnable clean-room training pipeline. Its cells include file modifications and should not be executed indiscriminately.

The original reference images, complete human CVAT exports, predictions and trained model checkpoints are not redistributed here. Patient-level independence could not be established, and the specialist discrepancy review covered only eight selected reference-negative images. See `data/derived/README.md` and the article for further limitations.

## Research integrity and reuse

The repository's **MIT license covers original software code** to the extent it is owned and licensable by the listed contributors. It does **not** supersede the original DentalOPG-1550 dataset's license or independently establish redistribution rights for radiographs. Users must obtain original images via Mendeley Data and comply with its license. The numerical derivative tables are published as research outputs; any reuse should cite this repository and the underlying source dataset and comply with applicable rights.

`documentation/RELEASE_PRIVACY_REVIEW.md` describes the audit limitations and the separately documented historical notebook with embedded outputs. A previous commit can remain retrievable after deletion from the main branch. The maintainers should verify the historical material and, if necessary, purge sensitive content from the full Git history before official archiving.

## Citation and archival DOI

Use the `CITATION.cff` metadata for software citation. **Before creating the first GitHub Release or triggering Zenodo archiving, replace its provisional group-author entry with the correct, approved researcher and code contributor names.** No Zenodo DOI has been assigned yet; add the registered DOI and release metadata **after** successful Zenodo archiving. The original dataset DOI is not the software's DOI.

## Status

Research artefacts are publicly visible. An **archived, fully verified v1.0.0 release is pending** the owner's final privacy, contributor metadata and reproducibility checks. Do not describe a pending Zenodo record as published.

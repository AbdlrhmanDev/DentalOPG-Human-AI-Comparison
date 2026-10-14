# Derived Detection Results

These files contain anonymous per-image detection counts from the internal test evaluation.

The original panoramic radiographs are not redistributed. They are available from the cited DentalOPG-1550 source.

Run `python code/reproduce_metrics.py` from the repository root to recompute aggregate precision, recall, and micro-F1 for the five evaluators.

The stored counts support paired image-level resampling. Reproducing the exact published bootstrap percentiles also requires the original random-number settings and resampling implementation.

These aggregate counts alone do not reproduce model training or bounding-box-level matching.

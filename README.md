# Pediatric deepPGSegNet inference package

This package contains the four-level 3D U-Net definition, trained pediatric weights, NIfTI inference, and participant-level evaluation utilities for *Transfer learning enables automated pituitary gland segmentation on brain MRI in 6-year-old children*.

## Scope

The model was developed on a single institution's approximately 6-year-old cohort using 1 mm isotropic T1-weighted MRI. It has not been validated across institutions, scanners or age groups. The fixed ROI can truncate or miss the gland. Visual review is required; this is not a diagnostic device.

The program does not use a reference segmentation or reference-derived centroid. It uses the archived training-set mean voxel index `(80, 117, 90)` by default. This index is meaningful only in an acquisition and stored-array grid compatible with the study. **RAS orientation and 1 mm spacing alone do not establish anatomical correspondence.** No registration, skull stripping, resampling or automatic gland localization is supplied. Arbitrary external MRI may require a separately validated localization/preparation procedure. The explicit `--prepared-study-grid` acknowledgement prevents silent application to an unprepared scan; it is not an automated check of anatomical alignment.

Inputs must be finite 3D RAS-oriented NIfTI images with 1 mm isotropic spacing. Study images had dimensions 160×208×256, except one with 154×208×256. The full stored image is normalized using its mean and standard deviation (including background), then a 32³ ROI is extracted and zero-padded if necessary. The model produces logits, sigmoid is applied once, and the mask threshold is 0.5. No connected-component filtering is applied. Predictions are placed into the original input grid and retain its affine and spatial header information. Do not crop the head differently without validating the resulting coordinate mapping.

## Installation

Python 3.10 or newer is required. Use a virtual environment and install an appropriate PyTorch build for your platform. CPU inference is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`environment-tested.txt` records the versions used for verification. CUDA packages vary by hardware; the tested environment is not a portable lockfile.

## Run

```bash
python inference.py --input prepared_t1.nii.gz --output-dir result \
  --prepared-study-grid --device cpu
```

Outputs are `mask.nii.gz`, `probability.nii.gz`, and `summary.json`. Volume is foreground voxel count multiplied by the absolute determinant of the voxel-to-world affine. Small floating-point deviations from 1 mm³ per voxel are possible. The summary records checkpoint hashes, grid, processing settings, runtime, and basic QC flags. Empty-mask and crop-boundary flags do **not** identify every failed localization; an incorrect small mask can escape them. No claimed clinical acceptance threshold is encoded.

The default checkpoint `weights/pediatric_mean.pt` is the parameter average of the five pediatric folds and corresponds to the revised segmentation-performance comparisons. The individual fold checkpoints are also included. To reproduce the probability-averaging ensemble used for the original held-out clinical volume estimates:

```bash
python inference.py --input prepared_t1.nii.gz --output-dir result_probability_mean \
  --weights weights/pediatric_fold1.pt weights/pediatric_fold2.pt \
  weights/pediatric_fold3.pt weights/pediatric_fold4.pt weights/pediatric_fold5.pt \
  --prepared-study-grid --device cpu
```

These ensembling operations are distinct. For the original training/validation participants, clinical volumes were generated using each participant's corresponding held-out fold and label-assisted localization; the public default does not reproduce that mixed-localization clinical analysis automatically.

`--center I J K` allows an explicitly supplied voxel-space center for technical evaluation. A center derived from a reference annotation is label-assisted inference and must not be reported as fully automatic localization. A manually selected center makes the workflow semiautomatic.

## Evaluation

`metrics.py` implements volume agreement, absolute-agreement ICC(A,1), participant bootstrap intervals, and paired comparisons. `evaluate.py` compares aligned predicted/reference masks and writes case-level measurements and a summary. Reference masks are required only for evaluation, never for default inference.

```bash
python evaluate.py --manifest evaluation_manifest.csv --output-dir evaluation
```

The CSV must have `case,prediction,reference` columns. Paths are resolved relative to the manifest. Include all intended test cases, including failed predictions. Undefined boundary distance for empty masks is recorded as missing and the number of defined observations is reported. The CSV may contain sensitive identifiers: use a de-identified copy when sharing results.

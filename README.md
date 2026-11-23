# Geometric and optic-disc-referenced resizing for cross-dataset DR grading

Code for the paper *A Controlled Comparison of Geometric and Optic-Disc-Referenced
Resizing for Cross-Dataset Diabetic Retinopathy Grading*.

## Installation

Python 3.10.

```bash
pip install -r requirements.txt
```

## Data

The datasets are not included. Obtain them from their sources and set the paths in
`src/dr_grading/config.py`, or override each path with an environment variable of
the same name.

| Dataset | Use | Source |
| --- | --- | --- |
| APTOS 2019 | training | Kaggle, APTOS 2019 Blindness Detection |
| IDRiD | external test | doi:10.3390/data3030025 |
| Aizawl | external test | IEEE DataPort, doi:10.21227/c20k-b753 |
| REFUGE | optic-disc detector | REFUGE challenge |

`python -m src.dr_grading.build_idrid_flat` repackages the official IDRiD grading
set into a flat folder. The loader also reads the official layout directly.

RETFound weights are available from the RETFound authors. Pass their path with
`--retfound-ckpt` or the `RETFOUND_CKPT` environment variable.

## Usage

Train the optic-disc detector and cache the disc diameters:

```bash
python -m src.dr_grading.train_od
python -m src.dr_grading.infer_od
```

Train a grader. `--arm` is `geometric` or `physical_scale`, `--backbone` is
`retfound` or `vit_l_imagenet`, and `--fov-in-od-units` sets the canonical window
`N` for the physical arm.

```bash
python -m src.dr_grading.train --arm geometric --backbone retfound --seed 42
python -m src.dr_grading.train --arm physical_scale --backbone retfound --seed 42 \
    --fov-in-od-units 7 --od-cache artifacts/cache/od_diameters.json
```

The paper uses seeds 42, 1, 2, 3 and 4, both backbones, and `N` of 7 and 9.

Run the analyses and build the tables and figures:

```bash
python -m src.dr_grading.od_fov_analysis
python -m src.dr_grading.eval_od_detector
python -m src.dr_grading.window_coverage
python -m src.dr_grading.jitter_matrix
python -m src.dr_grading.final_analysis
python -m src.dr_grading.make_result_tables
python -m src.dr_grading.make_pipeline_figure
python -m src.dr_grading.make_method_figures
python -m src.dr_grading.make_resolution_figure
python -m src.dr_grading.make_confusion_figure
```

Outputs are written to `artifacts/`.

## License

MIT, see `LICENSE`.

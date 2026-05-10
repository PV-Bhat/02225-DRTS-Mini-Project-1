# MiniProject1 Artifact Reproduction

## Run code
```bash
cd code
python generate_harmonic_tasksets.py --out Task-sets-harmonic --sets-per-util 20 --tasks 10 --seed 42
python analysis.py --task-sets-dir Task-sets-harmonic --no-plots --runs 50 > harmonic_analysis.log
```

## Build Overleaf bundle
```bash
python bundle_artifact.py
```
Produces `Group1_MiniProject1.zip` with `main.tex`, `figures/`, `code/`, and this README.


## One-command build
```bash
python build_submission_artifacts.py
```
This generates `figures/*.png` and `Group1_MiniProject1.zip` locally.

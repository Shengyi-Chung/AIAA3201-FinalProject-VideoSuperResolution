# AIAA3201 Project 1 - Part 3: Video Super-Resolution Exploration

This repository contains the cleaned Part 3 code and selected final results for the AIAA3201 Video Super-Resolution project.

Part 3 explores optimization and extension directions beyond the Part 1 classical baselines and the Part 2 BasicVSR/VSRGAN pipeline.

## Repository Structure

```text
Part3_github_clean/
├── code/                  # Part 3 experiment and analysis scripts
├── figures/               # Selected final figures and qualitative panels
├── results/               # Lightweight CSV/TXT result summaries
├── logs_selected/         # Selected important run logs
├── raw_experiment_index/  # Index of raw experiment folders
├── requirements_part3.txt
└── README.md
```

## Overview

Part 3 contains three exploration directions:

1. **Direction C: Reliability-Guided Fusion**
   - Main effective reconstruction strategy.
   - Designs a reliability-guided FUGR/FUGR-C style fusion pipeline.

2. **Direction B: Generative Prior / ControlNet Analysis**
   - Studies Stable Diffusion / ControlNet-Tile style generative priors.
   - Shows that direct generative enhancement can damage fidelity, while constrained frequency-based fusion gives small but interpretable gains.

3. **Direction A: Temporal Refinement**
   - First tests post-hoc temporal refinement.
   - Then introduces train-time temporal fine-tuning with a lightweight temporal refiner.

## Direction C: Reliability-Guided Fusion

Direction C is the main effective reconstruction direction. It evaluates reliability-guided fusion and FUGR-C style refinement.

### Main Scripts

```bash
cd code

python part3_fugr_vsr.py
python part3_fugr_vsr_final_noT.py
python eval_fugr_fullval.py
python eval_fugr_mask_ablation.py
python make_fugr_qual_panels.py
python benchmark_fugr_runtime.py
```

### Important Result Files

```text
results/DirectionC_reliability_fusion/final_C/
results/DirectionC_reliability_fusion/fugr_final_best/
figures/DirectionC/
```

### Main Finding

Reliability-guided fusion provides the strongest Part 3 reconstruction baseline.

## Direction B: Generative Prior / ControlNet Analysis

Direction B explores Stable Diffusion / ControlNet-Tile style generative priors and frequency-constrained fusion.

### Main Scripts

```bash
cd code

python directionB_controlnet_tile_experiment.py
python directionB_frequency_fusion.py
python directionB_calib_test_frequency.py
python directionB_final_deep_analysis.py
python directionB_b19_zoom_panels.py
```

### Important Result Files

```text
results/DirectionB_generative_prior/
figures/DirectionB/
```

### Main Finding

Direct ControlNet enhancement can damage fidelity and temporal consistency. Constrained frequency-based high-frequency fusion provides small but interpretable gains.

## Direction A: Temporal Refinement

Direction A first tests the boundary of post-hoc temporal refinement and then introduces train-time temporal fine-tuning.

### Main Scripts

```bash
cd code

python directionA7_diagnostic_oracle.py
python directionA8_adaptive_policy.py
python directionA9_learned_selector.py
python directionA10_calibrated_ensemble.py
python directionA11_train_temporal_refiner.py
```

### Important Result Files

```text
results/DirectionA_temporal_learning/
figures/DirectionA/
```

### Main Finding

Post-hoc temporal refinement after FUGR-C has limited headroom. Train-time temporal fine-tuning with a lightweight temporal refiner gives stable positive improvements.

## Environment

Install core dependencies:

```bash
pip install -r requirements_part3.txt
```

Core packages include:

```text
numpy
opencv-python-headless
scikit-image
matplotlib
tqdm
torch
torchvision
torchaudio
Pillow
diffusers
transformers
accelerate
safetensors
```

For Direction B diffusion / ControlNet experiments, make sure that Hugging Face model access and cache paths are configured properly.

## Data and Weights

Large datasets, pretrained model weights, generated frame folders, and videos are not included in this GitHub repository.

Expected inputs are generated from the Part 1 / Part 2 / Part 3 pipelines. Full processed videos are submitted separately as `videos.zip`.

This repository intentionally excludes:

```text
*.pth
*.pt
*.ckpt
*.safetensors
*.mp4
raw frame folders
full raw experiment folders
```

## Results and Figures

Selected lightweight result summaries are included under:

```text
results/
```

Selected final figures and qualitative panels are included under:

```text
figures/
```

The full raw experiment folders are not included to keep the repository clean. Directory indexes are provided under:

```text
raw_experiment_index/
```

## Suggested Review Path

For a quick review, start with:

```text
results/DirectionC_reliability_fusion/final_C/fullval_allframes_summary.txt
results/DirectionB_generative_prior/README_Direction_B_final.md
results/DirectionA_temporal_learning/A_final_summary.txt
figures/
```

## Notes

This repository is intended as the clean Part 3 code and selected result repository for review and reproducibility.

---

## Updated Main Scripts

The current Part 3 code is organized under `part3/scripts/` by direction:

```bash
cd part3/scripts

# Direction C: Reliability-guided fusion
python direction_c_main_fugr/part3_fugr_vsr.py
python direction_c_main_fugr/part3_fugr_vsr_final_noT.py
python direction_c_main_fugr/eval_fugr_fullval.py
python direction_c_main_fugr/eval_fugr_mask_ablation.py
python direction_c_main_fugr/make_fugr_qual_panels.py

# Direction B: Generative prior and constrained frequency fusion
python direction_b_generative_prior/directionB_frequency_fusion.py
python direction_b_generative_prior/directionB_calib_test_frequency.py

# Direction A: Temporal refinement
python direction_a_temporal_refinement/directionA11_train_temporal_refiner.py
```

Supplementary LPIPS/FID/tLPIPS-style metrics for the report are stored under:

```text
part3/report_results/extra_metrics/lpips_fid_tlpips_selected8.csv
```

# AIAA3201 Final Project - Video Super-Resolution

This repository contains the code and selected results for the AIAA3201 final project on video super-resolution (VSR). The project is organized into three parts:

- **Part 1:** classical baselines and a lightweight SRCNN pipeline
- **Part 2:** BasicVSR-style video super-resolution with GAN fine-tuning
- **Part 3:** reliability-guided fusion, generative-prior analysis, and temporal refinement

The final report studies a full VSR pipeline from hand-crafted baselines to recurrent reconstruction and reliability-guided fusion.

---

## Repository Layout

```text
Part1/                  # Bicubic/Lanczos/SRCNN baselines and temporal averaging
Part2/                  # BasicVSR, VSRGAN, SpyNet, inference, evaluation, weights
part3/                  # Part 3 exploration code, selected results, figures, and logs
requirements.txt        # Unified dependency list
README.md               # Project-level usage guide
```

Large generated frames and processed videos are not tracked in GitHub. The processed videos are submitted separately to Canvas as `videos.zip`.

---

## Environment Setup

Install the shared dependencies from the project root:

```bash
pip install -r requirements.txt
```

The unified dependency list covers the core Part 1, Part 2, and Part 3 code. Part 3 also uses additional packages for perceptual metrics and generative-prior experiments, including:

```text
lpips
torchmetrics
torch-fidelity
diffusers
transformers
accelerate
safetensors
```

If your environment already contains PyTorch and CUDA, make sure the installed PyTorch version matches your CUDA driver.

---

## Model Weights

Part 2 model weights are required for inference and evaluation. In the project repository, the large `.pth` checkpoints may be tracked with Git LFS. After cloning the repository, run the following commands if Git LFS is enabled:

```bash
git lfs install
git lfs pull
```

The expected weight files are:

```text
Part2/weights/basicvsr_stage1.pth
Part2/weights/basicvsr_gan.pth
Part2/weights/spynet.pth
Part2/weights/vgg19-dcbb9e9d.pth
```

If these files are only around 100 bytes after cloning, Git LFS has not pulled the real files yet. Run `git lfs pull` again, or manually place the provided checkpoint files into `Part2/weights/` before inference or evaluation.

---

## Data

The project assumes Vimeo/REDS-style validation data with paired LR and HR frames. In our experiments, the validation data follows a structure similar to:

```text
val/
├── val_sharp/
│   ├── 000/
│   ├── 001/
│   └── ...
└── val_sharp_bicubic/
    └── X4/
        ├── 000/
        ├── 001/
        └── ...
```

For local runs, update the dataset root paths in the command-line arguments or scripts according to your machine.

---

## Part 1: Classical and SRCNN Baselines

Part 1 provides spatial and simple temporal baselines:

- Bicubic interpolation
- Lanczos interpolation
- SRCNN
- weighted multi-frame temporal averaging
- temporal averaging with unsharp masking

Main entry points:

```bash
cd Part1
python train_srcnn.py
python eval.py
python temporal_baseline.py
python infer_srcnn.py
bash run_part1_report.sh
```

Important outputs:

```text
Part1/checkpoints/srcnn_best.pt
Part1/results_part1.csv
Part1/temporal_metrics.csv
```

The Part 1 result table used in the report is stored in:

```text
Part1/results_part1.csv
```

---

## Part 2: BasicVSR and VSRGAN

Part 2 implements a BasicVSR-style recurrent video super-resolution pipeline with SpyNet alignment and a GAN fine-tuning stage.

Main components:

```text
Part2/model_basicvsr.py
Part2/model_spynet.py
Part2/model_discriminator.py
Part2/loss_gan.py
Part2/vsr_dataset.py
Part2/train_vsr_stage1.py
Part2/train_vsr_gan.py
Part2/eval_vsr.py
Part2/inference_vsr.py
Part2/frames_to_video.py
```

Typical workflow:

```bash
cd Part2

# Stage 1 fidelity-oriented training
python train_vsr_stage1.py

# Stage 2 perceptual/GAN fine-tuning
python train_vsr_gan.py

# Evaluation
python eval_vsr.py

# Inference
python inference_vsr.py
```

Expected outputs include:

```text
Part2/weights/
Part2/evalresult/
Part2/visual/
```

Generated output frames such as `Part2/basicvsr_result/` and `Part2/vsrgan_result/` are not tracked in GitHub because they are large.

---

## Part 3: Reliability-Guided Fusion and Extensions

Part 3 contains the project extensions and analysis. It is organized by direction:

```text
part3/scripts/direction_a_temporal_refinement/
part3/scripts/direction_b_generative_prior/
part3/scripts/direction_c_main_fugr/
```

### Direction A: Temporal Refinement

This direction studies post-hoc temporal filtering and train-time temporal fine-tuning.

Example entry point:

```bash
cd part3/scripts
python direction_a_temporal_refinement/directionA11_train_temporal_refiner.py
```

### Direction B: Generative Prior and Frequency-Constrained Fusion

This direction evaluates ControlNet-Tile / generative-prior enhancement and constrained high-frequency fusion.

Example entry points:

```bash
cd part3/scripts
python direction_b_generative_prior/directionB_frequency_fusion.py
python direction_b_generative_prior/directionB_calib_test_frequency.py
```

### Direction C: Reliability-Guided Fusion

This is the main Part 3 direction. It fuses BasicVSR and sharper GAN/generative cues using reliability masks and high-frequency residuals.

Example entry points:

```bash
cd part3/scripts
python direction_c_main_fugr/part3_fugr_vsr.py
python direction_c_main_fugr/part3_fugr_vsr_final_noT.py
python direction_c_main_fugr/eval_fugr_fullval.py
python direction_c_main_fugr/eval_fugr_mask_ablation.py
python direction_c_main_fugr/make_fugr_qual_panels.py
```

---

## Supplementary Metrics

Supplementary LPIPS, FID, and tLPIPS-style metrics are evaluated on representative validation sequences `000--007`. The CSV used in the report is available at:

```text
part3/report_results/extra_metrics/lpips_fid_tlpips_selected8.csv
```

Selected FUGR summary files are provided at:

```text
part3/report_results/fugr_selected_summaries/
```

The metric script is:

```text
part3/eval_required_extra_metrics.py
```

Example command:

```bash
python part3/eval_required_extra_metrics.py   --gt_root /path/to/val/val_sharp   --seqs 000 001 002 003 004 005 006 007   --methods     BasicVSR=/path/to/Part2/basicvsr_result     VSRGAN=/path/to/Part2/vsrgan_result     FUGR=/path/to/part3/results/final_fugr_frames   --out_csv part3/report_results/extra_metrics/lpips_fid_tlpips_selected8.csv
```

---

## Selected Results

The final report includes:

- Part 1 spatial and temporal baselines
- Part 2 BasicVSR vs. VSRGAN comparison
- Part 3 Direction C full-validation FUGR results
- Direction C mask ablation and sequence-level delta analysis
- Direction B generative-prior and frequency-fusion analysis
- Direction A temporal-refiner analysis
- supplementary LPIPS/FID/tLPIPS-style metrics
- qualitative comparison panels and appendix figures

Selected lightweight result files are tracked under:

```text
part3/report_results/
```

Large generated frames, raw videos, and processed demo videos are not tracked in GitHub. Processed videos are submitted separately as `videos.zip`.

---

## Suggested Reading Order

For the fastest understanding of the project:

1. Read `Part1/` for classical and SRCNN baselines.
2. Read `Part2/` for the BasicVSR + VSRGAN implementation.
3. Read `part3/scripts/direction_c_main_fugr/` for the final reliability-guided fusion method.
4. Read `part3/scripts/direction_a_temporal_refinement/` and `part3/scripts/direction_b_generative_prior/` for additional extension experiments.
5. Check `part3/report_results/` for the lightweight result files used in the report.

---

## Notes

- The repository is structured so that each part can be inspected and run independently.
- Paths may need to be adjusted depending on the local dataset and storage layout.
- Model weights are handled with Git LFS.
- Generated videos and large output-frame folders are excluded from GitHub and submitted separately when required.

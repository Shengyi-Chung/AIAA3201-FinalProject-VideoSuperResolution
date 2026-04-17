# Part1 Baseline Guide

This folder implements Part1 baseline for video super-resolution:
- Spatial baseline: Bicubic / Lanczos / SRCNN
- Temporal baseline: Weighted multi-frame average + Unsharp mask

## Environment

Use the environment created on 1T storage:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
export CONDARC=/home/schung760/my_storage_1T/condarc_aiaa3201.yaml
export TMPDIR=/home/schung760/my_storage_1T/tmp
export PIP_CACHE_DIR=/home/schung760/my_storage_1T/.cache/pip
conda activate /home/schung760/my_storage_1T/envs/aiaa3201-sr
cd /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1
```

## 1) Train SRCNN

Quick run on subset (recommended first):

```bash
python train_srcnn.py \
  --project1-root /home/schung760/shared_data/project1 \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --patch-size 33 \
  --label-size 21 \
  --stride 14 \
  --max-train-pairs 200 \
  --max-val-pairs 80 \
  --max-train-patches 4000 \
  --max-val-patches 800 \
  --save-dir /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/checkpoints
```

## 2) Evaluate Spatial Baselines + SRCNN

```bash
python eval.py \
  --project1-root /home/schung760/shared_data/project1 \
  --split val \
  --max-pairs 200 \
  --srcnn-ckpt /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/checkpoints/srcnn_best.pt \
  --csv-out /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/results_part1.csv
```

## 3) Evaluate Temporal Baseline

```bash
python temporal_baseline.py \
  --test-root /home/schung760/shared_data/project1/vimeo_super_resolution_test \
  --weights 0.25 0.5 0.25 \
  --csv-out /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/temporal_metrics.csv
```

## Output Files

- `checkpoints/srcnn_best.pt`
- `results_part1.csv`
- `temporal_metrics.csv`

## One-Command Report Workflow

Run the full report-oriented pipeline with one command:

```bash
bash /home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1/run_part1_report.sh
```

This will generate:

- `checkpoints_report/srcnn_best.pt`
- `results_part1_report.csv`
- `temporal_metrics_report.csv`
- `outputs_srcnn_report/` (input/srcnn/gt/panel images)
- `temporal_preview_report/` (temporal preview images)
- `report_summary.md` (ready-to-use markdown summary)

## Notes

- `train_sharp_bicubic/X4` in this dataset may be lower resolution than HR. The loader upsamples LR to HR size before patch extraction.
- If GPU is unavailable due old driver, training runs on CPU automatically.

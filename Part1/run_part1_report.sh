#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/schung760/my_storage_1T/AIAA3201-FinalProject-VideoSuperResolution/Part1"
PROJECT1_ROOT="/home/schung760/shared_data/project1"
ENV_PREFIX="/home/schung760/my_storage_1T/envs/aiaa3201-sr"

source /opt/miniconda3/etc/profile.d/conda.sh
export CONDARC="/home/schung760/my_storage_1T/condarc_aiaa3201.yaml"
export TMPDIR="/home/schung760/my_storage_1T/tmp"
export PIP_CACHE_DIR="/home/schung760/my_storage_1T/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

mkdir -p "$ROOT/checkpoints_report"

echo "[1/4] Train SRCNN on report subset"
conda run -p "$ENV_PREFIX" python -u "$ROOT/train_srcnn.py" \
  --project1-root "$PROJECT1_ROOT" \
  --epochs 3 \
  --batch-size 8 \
  --num-workers 0 \
  --patch-size 33 \
  --label-size 21 \
  --stride 14 \
  --max-train-pairs 80 \
  --max-val-pairs 20 \
  --max-train-patches 4000 \
  --max-val-patches 800 \
  --save-dir "$ROOT/checkpoints_report"

echo "[2/4] Evaluate spatial baselines + SRCNN"
conda run -p "$ENV_PREFIX" python "$ROOT/eval.py" \
  --project1-root "$PROJECT1_ROOT" \
  --split val \
  --max-pairs 120 \
  --srcnn-ckpt "$ROOT/checkpoints_report/srcnn_best.pt" \
  --csv-out "$ROOT/results_part1_report.csv"

echo "[3/4] Export SRCNN visual previews"
conda run -p "$ENV_PREFIX" python "$ROOT/infer_srcnn.py" \
  --project1-root "$PROJECT1_ROOT" \
  --split val \
  --srcnn-ckpt "$ROOT/checkpoints_report/srcnn_best.pt" \
  --max-pairs 12 \
  --out-dir "$ROOT/outputs_srcnn_report"

echo "[4/4] Run temporal baseline + preview"
conda run -p "$ENV_PREFIX" python "$ROOT/temporal_baseline.py" \
  --test-root "$PROJECT1_ROOT/vimeo_super_resolution_test" \
  --weights 0.25 0.5 0.25 \
  --save-preview \
  --preview-dir "$ROOT/temporal_preview_report" \
  --csv-out "$ROOT/temporal_metrics_report.csv"

echo "Generate report summary"
conda run -p "$ENV_PREFIX" python "$ROOT/report_summary.py" \
  --spatial-csv "$ROOT/results_part1_report.csv" \
  --temporal-csv "$ROOT/temporal_metrics_report.csv" \
  --out "$ROOT/report_summary.md"

echo "Done. Outputs:"
echo "  $ROOT/results_part1_report.csv"
echo "  $ROOT/temporal_metrics_report.csv"
echo "  $ROOT/outputs_srcnn_report"
echo "  $ROOT/temporal_preview_report"
echo "  $ROOT/report_summary.md"

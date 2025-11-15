#!/usr/bin/env bash
set -euo pipefail

# Batch Stage 1 calibration + evaluation across manifests
# Usage:
#   scripts/run_stage1_eval_batch.sh \
#     --model_path outputs/stage1/run_YYYYMMDD_HHMMSS/best_model.pth \
#     [--val_manifest manifests/faceforensics_val_balanced.csv] \
#     [--output_root outputs/stage1] \
#     [--cpu]
#
# Notes:
# - Assumes running from repo root (MobileDeepfakeDetection)
# - Uses repository manifests with paths relative to '.' (data_dir set to '.')

MODEL_PATH=""
VAL_MANIFEST="manifests/faceforensics_val_balanced.csv"
OUTPUT_ROOT="outputs/stage1"
FORCE_CPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      MODEL_PATH="$2"; shift 2 ;;
    --val_manifest)
      VAL_MANIFEST="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --cpu)
      FORCE_CPU=1; shift ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: --model_path is required" >&2
  exit 1
fi

if [[ ${FORCE_CPU} -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "Info: Forcing CPU by clearing CUDA_VISIBLE_DEVICES"
fi

echo "== Stage 1: Calibration =="
python src/stage1/calibrate_model.py \
  --model_path "${MODEL_PATH}" \
  --data_dir . \
  --val_manifest "${VAL_MANIFEST}" \
  --output_dir "${OUTPUT_ROOT}" \
  --num_workers 0 \
  --batch_size 32

CAL_FILE="${OUTPUT_ROOT}/calibration_temp.json"
if [[ ! -f "${CAL_FILE}" ]]; then
  echo "Error: Calibration file not found: ${CAL_FILE}" >&2
  exit 1
fi

declare -A TESTS
TESTS[faceforensics]="manifests/faceforensics_test_balanced.csv"
TESTS[celebdf_v2]="manifests/celebdf_v2_test_balanced.csv"
TESTS[dfdc]="manifests/dfdc_test_balanced.csv"
TESTS[deeperforensics]="manifests/deeperforensics_test_balanced.csv"

echo "== Stage 1: Evaluation across datasets =="
for ds in "${!TESTS[@]}"; do
  outdir="${OUTPUT_ROOT}/evaluation/${ds}"
  mkdir -p "${outdir}"
  echo "-- Evaluating ${ds} -> ${outdir}"
  python src/stage1/evaluate_stage1.py \
    --model_path "${MODEL_PATH}" \
    --data_dir . \
    --test_manifest "${TESTS[$ds]}" \
    --use_calibration \
    --calibration_file "${CAL_FILE}" \
    --output_dir "${outdir}" \
    --num_workers 0 \
    --batch_size 64
done

echo "== Done. Results under: ${OUTPUT_ROOT}/evaluation/{faceforensics,celebdf_v2,dfdc,deeperforensics} =="


#!/bin/bash
set -euo pipefail

# ---- config ----
GPUS=(0 1 2 3)                       # 4 concurrent jobs max
DATASETS=(train_dataset_{1..8})
ENCODING_TYPE=tcr_bert
BASE=/oak/stanford/groups/akundaje/abuen/kaggle/AttentionDeepMIL
PY=$BASE/main.py
LOGDIR=$BASE/logs
mkdir -p "$LOGDIR"

# FIFO GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"
for g in "${GPUS[@]}"; do echo "$g" >&3; done

for DATASET in "${DATASETS[@]}"; do
  read -r gpu <&3  # blocks until a GPU is free

  {
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/mil_${ENCODING_TYPE}_${DATASET}_${ts}.log"
    CKPT_PATH="$BASE/models/${DATASET}/${ENCODING_TYPE}_best.pt"

    echo "[$(date +%T)] start $DATASET on GPU $gpu -> $log"

    CUDA_VISIBLE_DEVICES="$gpu" python "$PY" \
      --dataset_name "$DATASET" \
      --encoding_type "$ENCODING_TYPE" \
      --model attention \
      --ckpt_path "$CKPT_PATH" \
      --max_instances 4096 \
      --mc_samples 1 \
      --epochs 100 \
      >"$log" 2>&1

    status=$?
    echo "$gpu" >&3  # return GPU token
    echo "[$(date +%T)] done  $DATASET on GPU $gpu (exit $status) | log: $log"
    exit $status
  } &
done

wait
echo "All datasets finished."

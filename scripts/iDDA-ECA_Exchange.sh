#!/usr/bin/env bash
set -euo pipefail

# ============================================
# iDDA-Cast Experiment
# Dataset: Exchange
# Model: iDDA-ECA
# ============================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="iDDA-ECA"
DATASET="Exchange"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IDDA_AGGREGATION="${MODEL#iDDA-}"

EPOCHS="${IDDA_EPOCHS:-5}"
LR="${IDDA_LR:-0.0001}"
BATCH_SIZE=8

LLM_MODEL=GPT2
LLM_LAYERS=6
LLM_DIM=768

SEQ_LEN=512
LABEL_LEN=48
AR_LEN=96

D_MODEL=32
D_FF=128

PATCH_LEN=16
STRIDE=8
SEASON_PATCH=12
SEASON_STRIDE=6
PERIOD=24
TARGET="${IDDA_TARGET:-OT}"

if (( $# > 0 )); then
    PREDS=("$@")
else
    PREDS=(96 192 336)
fi

case "$DATASET" in
    ETTh1|ETTh2)
        DATA="$DATASET"
        ROOT_PATH="./dataset/ETT-small/"
        DATA_PATH="${DATASET}.csv"
        ENC_IN=7
        FREQ=h
        ;;

    ETTm1|ETTm2)
        DATA="$DATASET"
        ROOT_PATH="./dataset/ETT-small/"
        DATA_PATH="${DATASET}.csv"
        ENC_IN=7
        FREQ=t
        PERIOD=96
        ;;

    Weather)
        DATA=Weather
        ROOT_PATH="./dataset/weather/"
        DATA_PATH=weather.csv
        ENC_IN=21
        FREQ=h
        ;;

    Exchange)
        DATA=Exchange
        ROOT_PATH="./dataset/exchange_rate/"
        DATA_PATH=exchange_rate.csv
        ENC_IN=8
        FREQ=d
        PERIOD=7
        SEASON_PATCH=7
        SEASON_STRIDE=7
        ;;

    Wind)
        DATA=Wind
        : "${WIND_CSV:?Set WIND_CSV to the actual CSV path}"
        ROOT_PATH="$(dirname "$WIND_CSV")/"
        DATA_PATH="$(basename "$WIND_CSV")"
        ENC_IN="${WIND_ENC_IN:-7}"
        FREQ=t
        PERIOD=96
        PATCH_LEN=36
        STRIDE=12
        SEASON_PATCH=72
        SEASON_STRIDE=24
        ;;

    AQ)
        DATA="${AQ_DATA_KEY:-AQShunyi}"
        : "${AQ_CSV:?Set AQ_CSV to the actual CSV path}"
        : "${AQ_ENC_IN:?Set AQ_ENC_IN to the feature count}"
        ROOT_PATH="$(dirname "$AQ_CSV")/"
        DATA_PATH="$(basename "$AQ_CSV")"
        ENC_IN="$AQ_ENC_IN"
        FREQ=h
        PERIOD=24
        TARGET="${AQ_TARGET:-PM2.5}"
        ;;
esac

if [[ ! -f "models/${MODEL}.py" ]]; then
    echo "ERROR: models/${MODEL}.py not found"
    exit 1
fi

if [[ ! -f "$ROOT_PATH/$DATA_PATH" ]]; then
    echo "ERROR: Dataset file not found"
    echo "$ROOT_PATH/$DATA_PATH"
    exit 1
fi

mkdir -p "logs/${MODEL}/${DATASET}"

for PRED in "${PREDS[@]}"; do

    NAME="${MODEL}_${DATASET}_512_${PRED}_Ep${EPOCHS}"

    echo "====================================="
    echo "Experiment: $NAME"
    echo "====================================="

    accelerate launch \
      --mixed_precision fp16 \
      --num_processes 1 \
      --num_machines 1 \
      --dynamo_backend no \
      run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model "$MODEL" \
      --model_id "$NAME" \
      --model_comment "$NAME" \
      --data "$DATA" \
      --root_path "$ROOT_PATH" \
      --data_path "$DATA_PATH" \
      --features M \
      --target "$TARGET" \
      --freq "$FREQ" \
      --embed timeF \
      --seq_len "$SEQ_LEN" \
      --label_len "$LABEL_LEN" \
      --pred_len "$PRED" \
      --ar_len "$AR_LEN" \
      --enc_in "$ENC_IN" \
      --dec_in "$ENC_IN" \
      --c_out "$ENC_IN" \
      --llm_model "$LLM_MODEL" \
      --llm_dim "$LLM_DIM" \
      --llm_layers "$LLM_LAYERS" \
      --d_model "$D_MODEL" \
      --d_ff "$D_FF" \
      --n_heads 8 \
      --patch_len "$PATCH_LEN" \
      --stride "$STRIDE" \
      --seasonal_patch_len "$SEASON_PATCH" \
      --seasonal_stride "$SEASON_STRIDE" \
      --period "$PERIOD" \
      --batch_size "$BATCH_SIZE" \
      --learning_rate "$LR" \
      --train_epochs "$EPOCHS" \
      --prompt_domain 1 \
      --component_eval 0 \
      --num_workers 0 \
      2>&1 | tee "logs/${MODEL}/${DATASET}/${NAME}.log"

done
#!/bin/bash
# Run the unadapted baseline, TTRL training, or evaluation of an exported HF model.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:-}"
if [[ "$MODE" != baseline && "$MODE" != train && "$MODE" != evaluate ]]; then
  echo "Usage: bash $0 {baseline|train|evaluate} [Hydra overrides...]" >&2
  exit 2
fi
shift

if [[ "$MODE" == evaluate && -z "${MODEL_PATH:-}" ]]; then
  echo "For evaluate, set MODEL_PATH to the exported Hugging Face model directory." >&2
  exit 2
fi

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}"
export DATA_LOCAL_DIR="${DATA_LOCAL_DIR:-${VERL_ROOT}/data}"
N_GPUS="${N_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_ROOT}/outputs/math500/${MODE}-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
export OUTPUT_DIR

for split in train test; do
  if [[ ! -f "${DATA_LOCAL_DIR}/MATH-TTT/${split}.parquet" ]]; then
    echo "Missing ${DATA_LOCAL_DIR}/MATH-TTT/${split}.parquet" >&2
    echo "Run: python ${VERL_ROOT}/data/preprocess.py --data-source MATH-TTT --data-dir ${DATA_LOCAL_DIR}" >&2
    exit 2
  fi
done

mode_overrides=()
if [[ "$MODE" == train ]]; then
  mode_overrides+=(trainer.val_only=False ttrl.enable=True)
else
  mode_overrides+=(trainer.val_only=True ttrl.enable=False)
fi

echo "Mode: $MODE"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# Forward the recipe unchanged except logging/checkpoint controls and explicit evaluation mode.
# The default eight-GPU topology matches the authors' recipe.
bash "${SCRIPT_DIR}/math.sh" \
  "trainer.n_gpus_per_node=${N_GPUS}" \
  'trainer.logger=[console]' \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=3 \
  "trainer.default_local_dir='${OUTPUT_DIR}'" \
  "trainer.validation_data_dir='${OUTPUT_DIR}/validation'" \
  "${mode_overrides[@]}" "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"

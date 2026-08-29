#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_DIR}" /workspace/hf-cache /workspace/car360-output

BASE="${MODEL_DIR}/Wan2.2-I2V-A14B"
LORA="${MODEL_DIR}/Wan2.2-frames-to-video"

echo "=========================================================="
echo " CAR360 RunPod"
echo " Web UI: port 8000"
echo " Models: ${MODEL_DIR}"
echo "=========================================================="

if [ ! -f "${BASE}/config.json" ] && [ ! -d "${BASE}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
  echo "[setup] Downloading Wan2.2 I2V A14B model..."
  huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    --local-dir "${BASE}"
else
  echo "[setup] Wan2.2 base model already present."
fi

if [ ! -f "${LORA}/lora_interpolation_high_noise_final.safetensors" ]; then
  echo "[setup] Downloading Morphic Frames-to-Video LoRA..."
  huggingface-cli download morphic/Wan2.2-frames-to-video \
    --local-dir "${LORA}"
else
  echo "[setup] Morphic LoRA already present."
fi

# Morphic's README sometimes calls the base folder "...-Interpolation".
# Keep a compatibility symlink so either name works.
ln -sfn "${BASE}" "${MODEL_DIR}/Wan2.2-I2V-A14B-Interpolation"

echo "[setup] Starting Car360 web UI..."
exec uvicorn main:app --host 0.0.0.0 --port 8000

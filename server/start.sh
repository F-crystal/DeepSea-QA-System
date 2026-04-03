#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# DeepSea QA - Universal start.sh
# - Auto detect GPU architecture
# - Auto select proper PyTorch build
# - Compatible with RTX 5090 / V100 / A100 / 3090
# =====================================================

PORT=80
APP_MODULE="app:app"

# -----------------------------
# 0) 代码根目录
# -----------------------------
if [ -n "${DEEPSEA_QA_ROOT:-}" ]; then
  ROOT_DIR="${DEEPSEA_QA_ROOT}"
else
  ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

cd "${ROOT_DIR}"
REQ_FILE="${ROOT_DIR}/requirements.txt"

# -----------------------------
# 1) 数据路径
# -----------------------------
export KB_DIR="${KB_DIR:-/openbayes/home/deepsea_qa/artifacts/kb}"
export EMB_MODEL_DIR="${EMB_MODEL_DIR:-/openbayes/home/bge-m3}"
export BERTSCORE_MODEL_DIR="${BERTSCORE_MODEL_DIR:-/openbayes/home/bert-base-chinese}"

# -----------------------------
# 2) HuggingFace 离线模式
# -----------------------------
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export HF_HOME="${HF_HOME:-/openbayes/home/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/openbayes/home/.cache/pip}"

# -----------------------------
# 3) pip 源
# -----------------------------
TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI="https://pypi.org/simple"

# -----------------------------
# 4) GPU 信息
# -----------------------------
echo "========== GPU INFO =========="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
else
    GPU_NAME="CPU"
fi
echo "[INFO] GPU detected: ${GPU_NAME}"
echo "=============================="

# -----------------------------
# 5) 选择 PyTorch wheel
# -----------------------------
TORCH_INDEX=""
TORCH_PRE=""

if [[ "${GPU_NAME}" == *"5090"* ]]; then
    echo "[INFO] Detected Blackwell GPU (RTX 5090)"
    TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
    TORCH_PRE="--pre"

elif [[ "${GPU_NAME}" == *"V100"* ]]; then
    echo "[INFO] Detected Tesla V100"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"

elif [[ "${GPU_NAME}" == *"A100"* ]]; then
    echo "[INFO] Detected A100"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"

elif [[ "${GPU_NAME}" == *"3090"* ]]; then
    echo "[INFO] Detected RTX 3090"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"

else
    echo "[WARN] Unknown GPU, fallback to cu121"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

echo "[INFO] Using PyTorch index: ${TORCH_INDEX}"

# -----------------------------
# 6) Fail Fast
# -----------------------------
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] REQ_FILE=${REQ_FILE}"

test -f "${REQ_FILE}" || (echo "[ERROR] requirements.txt missing" && exit 1)
test -f "${ROOT_DIR}/app.py" || (echo "[ERROR] app.py missing" && exit 1)
test -d "${KB_DIR}" || (echo "[ERROR] KB_DIR missing" && exit 1)
test -d "${EMB_MODEL_DIR}" || (echo "[ERROR] EMB_MODEL_DIR missing" && exit 1)

# -----------------------------
# 7) Python 信息
# -----------------------------
python -V
python -m pip -V

# -----------------------------
# 8) 升级 pip
# -----------------------------
echo "[INFO] Upgrading pip"
python -m pip install --upgrade pip -i "${PYPI}" --cache-dir "${PIP_CACHE_DIR}"

# -----------------------------
# 9) 安装 PyTorch
# -----------------------------
echo "[INFO] Installing PyTorch"

python -m pip install ${TORCH_PRE} torch torchvision torchaudio \
    --index-url "${TORCH_INDEX}" \
    --cache-dir "${PIP_CACHE_DIR}"

# -----------------------------
# 10) 安装 requirements
# -----------------------------
echo "[INFO] Installing requirements (Tsinghua mirror)"

if ! python -m pip install -r "${REQ_FILE}" \
    -i "${TSINGHUA}" \
    --cache-dir "${PIP_CACHE_DIR}"; then

    echo "[WARN] Tsinghua mirror failed, fallback to official PyPI"

    python -m pip install -r "${REQ_FILE}" \
        -i "${PYPI}" \
        --cache-dir "${PIP_CACHE_DIR}"
fi

# -----------------------------
# 11) 依赖检查
# -----------------------------
python - << 'PY'
import os
import torch

print("==== Runtime Check ====")
print("TRANSFORMERS_OFFLINE:", os.environ.get("TRANSFORMERS_OFFLINE"))
print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

import transformers
import sentence_transformers
import faiss

print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("faiss ok")

print("======================")
PY

# -----------------------------
# 12) 启动服务
# -----------------------------
echo "[INFO] Starting uvicorn..."

exec python -m uvicorn "${APP_MODULE}" \
    --host 0.0.0.0 \
    --port "${PORT}"
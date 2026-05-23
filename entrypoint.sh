#!/bin/bash
set -e

echo "Starting Speaker Diarization API..."
echo "INFERENCE_DEVICE env: ${INFERENCE_DEVICE:-auto}"
python - <<'PY'
import torch

cuda_available = torch.cuda.is_available()
selected = "cuda" if cuda_available else "cpu"

print(f"[startup] torch_version={torch.__version__}")
print(f"[startup] cuda_available={cuda_available}")
print(f"[startup] selected_runtime_device={selected}")
print(f"[startup] cuda_runtime_version={torch.version.cuda}")

if cuda_available:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    print(f"[startup] gpu_index={idx}")
    print(f"[startup] gpu_name={props.name}")
    print(f"[startup] gpu_total_memory_gb={round(props.total_memory / (1024 ** 3), 2)}")
else:
    print("[startup] gpu_name=not_available")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[startup] nvidia-smi detected, printing GPU view:"
  nvidia-smi || true
else
  echo "[startup] nvidia-smi not found in container"
fi

# Run the FastAPI application
exec uvicorn app:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 1200
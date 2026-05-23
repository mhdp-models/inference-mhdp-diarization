# Use prebuilt CUDA + PyTorch runtime image aligned with pyannote.audio deps
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Print base runtime versions during build for traceability in CI logs
RUN python - <<'PY'
import sys
import torch
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version in base image: {torch.__version__}")
print(f"CUDA available at build time: {torch.cuda.is_available()}")
print(f"CUDA runtime version in torch: {torch.version.cuda}")
PY

# Copy and install Python dependencies with transitive deps included
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps torchaudio==2.8.0 torchvision==0.23.0 \
    && python - <<'PY'
import torch
import torchaudio
import torchvision

print(f"Runtime torch version: {torch.__version__}")
print(f"Runtime torchaudio version: {torchaudio.__version__}")
print(f"Runtime torchvision version: {torchvision.__version__}")
print(f"Runtime CUDA available: {torch.cuda.is_available()}")
print(f"Runtime CUDA version: {torch.version.cuda}")

# This triggers extension loading and fails build if torchvision ops are ABI-mismatched.
print(f"Torchvision compiled ops available: {torchvision.extension._has_ops()}")
PY

# Copy application files
COPY app.py .
COPY model.py .
COPY gender_app.py .
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
ENTRYPOINT ["/entrypoint.sh"]

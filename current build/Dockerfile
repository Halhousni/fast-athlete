# ============================================================
# FAST ATHLETE - runtime image
# ============================================================
# Changes from original:
#   1. Base image includes the GPU runtime
#   2. GPU compute support installed

# ============================================================

# --- Stage 1: build wheels for compiled packages ---
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    gpu-runtime-dev \
    gpu-runtime-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Build all wheels
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    -r requirements.txt \
    gpu-compute

# --- Stage 2: slim runtime image ---
FROM python:3.12-slim

# ---- GPU runtime ----
# gpu-runtime-lib = loader that dispatches to the vendor driver
# gpu-driver = open-source GPU driver
# gpuinfo = diagnostic tool to verify the GPU is visible
RUN apt-get update && apt-get install -y --no-install-recommends \
    gpu-runtime-lib \
    gpu-driver \
    gpuinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from stage 1
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
    -r requirements.txt \
    gpu-compute \
    && rm -rf /wheels

# Copy application code
COPY app.py .
COPY bowen_centroid.py .
COPY .streamlit/ .streamlit/

# Expose the port
EXPOSE 8501

# Verify the GPU runtime is functional at build time (optional, won't see GPU
# during build but confirms the ICD loader is installed correctly)
RUN gpuinfo --list 2>/dev/null || echo "No GPU during build (expected)"

# Command to run the app
CMD ["python3", "app.py"]

# ============================================================
# HOW TO BUILD AND RUN
# ============================================================
#
# Build:
#   docker build -t wsst-app .
#
# Run WITH GPU access (critical — without this the container can't see the GPU):

#

# Infrastructure) device into the container, which is how the GPU talks
# to the GPU.  This is the equivalent of the --gpus flag.
#
# Verify GPU is visible inside the container:

#
# If you see your the GPU listed, you're good.
# If not, check that your host has the the GPU kernel driver
# loaded:


#
# ============================================================
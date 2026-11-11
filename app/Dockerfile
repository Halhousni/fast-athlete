# ============================================================
# FAST — runtime image
# ============================================================
# CPU only. The pipeline is one wavelet synchrosqueezed transform
# (wsstpy, pure numpy/scipy) per channel, so no GPU packages and no
# device pass-through are needed.
# ============================================================

FROM python:3.12-slim

# git is only needed to install wsstpy, which is not on PyPI yet.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FAST_DATA_DIR=/data \
    FAST_BIND=0.0.0.0 \
    FAST_PORT=8501

WORKDIR /app

# Dependencies first, so a code-only change still hits the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app.py engine.py ingest.py store.py version.py ./
COPY templates ./templates
COPY static ./static
COPY demo_recording.csv ./

# The commit stamp goes in after the dependencies, so changing it does not
# reinstall NumPy and SciPy.
ARG FAST_BUILD=local
ENV FAST_BUILD=${FAST_BUILD}

# The database lives on a volume; the app must own it, and must not run as root.
RUN useradd --system --uid 10001 --home /app fast \
 && mkdir -p /data \
 && chown -R fast:fast /data /app
USER fast

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/healthz', timeout=4).status == 200 else 1)"

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]

# ============================================================
# HOW TO BUILD AND RUN
# ============================================================
#
# Build:
#   docker build -t fast-athlete .
#
# Run:
#   docker run -p 127.0.0.1:8501:8501 -v fast_data:/data fast-athlete
#
# The app keeps its database at $FAST_DATA_DIR (default /data) and its
# uploads in memory only. Mount a volume at /data, or every saved profile
# and result disappears when the container is replaced.
#
# ============================================================

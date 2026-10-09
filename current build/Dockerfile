# ============================================================
# FAST ATHLETE - runtime image
# ============================================================
# The FAST pipeline runs on CPU. ssqueezepy is called without a backend
# argument, so it takes the numpy/scipy path, and its own dependencies are
# numba, numpy and scipy. No GPU driver packages are needed and the container
# requires no device pass-through.
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so a code-only change still hits the layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .
COPY bowen_centroid.py .
COPY demo.mp4 .
COPY demo_recording.csv .

ENV FAST_HOST=0.0.0.0
EXPOSE 8501

CMD ["python3", "app.py"]

# ============================================================
# HOW TO BUILD AND RUN
# ============================================================
#
# Build:
#   docker build -t fast-athlete .
#
# Run:
#   docker run -p 8501:8501 fast-athlete
#
# The app keeps its database at $FAST_DATA_DIR, which defaults to /data.
# Mount a volume there, or every saved profile and result disappears when
# the container is replaced.
#
# ============================================================

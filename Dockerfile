# DHM Reconstruction — headless Linux runtime
#
# Built so Sven's IT team can deploy the CLI session runner on the
# Linux + RTX 4090 box without a desktop GUI. The image installs
# the scientific deps, optional torch (with CUDA), and the test
# suite — but does NOT install Dear PyGui (headless image).
#
# Build:
#   docker build -t dhm:cli .
#
# Run a session:
#   docker run --rm -v $PWD/data:/data -v $PWD/out:/out \
#       --gpus all dhm:cli \
#       run /data/session.json --out /out --workers 4
#
# Run the bench:
#   docker run --rm --gpus all dhm:cli bench --backends default,torch \
#       --shapes 256,512,1024,2048
#
# The base CUDA image is the NVIDIA-blessed runtime. Switch to
# ``python:3.13-slim`` when targeting the CPU-only fallback box.

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DHM_USER=docker

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        libfftw3-dev libhdf5-dev libtiff5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dhm
COPY requirements.txt /opt/dhm/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
# Optional: torch with CUDA. Pin version when promoting to prod.
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch

COPY src /opt/dhm/src
COPY tests /opt/dhm/tests
COPY scripts /opt/dhm/scripts
COPY pytest.ini /opt/dhm/pytest.ini

ENV PYTHONPATH=/opt/dhm/src:/opt/dhm/tests

# Smoke test — fail-fast on a broken image.
RUN python3 -m pytest tests/test_autofocus_speed_baseline.py -q --no-header || \
    (echo "smoke test failed" && exit 1)

ENTRYPOINT ["python3", "-m", "cli.run_session"]
CMD ["--help"]

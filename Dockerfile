FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV HF_HOME=/workspace/hf-cache
ENV MODEL_DIR=/workspace/models
ENV APP_DIR=/opt/car360

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git git-lfs ffmpeg curl ca-certificates \
    build-essential ninja-build && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124

WORKDIR ${APP_DIR}

# Freeze the Morphic repository at build time so runtime isn't dependent on a git clone.
RUN git clone --depth 1 https://github.com/morphicfilms/frames-to-video.git /opt/frames-to-video && \
    python3 -m pip install -r /opt/frames-to-video/requirements.txt && \
    python3 -m pip install "huggingface_hub[cli]>=0.34.0"

COPY app/ ${APP_DIR}/
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
CMD ["/start.sh"]

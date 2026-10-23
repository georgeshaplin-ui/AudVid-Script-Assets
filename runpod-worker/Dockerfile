# syntax=docker/dockerfile:1.7
#
# WhisperX + pyannote 4 serverless worker for RunPod.
#
# cuDNN / cuBLAS come ONLY from the pip wheels that torch==2.8.0+cu128 pulls in
# (nvidia-cudnn-cu12 9.x, nvidia-cublas-cu12). The base image deliberately has no
# system cuDNN, so torch, ctranslate2 and pyannote all load the same cuDNN 9 build.
# Mixing a system cuDNN 8 with pip cuDNN 9 is what crashed the previous template.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG WHISPER_MODEL=large-v3
ARG ALIGN_LANGUAGES="en ar"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    NLTK_DATA=/models/nltk \
    WHISPER_MODEL_DIR=/models/whisper \
    WHISPER_MODEL=${WHISPER_MODEL} \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3.10-dev \
      ffmpeg libsndfile1 ca-certificates curl \
 && ln -sf /usr/bin/python3.10 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

ENV NV_LIBS=/usr/local/lib/python3.10/dist-packages/nvidia
ENV LD_LIBRARY_PATH=${NV_LIBS}/cudnn/lib:${NV_LIBS}/cublas/lib:${NV_LIBS}/cuda_runtime/lib:${NV_LIBS}/cuda_nvrtc/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN python -m pip install --upgrade "pip<25" setuptools wheel \
 && python -m pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt \
 && python -m pip install --no-deps whisperx==3.8.6 \
 && python - <<'PY'
import importlib.metadata as md
pins = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "whisperx": "3.8.6",
    "pyannote.audio": "4.0.7",
    "transformers": "5.20.2",
}
for pkg, want in pins.items():
    got = md.version(pkg)
    assert got.split("+")[0] == want, f"{pkg} drifted: {got} != {want}"
print("PINS_OK", {p: md.version(p) for p in pins})
PY

# Fail the build (not the worker) if the cuDNN 9 / cuBLAS 12 libraries cannot be resolved.
RUN python - <<'PY'
import ctypes
for lib in ("libcudnn.so.9", "libcudnn_cnn.so.9", "libcudnn_ops.so.9", "libcublas.so.12", "libcublasLt.so.12"):
    ctypes.CDLL(lib)
import ctranslate2, torch, whisperx.asr, whisperx.alignment, whisperx.diarize  # noqa: F401
from transformers import AutoModelForAudioFrameClassification, AutoProcessor  # noqa: F401
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
nemotron = [name for name in CONFIG_MAPPING_NAMES if "nemotron" in name and "diar" in name]
assert nemotron, "transformers 5.20.2 has no Nemotron diarization config"
print("LIBS_OK torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), "ct2", ctranslate2.__version__, "nemotron", nemotron)
PY

COPY download_models.py /opt/worker/download_models.py
RUN python /opt/worker/download_models.py --whisper "${WHISPER_MODEL}" --align ${ALIGN_LANGUAGES}

COPY handler.py /opt/worker/handler.py

ARG AUDVID_IMAGE_TAG=
ENV AUDVID_IMAGE_TAG=${AUDVID_IMAGE_TAG}

WORKDIR /opt/worker

CMD ["python", "-u", "handler.py"]

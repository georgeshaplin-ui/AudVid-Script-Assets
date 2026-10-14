import base64
import gc
import math
import os
import tempfile
import time
import traceback

import numpy as np
import requests
import runpod
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_MODEL_PATH = os.path.join(os.environ.get("WHISPER_MODEL_DIR", "/models/whisper"), WHISPER_MODEL)
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_MB", "500")) * 1024 * 1024
DOWNLOAD_TIMEOUT_S = int(os.environ.get("DOWNLOAD_TIMEOUT_S", "120"))
DIARIZATION_CACHE = "/runpod-volume/hf" if os.path.isdir("/runpod-volume") else None

print(f"[init] device={DEVICE} compute_type={COMPUTE_TYPE} model={WHISPER_MODEL_PATH}", flush=True)
_t0 = time.time()
ASR_MODEL = whisperx.load_model(
    WHISPER_MODEL_PATH if os.path.isdir(WHISPER_MODEL_PATH) else WHISPER_MODEL,
    DEVICE,
    compute_type=COMPUTE_TYPE,
    vad_method="pyannote",
)
print(f"[init] ASR ready in {time.time() - _t0:.1f}s", flush=True)

_align_models = {}
_diarize_pipeline = None


def _get_align_model(language):
    if language not in _align_models:
        _align_models[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
    return _align_models[language]


def _get_diarize_pipeline(token):
    global _diarize_pipeline
    if _diarize_pipeline is None:
        t0 = time.time()
        _diarize_pipeline = DiarizationPipeline(
            model_name=DIARIZATION_MODEL, token=token, device=DEVICE, cache_dir=DIARIZATION_CACHE
        )
        print(f"[diarize] pipeline ready in {time.time() - t0:.1f}s", flush=True)
    return _diarize_pipeline


def _materialize_audio(source, workdir):
    path = os.path.join(workdir, "input_audio")
    if source.startswith(("http://", "https://")):
        with requests.get(source, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
            resp.raise_for_status()
            size = 0
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise ValueError(f"audio exceeds {MAX_AUDIO_BYTES // (1024 * 1024)} MB")
                    fh.write(chunk)
        if size == 0:
            raise ValueError("downloaded audio is empty")
        return path
    if os.path.isfile(source):
        return source
    payload = source.split(",", 1)[1] if source.startswith("data:") else source
    data = base64.b64decode(payload, validate=False)
    if not data:
        raise ValueError("audio_file is neither a URL, a file path, nor valid base64")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _optional_int(job_input, key):
    value = job_input.get(key)
    return int(value) if value not in (None, "") else None


def handler(job):
    job_input = job.get("input") or {}
    source = job_input.get("audio_file") or job_input.get("audio_url") or job_input.get("audio_base64")
    if not source or not isinstance(source, str):
        return {"error": "audio_file (URL, path or base64) is required"}

    language = job_input.get("language") or None
    if language == "auto":
        language = None
    task = job_input.get("task") or "transcribe"
    if task not in ("transcribe", "translate"):
        return {"error": "task must be 'transcribe' or 'translate'"}
    batch_size = int(job_input.get("batch_size") or DEFAULT_BATCH_SIZE)
    align_output = bool(job_input.get("align_output", True))
    diarization = bool(job_input.get("diarization", False))
    hf_token = job_input.get("huggingface_access_token") or os.environ.get("HF_TOKEN")
    if diarization and not hf_token:
        return {"error": "huggingface_access_token (or HF_TOKEN env) is required for diarization"}

    timings = {}
    try:
        with tempfile.TemporaryDirectory() as workdir:
            t = time.time()
            audio = whisperx.load_audio(_materialize_audio(source, workdir))
            timings["load_audio_s"] = round(time.time() - t, 2)
        duration_s = round(len(audio) / 16000, 2)

        t = time.time()
        result = ASR_MODEL.transcribe(audio, batch_size=batch_size, language=language, task=task)
        timings["transcribe_s"] = round(time.time() - t, 2)
        detected_language = result.get("language")

        if align_output and result["segments"] and task == "transcribe":
            t = time.time()
            try:
                align_model, metadata = _get_align_model(detected_language)
                result = whisperx.align(
                    result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False
                )
            except ValueError as exc:
                print(f"[align] skipped: {exc}", flush=True)
            timings["align_s"] = round(time.time() - t, 2)

        speakers = []
        if diarization and result["segments"]:
            t = time.time()
            diarize_segments = _get_diarize_pipeline(hf_token)(
                audio,
                num_speakers=_optional_int(job_input, "num_speakers"),
                min_speakers=_optional_int(job_input, "min_speakers"),
                max_speakers=_optional_int(job_input, "max_speakers"),
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            speakers = sorted({s["speaker"] for s in result["segments"] if s.get("speaker")})
            timings["diarize_s"] = round(time.time() - t, 2)

        segments = result.get("segments", [])
        return _clean(
            {
                "segments": segments,
                "text": " ".join(s.get("text", "").strip() for s in segments).strip(),
                "detected_language": detected_language,
                "language_probability": None,
                "speakers": speakers,
                "duration_s": duration_s,
                "timings": timings,
                "model": WHISPER_MODEL,
            }
        )
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        fatal = isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc) or "cuDNN" in str(exc)
        return {"error": message, "refresh_worker": fatal}
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

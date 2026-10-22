import base64
import gc
import hashlib
import importlib.metadata
import math
import os
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
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
SAMPLE_RATE = 16000

# Maximum speaker separation for proxy audio: fast turns, pauses under a second,
# and background noise. community-1 ships at threshold 0.60 / Fb 0.80, which merges
# those voices. Lower distance and Fb keep more speakers; higher Fa trusts the
# acoustics. Values sit at the edge of the VBx search range on purpose.
DIAR_THRESHOLD = float(os.environ.get("DIARIZATION_THRESHOLD", "0.35"))
DIAR_FA = float(os.environ.get("DIARIZATION_FA", "0.50"))
DIAR_FB = float(os.environ.get("DIARIZATION_FB", "0.02"))
DIAR_MIN_DURATION_OFF = float(os.environ.get("DIARIZATION_MIN_DURATION_OFF", "0.0"))
# Stock pyannote drops a voice unless it owns 20% of a 10s chunk alone, so a short
# reply never gets an embedding and the whole file becomes SPEAKER_00.
DIAR_MIN_ACTIVE_RATIO = float(os.environ.get("DIARIZATION_MIN_ACTIVE_RATIO", "0.02"))
# Euclidean distance on unit embeddings. 0.38 ≈ cosine 0.93, so only near-identical
# windows stay in one cluster.
DIAR_SPLIT_DISTANCE = float(os.environ.get("DIARIZATION_SPLIT_DISTANCE", "0.38"))
DIAR_MAX_SPLIT_SIMILARITY = float(os.environ.get("DIARIZATION_MAX_SPLIT_SIMILARITY", "0.96"))
# A Pyannote label is split only when its own windows are clearly different voices.
# 0.75 Euclidean on unit vectors is cosine similarity ≈ 0.72. A shout or a small
# pitch change stays one person. Pyannote's own VBx threshold is unchanged.
DIAR_WITHIN_SPLIT_DISTANCE = float(os.environ.get("DIARIZATION_WITHIN_SPLIT_DISTANCE", "0.75"))
DIAR_WITHIN_MAX_SIMILARITY = float(os.environ.get("DIARIZATION_WITHIN_MAX_SIMILARITY", "0.72"))
DIAR_WITHIN_WINDOW_S = float(os.environ.get("DIARIZATION_WITHIN_WINDOW_S", "0.50"))
DIAR_WITHIN_HOP_S = float(os.environ.get("DIARIZATION_WITHIN_HOP_S", "0.25"))
DIAR_WITHIN_MIN_SPAN_S = float(os.environ.get("DIARIZATION_WITHIN_MIN_SPAN_S", "0.35"))
# A window fragment shorter than this is folded back into the longer voice.
# One third of a second is not a person. The word pass is what may still
# separate a real second voice.
DIAR_WITHIN_MIN_SPEAKER_S = float(os.environ.get("DIARIZATION_WITHIN_MIN_SPEAKER_S", "1.0"))
DIAR_WITHIN_MAX_CLUSTERS = int(os.environ.get("DIARIZATION_WITHIN_MAX_CLUSTERS", "4"))
# Voice prints are built from whole runs of words, then every word is assigned
# to the nearest print. A new print needs about a second of speech that is
# tighter with itself than with any print already known. A single odd word
# stays with the nearest known print. Nothing here reads the transcript text.
DIAR_WORD_WINDOW_S = float(os.environ.get("DIARIZATION_WORD_WINDOW_S", "0.50"))
DIAR_PRINT_CLUSTER_DISTANCE = float(os.environ.get("DIARIZATION_PRINT_CLUSTER_DISTANCE", "0.50"))
DIAR_PRINT_JOIN = float(os.environ.get("DIARIZATION_PRINT_JOIN", "0.88"))
DIAR_PRINT_FAR = float(os.environ.get("DIARIZATION_PRINT_FAR", "0.80"))
DIAR_PRINT_OBVIOUS = float(os.environ.get("DIARIZATION_PRINT_OBVIOUS", "0.55"))
DIAR_PRINT_BORDER = float(os.environ.get("DIARIZATION_PRINT_BORDER", "0.84"))
DIAR_PRINT_MARGIN = float(os.environ.get("DIARIZATION_PRINT_MARGIN", "0.12"))
DIAR_PRINT_SELF = float(os.environ.get("DIARIZATION_PRINT_SELF", "0.92"))
DIAR_PRINT_MIN_SECONDS = float(os.environ.get("DIARIZATION_PRINT_MIN_SECONDS", "1.0"))
DIAR_PRINT_MIN_WORDS = int(os.environ.get("DIARIZATION_PRINT_MIN_WORDS", "3"))
DIAR_WORD_SPLIT_DISTANCE = DIAR_PRINT_CLUSTER_DISTANCE
DIAR_WORD_SPLIT_MAX_SIMILARITY = DIAR_PRINT_BORDER
DIAR_WORD_MOVE_MARGIN = DIAR_PRINT_MARGIN
DIAR_WORD_MIN_MATCH = DIAR_PRINT_JOIN
DIAR_WORD_MIN_NEW_WORDS = DIAR_PRINT_MIN_WORDS
DIAR_WORD_MIN_NEW_SECONDS = DIAR_PRINT_MIN_SECONDS
DIAR_WINDOW_S = float(os.environ.get("DIARIZATION_WINDOW_S", "1.25"))
DIAR_MAX_WINDOWS = int(os.environ.get("DIARIZATION_MAX_WINDOWS", "160"))
DIAR_MIN_SPEAKER_S = float(os.environ.get("DIARIZATION_MIN_SPEAKER_S", "0.8"))
DIAR_MAX_AUTO_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_AUTO_SPEAKERS", "8"))
# Bounds sent to pyannote on every call when the request does not set them. A file
# with one real voice will be split in two unless the request sends min_speakers=1.
DIAR_DEFAULT_MIN_SPEAKERS = int(os.environ.get("DIARIZATION_MIN_SPEAKERS", "2"))
DIAR_DEFAULT_MAX_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "8"))

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


def _load_mono_16k(path):
    """Decode to one channel at 16 kHz explicitly; whisperx/ffmpeg is the fallback decoder."""
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
        channels = int(waveform.shape[0])
        if channels > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)
        audio = waveform.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)
        print(
            f"[audio] torchaudio source_sr={sample_rate} channels={channels} -> "
            f"mono sr={SAMPLE_RATE} samples={audio.shape[0]}",
            flush=True,
        )
        return audio, {"decoder": "torchaudio", "source_sr": int(sample_rate), "source_channels": channels}
    except Exception as exc:
        print(f"[audio] torchaudio load failed, using ffmpeg: {type(exc).__name__}: {exc}", flush=True)
    audio = whisperx.load_audio(path).astype(np.float32, copy=False)
    print(f"[audio] ffmpeg -> mono sr={SAMPLE_RATE} samples={audio.shape[0]}", flush=True)
    return audio, {"decoder": "ffmpeg", "source_sr": None, "source_channels": None}


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


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _compact_labels(labels):
    labels = np.asarray(labels, dtype=int)
    mapping = {old: new for new, old in enumerate(sorted(set(int(value) for value in labels)))}
    return np.array([mapping[int(value)] for value in labels], dtype=int)


def _min_centroid_cosine(matrix, labels):
    labels = np.asarray(labels, dtype=int)
    ids = sorted(set(int(value) for value in labels))
    if len(ids) < 2:
        return 1.0
    centroids = []
    for label in ids:
        block = matrix[labels == label]
        if block.size == 0:
            continue
        centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-8:
            continue
        centroids.append(centroid / norm)
    if len(centroids) < 2:
        return 1.0
    similarity = np.vstack(centroids) @ np.vstack(centroids).T
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    return float(np.min(similarity[mask]))


def _merge_to_max(matrix, labels, max_clusters):
    labels = np.asarray(labels, dtype=int).copy()
    while len(set(int(value) for value in labels)) > max_clusters:
        ids = sorted(set(int(value) for value in labels))
        centroids = []
        for label in ids:
            centroid = matrix[labels == label].mean(axis=0)
            centroids.append(centroid / max(np.linalg.norm(centroid), 1e-8))
        similarity = np.vstack(centroids) @ np.vstack(centroids).T
        np.fill_diagonal(similarity, -2.0)
        left, right = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
        labels[labels == ids[int(right)]] = ids[int(left)]
    return _compact_labels(labels)


def _kmeans_labels(matrix, n_clusters):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=int(n_clusters), n_init=10, random_state=0).fit_predict(matrix).astype(int)


def _cluster_embeddings(matrix, distance_threshold, min_clusters=1, max_clusters=8):
    """Cluster speaker embeddings.

    Rows are L2-normalized. Returns labels starting at 0 and the lowest cosine
    similarity between cluster centroids (1.0 when a single cluster remains).
    """
    matrix = _normalize_rows(matrix)
    count = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if count == 0:
        return np.zeros(0, dtype=int), 1.0
    max_clusters = int(max(1, min(int(max_clusters), count)))
    min_clusters = int(max(1, min(int(min_clusters), max_clusters)))
    if count == 1 or max_clusters == 1:
        return np.zeros(count, dtype=int), 1.0

    if min_clusters >= 2 and min_clusters == max_clusters:
        labels = _kmeans_labels(matrix, min_clusters)
    else:
        from scipy.cluster.hierarchy import fcluster, linkage

        linked = linkage(matrix, method="average", metric="euclidean")
        raw = fcluster(linked, t=float(distance_threshold), criterion="distance")
        labels = _merge_to_max(matrix, _compact_labels(raw), max_clusters)
        if len(set(int(value) for value in labels)) < min_clusters:
            labels = _kmeans_labels(matrix, min_clusters)
    return _compact_labels(labels), _min_centroid_cosine(matrix, labels)


def _fold_short_clusters(spans, labels, matrix, min_seconds, protect=1):
    labels = np.asarray(labels, dtype=int).copy()
    durations = {}
    for label, (start, end) in zip(labels, spans):
        durations[int(label)] = durations.get(int(label), 0.0) + max(0.0, float(end) - float(start))
    keep = [label for label, duration in durations.items() if duration >= min_seconds]
    if len(keep) < max(1, int(protect)) or len(keep) == len(durations):
        return labels
    centroids = {}
    for label in keep:
        centroid = matrix[labels == label].mean(axis=0)
        centroids[label] = centroid / max(np.linalg.norm(centroid), 1e-8)
    for label, duration in durations.items():
        if duration >= min_seconds or not np.any(labels == label):
            continue
        point = matrix[labels == label].mean(axis=0)
        point = point / max(np.linalg.norm(point), 1e-8)
        nearest = max(keep, key=lambda item: float(np.dot(point, centroids[item])))
        labels[labels == label] = nearest
    return labels


def _highpass(audio):
    """Drop rumble below ~80 Hz. It smears speaker embeddings and is not speech."""
    from scipy.signal import butter, sosfiltfilt

    if audio.size < SAMPLE_RATE // 4:
        return audio
    sos = butter(2, 80.0, btype="highpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32, copy=False)


def _suppress_stationary_noise(audio):
    """Attenuate steady noise (hum, hiss, room tone) on the diarization copy only."""
    from scipy.signal import istft, stft

    if audio.size < SAMPLE_RATE:
        return audio
    frame = 512
    _, _, spectrum = stft(audio, fs=SAMPLE_RATE, nperseg=frame, noverlap=frame - 128, window="hann")
    magnitude = np.abs(spectrum)
    if magnitude.size == 0:
        return audio
    noise = np.percentile(magnitude, 20, axis=1, keepdims=True)
    signal = np.median(magnitude, axis=1, keepdims=True)
    if float(np.mean(noise)) < 0.2 * float(np.mean(signal) + 1e-8):
        return audio
    cleaned = np.maximum(magnitude - noise, 0.08 * magnitude)
    _, restored = istft(
        cleaned * np.exp(1j * np.angle(spectrum)),
        fs=SAMPLE_RATE,
        nperseg=frame,
        noverlap=frame - 128,
        window="hann",
    )
    restored = np.asarray(restored, dtype=np.float32)
    if restored.size >= audio.size:
        return restored[: audio.size]
    padded = np.zeros_like(audio)
    padded[: restored.size] = restored
    return padded


def _prepare_diarization_audio(audio):
    """Make proxy WAVs separable: level, rumble, and steady noise. Transcription audio stays raw."""
    prepared = np.array(audio, dtype=np.float32, copy=True)
    if prepared.size == 0:
        return prepared
    prepared -= float(np.mean(prepared))
    try:
        prepared = _highpass(prepared)
        prepared = _suppress_stationary_noise(prepared)
    except Exception as exc:
        _diag_extra["audio_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[diarize] audio cleanup skipped: {type(exc).__name__}: {exc}", flush=True)
    peak = float(np.max(np.abs(prepared))) if prepared.size else 0.0
    if peak < 1e-5:
        return prepared
    if peak < 0.2 or peak > 0.98:
        prepared *= 0.9 / peak
    return np.clip(prepared, -1.0, 1.0)


def _assign_param(owner, name, value):
    if owner is None or not hasattr(owner, name):
        return False
    try:
        setattr(owner, name, value)
        return True
    except Exception as exc:
        try:
            owner.__dict__[name] = value
            return True
        except Exception:
            print(f"[diarize] could not set {name}: {exc}", flush=True)
            return False


def _install_active_ratio(clustering, ratio):
    current = getattr(clustering, "filter_embeddings", None)
    if current is None:
        return
    if getattr(current, "_audvid_wrapped", False):
        current._audvid_ratio = float(ratio)
        return
    original = current

    def wrapped(embeddings, segmentations=None, min_active_ratio=None, _orig=original):
        used = wrapped._audvid_ratio if min_active_ratio is None else min_active_ratio
        try:
            kept = _orig(embeddings, segmentations, min_active_ratio=used)
        except TypeError:
            kept = _orig(embeddings, segmentations)
        _record_clustering_input(embeddings, kept, used)
        return kept

    wrapped._audvid_wrapped = True
    wrapped._audvid_ratio = float(ratio)
    clustering.filter_embeddings = wrapped


def _configure_pipeline(model):
    params = {
        "segmentation": {"min_duration_off": DIAR_MIN_DURATION_OFF},
        "clustering": {"threshold": DIAR_THRESHOLD, "Fa": DIAR_FA, "Fb": DIAR_FB},
    }
    try:
        model.instantiate(params)
    except Exception as exc:
        print(f"[diarize] instantiate skipped: {exc}", flush=True)
    clustering = getattr(model, "clustering", None)
    segmentation = getattr(model, "segmentation", None)
    _assign_param(clustering, "threshold", DIAR_THRESHOLD)
    _assign_param(clustering, "Fa", DIAR_FA)
    _assign_param(clustering, "Fb", DIAR_FB)
    _assign_param(segmentation, "min_duration_off", 0.0)
    _assign_param(segmentation, "min_duration_on", 0.0)
    _install_active_ratio(clustering, DIAR_MIN_ACTIVE_RATIO)
    print(
        "[diarize] sensitivity "
        f"threshold={getattr(clustering, 'threshold', None)} "
        f"Fa={getattr(clustering, 'Fa', None)} "
        f"Fb={getattr(clustering, 'Fb', None)} "
        f"min_duration_off={getattr(segmentation, 'min_duration_off', None)} "
        f"min_active_ratio={DIAR_MIN_ACTIVE_RATIO}",
        flush=True,
    )


_diag_runs = []
# Extra per-job diagnostics. Write-only from the algorithm's point of view: nothing in the
# separation code reads it back. Cleared by _separate_speakers, returned as output["diarization_debug"].
_diag_extra = {}


def _diag_current():
    if not _diag_runs:
        _diag_runs.append({})
    return _diag_runs[-1]


def _record_clustering_input(embeddings, kept, ratio):
    diag = _diag_current()
    try:
        train = kept[0] if isinstance(kept, tuple) else kept
        total = int(embeddings.shape[0] * embeddings.shape[1])
        valid = int(np.sum(~np.any(np.isnan(embeddings), axis=2)))
        diag["clustering_input"] = int(train.shape[0])
        diag["embeddings_total"] = total
        diag["embeddings_valid"] = valid
        print(
            f"[diarize] before clustering: embeddings total={total} valid={valid} "
            f"kept_for_clustering={int(train.shape[0])} min_active_ratio={ratio}",
            flush=True,
        )
        if int(train.shape[0]) < 2:
            print(
                "[diarize] WARNING fewer than 2 embeddings reach clustering; "
                "pyannote will return a single speaker",
                flush=True,
            )
    except Exception as exc:
        print(f"[diarize] clustering-input log failed: {exc}", flush=True)


def _diagnostic_hook(step_name, step_artifact, file=None, total=None, completed=None):
    if completed is not None or step_artifact is None:
        return
    diag = _diag_current()
    try:
        if step_name == "segmentation":
            chunks, frames, local = step_artifact.data.shape
            active = int(np.sum(np.nanmax(step_artifact.data, axis=1) > 0.5))
            diag["segmentation_chunks"] = int(chunks)
            diag["segmentation_local_speakers"] = int(local)
            diag["segmentation_active_tracks"] = active
            print(
                f"[diarize] segmentation: chunks={chunks} frames_per_chunk={frames} "
                f"local_speakers={local} active_chunk_speakers={active}",
                flush=True,
            )
        elif step_name == "speaker_counting":
            data = np.asarray(step_artifact.data)
            step = float(step_artifact.sliding_window.step)
            speech_s = round(float(np.sum(data >= 1)) * step, 2)
            overlap_s = round(float(np.sum(data >= 2)) * step, 2)
            diag["speech_s"] = speech_s
            diag["overlap_s"] = overlap_s
            diag["speaker_count_step_s"] = round(step, 5)
            diag["max_simultaneous"] = int(np.nanmax(data)) if data.size else 0
            print(
                f"[diarize] speaker counting: speech={speech_s}s overlap={overlap_s}s "
                f"max_simultaneous={diag['max_simultaneous']}",
                flush=True,
            )
        elif step_name == "embeddings":
            shape = tuple(int(v) for v in np.shape(step_artifact))
            nan_rows = int(np.sum(np.any(np.isnan(step_artifact), axis=-1))) if len(shape) == 3 else -1
            diag["embeddings_shape"] = list(shape)
            diag["embeddings_nan"] = nan_rows
            print(f"[diarize] embeddings: shape={shape} nan_rows={nan_rows}", flush=True)
        elif step_name == "discrete_diarization":
            data = np.asarray(step_artifact.data)
            speakers = int(np.sum(np.nanmax(data, axis=0) > 0)) if data.size else 0
            diag["clustered_speakers"] = speakers
            print(f"[diarize] after clustering: speakers={speakers}", flush=True)
    except Exception as exc:
        print(f"[diarize] hook log failed at {step_name}: {exc}", flush=True)


def _call_pipeline(model, audio, num_speakers, min_speakers, max_speakers):
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).reshape(1, -1)
    _diag_runs.append(
        {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "waveform_shape": list(waveform.shape),
            "sample_rate": SAMPLE_RATE,
        }
    )
    print(
        f"[diarize] pipeline input waveform={tuple(waveform.shape)} sr={SAMPLE_RATE} "
        f"num_speakers={num_speakers} min_speakers={min_speakers} max_speakers={max_speakers}",
        flush=True,
    )
    return model(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hook=_diagnostic_hook,
    )


def _speaker_bounds(num_speakers, min_speakers, max_speakers):
    if num_speakers:
        return int(num_speakers), int(num_speakers), int(num_speakers)
    low = int(min_speakers) if min_speakers else DIAR_DEFAULT_MIN_SPEAKERS
    high = int(max_speakers) if max_speakers else DIAR_DEFAULT_MAX_SPEAKERS
    low = max(1, low)
    high = max(low, high)
    return None, low, high


def _label_count(annotation):
    if annotation is None:
        return 0
    try:
        return len(list(annotation.labels()))
    except Exception:
        return 0


def _best_annotation(output):
    from pyannote.core import Annotation

    if isinstance(output, Annotation):
        return output
    inclusive = getattr(output, "speaker_diarization", None)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    # Inclusive keeps a second voice during overlap. Exclusive deletes one of them,
    # so two simultaneous people become one label. Prefer inclusive whenever it exists.
    if inclusive is not None and len(inclusive) > 0:
        return inclusive
    return exclusive


def _probe_window_s(audio):
    duration = len(audio) / SAMPLE_RATE
    if duration < DIAR_WINDOW_S * 2:
        return max(1.0, duration / 2.2)
    return DIAR_WINDOW_S


def _speech_spans(audio, annotation, window_s):
    win = int(window_s * SAMPLE_RATE)
    minimum = int(SAMPLE_RATE)
    regions = []
    if annotation is not None and len(annotation) > 0:
        for segment, _, _speaker in annotation.itertracks(yield_label=True):
            start = int(max(0.0, float(segment.start)) * SAMPLE_RATE)
            end = int(min(len(audio) / SAMPLE_RATE, float(segment.end)) * SAMPLE_RATE)
            start = max(0, min(start, len(audio)))
            end = max(start, min(end, len(audio)))
            if end - start >= minimum:
                regions.append((start, end))
    if not regions and len(audio) >= minimum:
        regions.append((0, len(audio)))

    spans = []
    for start, end in regions:
        if end - start < win:
            spans.append((start, end))
            continue
        cursor = start
        while cursor + win <= end:
            spans.append((cursor, cursor + win))
            cursor += win
    _diag_note("probe", windows_before_cap=len(spans), window_cap=DIAR_MAX_WINDOWS, window_s=window_s)
    if len(spans) > DIAR_MAX_WINDOWS:
        chosen = np.linspace(0, len(spans) - 1, DIAR_MAX_WINDOWS).astype(int)
        spans = [spans[int(index)] for index in chosen]
    return spans


def _embed_spans(embedding_model, audio, spans, window_s, min_samples=None):
    win = int(window_s * SAMPLE_RATE)
    if min_samples is None:
        min_samples = SAMPLE_RATE
    batch = []
    kept = []
    vectors = []

    def flush():
        if not batch:
            return
        stacked = np.stack(batch).astype(np.float32, copy=False)
        tensor = torch.from_numpy(stacked[:, None, :])
        embedded = np.asarray(embedding_model(tensor), dtype=np.float32)
        if embedded.ndim == 1:
            embedded = embedded.reshape(1, -1)
        vectors.append(embedded)
        batch.clear()

    for start, end in spans:
        clip = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        if clip.size < min_samples:
            continue
        rms = float(np.sqrt(np.mean(np.square(clip))))
        if rms < 0.008:
            continue
        if clip.shape[0] > win:
            clip = clip[:win]
        elif clip.shape[0] < win:
            clip = np.pad(clip, (0, win - clip.shape[0]))
        batch.append(clip)
        kept.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
        if len(batch) >= 8:
            flush()
    flush()
    if not vectors:
        return [], None
    return kept, np.vstack(vectors)


def _annotation_from_spans(spans, labels):
    from pyannote.core import Annotation, Segment

    merged = []
    for (start, end), label in zip(spans, labels):
        label = int(label)
        if merged and merged[-1][2] == label and start <= merged[-1][1] + 0.45:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end), label])
    order = []
    for _start, _end, label in merged:
        if label not in order:
            order.append(label)
    names = {label: f"SPEAKER_{index:02d}" for index, label in enumerate(order)}
    annotation = Annotation()
    for index, (start, end, label) in enumerate(merged):
        if end - start < 0.3:
            continue
        annotation[Segment(start, end), index] = names[label]
    return annotation


def _probe_speakers(model, audio, annotation, num_speakers, min_speakers, max_speakers):
    embedding_model = getattr(model, "_embedding", None)
    if embedding_model is None:
        return None, 1.0, None, None
    window_s = _probe_window_s(audio)
    spans = _speech_spans(audio, annotation, window_s)
    if len(spans) < 2:
        return None, 1.0, None, None
    kept, matrix = _embed_spans(embedding_model, audio, spans, window_s)
    if matrix is None or len(kept) < 2:
        return None, 1.0, None, None
    finite = np.all(np.isfinite(matrix), axis=1)
    if not np.all(finite):
        kept = [span for span, ok in zip(kept, finite) if ok]
        matrix = matrix[finite]
    if len(kept) < 2:
        return None, 1.0, None, None

    forced = num_speakers
    ceiling = max_speakers or DIAR_MAX_AUTO_SPEAKERS
    if forced:
        ceiling = forced
        floor = forced
    elif min_speakers and min_speakers > 1:
        floor = min_speakers
        ceiling = max(ceiling, min_speakers)
    else:
        floor = 1
    ceiling = max(1, min(int(ceiling), len(kept), DIAR_MAX_AUTO_SPEAKERS if not forced else int(ceiling)))
    floor = max(1, min(int(floor), ceiling))

    labels, similarity = _cluster_embeddings(
        matrix,
        DIAR_SPLIT_DISTANCE,
        min_clusters=floor if floor > 1 else 1,
        max_clusters=ceiling,
    )
    normalized = _normalize_rows(matrix)
    labels = _fold_short_clusters(
        kept,
        labels,
        normalized,
        DIAR_MIN_SPEAKER_S,
        protect=floor if floor > 1 else 1,
    )
    labels = _compact_labels(labels)
    similarity = _min_centroid_cosine(normalized, labels)
    found = len(set(int(value) for value in labels))
    distinct = found >= 2 and similarity <= DIAR_MAX_SPLIT_SIMILARITY
    if forced and found >= 2:
        target = found
    elif floor > 1 and found >= 2:
        target = found
    elif distinct:
        target = found
    else:
        target = None
    return target, similarity, kept, labels


def _speaker_windows(audio, annotation, speaker, window_s, hop_s=None, min_s=1.0):
    """Windows inside one Pyannote speaker. Hop may be shorter than the window."""
    win = max(1, int(window_s * SAMPLE_RATE))
    hop = win if hop_s is None else max(1, int(hop_s * SAMPLE_RATE))
    minimum = max(1, int(min_s * SAMPLE_RATE))
    spans = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        if str(label) != str(speaker):
            continue
        start = int(max(0.0, float(segment.start)) * SAMPLE_RATE)
        end = int(min(len(audio) / SAMPLE_RATE, float(segment.end)) * SAMPLE_RATE)
        start = max(0, min(start, len(audio)))
        end = max(start, min(end, len(audio)))
        if end - start < minimum:
            continue
        if end - start <= win:
            spans.append((start, end))
            continue
        cursor = start
        while cursor + minimum <= end:
            spans.append((cursor, min(end, cursor + win)))
            if cursor + win >= end:
                break
            cursor += hop
    cap = 480
    if len(spans) > cap:
        chosen = np.linspace(0, len(spans) - 1, cap).astype(int)
        spans = [spans[int(index)] for index in chosen]
    return spans


def _covered_seconds(spans):
    """Longest contiguous piece, so three brief boundary blips are not one voice."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        return 0.0
    return float(max(end - start for start, end in merged))


def _fold_short_if_same_voice(spans, labels, matrix, min_seconds, max_similarity):
    """Fold every fragment shorter than min_seconds into the nearest longer voice.

    A third of a second used to become its own speaker and then could not be
    merged, because the fragment matched itself perfectly. Short replies of a
    person who already has a real print are restored when words are assigned.
    max_similarity is unused; it remains so older callers keep the same signature.
    """
    del max_similarity
    labels = np.asarray(labels, dtype=int).copy()
    spans_by_label = {}
    for label, (start, end) in zip(labels, spans):
        spans_by_label.setdefault(int(label), []).append((float(start), float(end)))
    durations = {label: _covered_seconds(pieces) for label, pieces in spans_by_label.items()}
    long = [label for label, duration in durations.items() if duration >= min_seconds]
    if not long:
        return labels
    centroids = {}
    for label in long:
        centroid = matrix[labels == label].mean(axis=0)
        centroids[label] = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
    for label, duration in durations.items():
        if duration >= min_seconds or not np.any(labels == label):
            continue
        point = matrix[labels == label].mean(axis=0)
        point = point / max(float(np.linalg.norm(point)), 1e-8)
        nearest = max(long, key=lambda item: float(np.dot(point, centroids[item])))
        labels[labels == label] = nearest
    return labels


def _collapse_named_windows(windows):
    by_name = {}
    for start, end, name in windows:
        by_name.setdefault(name, []).append((float(start), float(end)))
    tracks = []
    for name, spans in by_name.items():
        spans.sort()
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            if end - start >= 0.05:
                tracks.append((start, end, name))
    tracks.sort()
    return tracks


def _annotation_from_named(tracks):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for index, (start, end, name) in enumerate(tracks):
        start, end = float(start), float(end)
        if end - start < 0.05 or not name:
            continue
        annotation[Segment(start, end), index] = str(name)
    return annotation


def _split_within_speakers(model, audio, annotation):
    """Split one Pyannote label when its own short windows are not the same voice.

    Different Pyannote labels are never merged. A split is kept when the centroid
    cosine is at or below DIAR_WITHIN_MAX_SIMILARITY. A short fragment is folded
    back only when it still matches the longer voice, so a fast reply survives.
    """
    embedding_model = getattr(model, "_embedding", None)
    if embedding_model is None or annotation is None or len(annotation) == 0:
        return None, 0, [{
            "reason": "no_embedding_model" if embedding_model is None else "no_annotation",
        }]
    window_s = DIAR_WITHIN_WINDOW_S
    durations = {}
    for segment, _, label in annotation.itertracks(yield_label=True):
        name = str(label)
        durations[name] = durations.get(name, 0.0) + max(0.0, float(segment.end) - float(segment.start))
    order = sorted(durations, key=lambda name: durations[name], reverse=True)
    used = set(order)
    tracks = []
    notes = []
    splits = 0
    ceiling = max(DIAR_MAX_AUTO_SPEAKERS, len(order)) + 8
    for speaker in order:
        original = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            if str(label) == speaker:
                original.append((float(segment.start), float(segment.end), speaker))
        if len(used) >= ceiling:
            tracks.extend(original)
            continue
        spans = _speaker_windows(
            audio, annotation, speaker, window_s, DIAR_WITHIN_HOP_S, DIAR_WITHIN_MIN_SPAN_S
        )
        kept, matrix = (
            _embed_spans(
                embedding_model, audio, spans, window_s,
                min_samples=int(DIAR_WITHIN_MIN_SPAN_S * SAMPLE_RATE),
            )
            if len(spans) >= 2 else ([], None)
        )
        if matrix is None or len(kept) < 2:
            tracks.extend(original)
            notes.append({"speaker": speaker, "windows": len(kept), "split": False, "reason": "too_few_windows"})
            continue
        finite = np.all(np.isfinite(matrix), axis=1)
        if not np.all(finite):
            kept = [span for span, ok in zip(kept, finite) if ok]
            matrix = matrix[finite]
        if len(kept) < 2:
            tracks.extend(original)
            notes.append({"speaker": speaker, "windows": len(kept), "split": False, "reason": "non_finite"})
            continue
        room = max(2, min(DIAR_WITHIN_MAX_CLUSTERS, ceiling - len(used) + 1))
        labels, _similarity = _cluster_embeddings(
            matrix, DIAR_WITHIN_SPLIT_DISTANCE, min_clusters=1, max_clusters=room
        )
        normalized = _normalize_rows(matrix)
        labels = _fold_short_if_same_voice(
            kept, labels, normalized, DIAR_WITHIN_MIN_SPEAKER_S, DIAR_WITHIN_MAX_SIMILARITY
        )
        labels = _compact_labels(labels)
        similarity = _min_centroid_cosine(normalized, labels)
        found = len(set(int(value) for value in labels))
        if found < 2 or similarity > DIAR_WITHIN_MAX_SIMILARITY:
            tracks.extend(original)
            notes.append({
                "speaker": speaker,
                "windows": len(kept),
                "split": False,
                "clusters": found,
                "centroid_cosine": round(float(similarity), 4),
                "reason": "same_voice",
            })
            continue
        sizes = {int(value): 0.0 for value in labels}
        for label, (start, end) in zip(labels, kept):
            sizes[int(label)] = sizes[int(label)] + (float(end) - float(start))
        primary = max(sizes, key=sizes.get)
        rename = {primary: speaker}
        for label in sorted(sizes):
            if label == primary:
                continue
            number = 0
            while True:
                candidate = f"SPEAKER_{number:02d}"
                number += 1
                if candidate not in used and candidate not in rename.values():
                    rename[label] = candidate
                    used.add(candidate)
                    break
        named = []
        for (start, end), label in zip(kept, labels):
            named.append((float(start), float(end), rename[int(label)]))
        tracks.extend(_collapse_named_windows(named))
        splits += found - 1
        notes.append({
            "speaker": speaker,
            "windows": len(kept),
            "split": True,
            "clusters": found,
            "centroid_cosine": round(float(similarity), 4),
            "new_labels": sorted(set(rename.values())),
        })
    if splits <= 0:
        return None, 0, notes
    return _annotation_from_named(tracks), splits, notes


def _separate_speakers(pipeline, audio, num_speakers, min_speakers, max_speakers):
    model = pipeline.model
    _diag_runs.clear()
    _diag_extra.clear()
    _diag_note("bounds", requested={"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers})
    num_speakers, min_speakers, max_speakers = _speaker_bounds(num_speakers, min_speakers, max_speakers)
    _diag_note("bounds", effective={"num_speakers": num_speakers, "min_speakers": min_speakers, "max_speakers": max_speakers})
    prepared = _prepare_diarization_audio(audio)
    _diag_record_audio(audio, prepared)
    _configure_pipeline(model)
    _diag_record_settings(model)
    output = _call_pipeline(model, prepared, num_speakers, min_speakers, max_speakers)
    annotation = _best_annotation(output)
    found = _label_count(annotation)
    _diag_record_s0(output, annotation)
    rescue = "none"
    similarity = None
    requested = num_speakers or 0
    minimum = min_speakers or 0
    collapsed = found == 1 and requested != 1 and max_speakers != 1
    short = found >= 1 and ((requested and found < requested) or (minimum and found < minimum))
    within_ann, within_splits, within_notes = None, 0, []
    try:
        within_ann, within_splits, within_notes = _split_within_speakers(model, prepared, annotation)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        print(f"[diarize] within-speaker split skipped: {type(exc).__name__}: {exc}", flush=True)
        _diag_note("within_split", status="failed", reason=f"{type(exc).__name__}: {exc}")
    else:
        _diag_note(
            "within_split",
            status="ran",
            splits=int(within_splits),
            speakers=within_notes,
        )
    if within_ann is not None and within_splits > 0 and _label_count(within_ann) > found:
        annotation = within_ann
        rescue = "split_within"
    _diag_note(
        "trigger",
        collapsed=bool(collapsed),
        short=bool(short),
        probe_ran=True,
        within_split=rescue == "split_within",
        s0_labels=found,
    )
    if rescue != "split_within" and (collapsed or short):
        try:
            target, similarity, spans, labels = _probe_speakers(
                model, prepared, annotation, num_speakers, min_speakers, max_speakers
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            print(f"[diarize] speaker probe skipped: {type(exc).__name__}: {exc}", flush=True)
            target, spans, labels = None, None, None
            _diag_note("probe", status="failed", reason=f"{type(exc).__name__}: {exc}")
        _diag_record_probe(target, similarity, spans, labels)
        if target and target > found:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            retried = None
            try:
                retried = _best_annotation(
                    _call_pipeline(model, prepared, target, target, target)
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                print(f"[diarize] recount skipped: {type(exc).__name__}: {exc}", flush=True)
            _diag_note("probe", retry_target=int(target), retry_labels=_label_count(retried))
            if _label_count(retried) >= 2:
                annotation = retried
                rescue = "num_speakers"
            elif spans is not None and labels is not None and len(set(int(v) for v in labels)) >= 2:
                annotation = _annotation_from_spans(spans, labels)
                rescue = "windows"
    final = _label_count(annotation)
    _diag_note("final", rescue=rescue, s0_labels=found, final_labels=final)
    print(
        f"[diarize] speakers {found} -> {final} rescue={rescue} "
        f"similarity={None if similarity is None else round(similarity, 3)}",
        flush=True,
    )
    return annotation, rescue


def _rows_from_annotation(annotation):
    rows = []
    if annotation is None:
        return rows
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end - start < 0.05 or speaker is None or speaker == "":
            continue
        rows.append({"start": round(start, 3), "end": round(end, 3), "speaker": str(speaker)})
    return rows


# ---------------------------------------------------------------------------------------
# Diagnostics only (diarization_raw / diarization_debug).
# Everything below observes; it never mutates the annotation, the audio, the pipeline, the
# transcription result or any algorithm constant. Every entry point swallows its own errors,
# so a diagnostics bug can never fail or change a transcription.
# ---------------------------------------------------------------------------------------
DIAG_PATCH_VERSION = 6
DIAG_BASE_COMMIT = "c99803c"
ROWS_CAP = 5000


def _diag_note(section, **values):
    try:
        _diag_extra.setdefault(section, {}).update(values)
    except Exception:
        pass


def _as_number(value):
    try:
        number = float(value)
    except Exception:
        return None
    return None if (math.isnan(number) or math.isinf(number)) else number


def _effective_settings(model):
    """Read back what the pipeline object actually holds after _configure_pipeline."""
    clustering = getattr(model, "clustering", None)
    segmentation = getattr(model, "segmentation", None)
    filter_fn = getattr(clustering, "filter_embeddings", None)
    return {
        "clustering_class": type(clustering).__name__ if clustering is not None else None,
        "ahc_threshold": _as_number(getattr(clustering, "threshold", None)),
        "Fa": _as_number(getattr(clustering, "Fa", None)),
        "Fb": _as_number(getattr(clustering, "Fb", None)),
        "min_duration_on": _as_number(getattr(segmentation, "min_duration_on", None)),
        "min_duration_off": _as_number(getattr(segmentation, "min_duration_off", None)),
        "min_active_ratio_wrapper_installed": bool(getattr(filter_fn, "_audvid_wrapped", False)),
        "min_active_ratio": _as_number(getattr(filter_fn, "_audvid_ratio", None)),
        "probe": {
            "split_distance_euclidean": DIAR_SPLIT_DISTANCE,
            "max_split_similarity": DIAR_MAX_SPLIT_SIMILARITY,
            "within_split_distance_euclidean": DIAR_WITHIN_SPLIT_DISTANCE,
            "within_max_similarity": DIAR_WITHIN_MAX_SIMILARITY,
            "within_window_s": DIAR_WITHIN_WINDOW_S,
            "within_hop_s": DIAR_WITHIN_HOP_S,
            "within_min_speaker_s": DIAR_WITHIN_MIN_SPEAKER_S,
            "window_s": DIAR_WINDOW_S,
            "max_windows": DIAR_MAX_WINDOWS,
            "min_speaker_s": DIAR_MIN_SPEAKER_S,
            "max_auto_speakers": DIAR_MAX_AUTO_SPEAKERS,
        },
    }


def _library_versions():
    names = (
        "torch", "torchaudio", "whisperx", "pyannote.audio", "faster-whisper",
        "ctranslate2", "numpy", "scikit-learn", "runpod",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def _diag_record_audio(raw, prepared):
    try:
        def level(values):
            # Block-wise so a long recording never allocates a full float64 copy.
            values = np.asarray(values).reshape(-1)
            if values.size == 0:
                return {"samples": 0}
            total = 0.0
            peak = 0.0
            for offset in range(0, values.size, 1 << 20):
                block = values[offset:offset + (1 << 20)].astype(np.float64)
                total += float(np.dot(block, block))
                peak = max(peak, float(np.max(np.abs(block))))
            return {
                "samples": int(values.size),
                "peak": round(peak, 5),
                "rms": round(math.sqrt(total / values.size), 5),
            }

        _diag_extra["audio_prep"] = {
            "input": level(raw),
            "prepared": level(prepared),
            "cleanup_error": _diag_extra.get("audio_cleanup_error"),
        }
    except Exception:
        pass


def _diag_record_settings(model):
    try:
        _diag_extra["effective_settings"] = _effective_settings(model)
    except Exception as exc:
        _diag_extra["effective_settings"] = {"error": f"{type(exc).__name__}: {exc}"}


def _diag_record_s0(output, annotation):
    """S0 = the annotation _best_annotation picked from the FIRST Pyannote call, as plain rows."""
    try:
        inclusive = getattr(output, "speaker_diarization", None)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is not None and annotation is exclusive:
            selected, other_kind, other = "exclusive", "inclusive", inclusive
        elif annotation is not None and annotation is inclusive:
            selected, other_kind, other = "inclusive", "exclusive", exclusive
        else:
            selected, other_kind, other = "annotation", None, None
        _diag_extra["s0"] = {
            "selected": selected,
            "labels": _label_count(annotation),
            "inclusive_labels": _label_count(inclusive),
            "exclusive_labels": _label_count(exclusive),
            "rows": _rows_from_annotation(annotation),
            "alternative_kind": other_kind,
            "alternative_rows": _rows_from_annotation(other) if other is not None else None,
        }
    except Exception as exc:
        _diag_extra["s0_error"] = f"{type(exc).__name__}: {exc}"


def _diag_record_probe(target, similarity, spans, labels):
    try:
        _diag_note(
            "probe",
            target=int(target) if target else None,
            centroid_similarity=_as_number(similarity),
            windows_embedded=len(spans) if spans is not None else None,
            clusters=len(set(int(value) for value in labels)) if labels is not None else None,
        )
    except Exception:
        pass


def _count_words(segments):
    total = tagged = 0
    for segment in segments:
        for word in segment.get("words") or []:
            total += 1
            if word.get("speaker"):
                tagged += 1
    return total, tagged


def _binding_stats(segments, rows):
    """Measured on the final word tags against the S1 rows. Read-only."""
    if not rows:
        return {}
    starts = np.array([row["start"] for row in rows], dtype=np.float64)
    ends = np.array([row["end"] for row in rows], dtype=np.float64)
    names = [row["speaker"] for row in rows]
    per_speaker = {}
    untagged = no_timestamp = overlapping = no_overlap = differs_from_max_overlap = 0
    for segment in segments:
        for word in segment.get("words") or []:
            speaker = word.get("speaker")
            if not speaker:
                untagged += 1
                continue
            per_speaker[str(speaker)] = per_speaker.get(str(speaker), 0) + 1
            if word.get("start") is None:
                no_timestamp += 1
                continue
            start = float(word["start"])
            end = float(word["end"]) if word.get("end") is not None else start
            overlap = np.minimum(end, ends) - np.maximum(start, starts)
            best = int(np.argmax(overlap))
            if float(overlap[best]) > 0.0:
                overlapping += 1
                if names[best] != speaker:
                    differs_from_max_overlap += 1
            else:
                no_overlap += 1
    return {
        "words_per_speaker": per_speaker,
        "untagged_final": untagged,
        "tagged_without_timestamp": no_timestamp,
        "tagged_overlapping_a_row": overlapping,
        "tagged_without_any_overlap_nearest_assigned": no_overlap,
        "tag_differs_from_max_overlap_row": differs_from_max_overlap,
    }


def _capped_rows(rows):
    rows = rows or []
    return {"total": len(rows), "truncated": len(rows) > ROWS_CAP, "rows": rows[:ROWS_CAP]}


def _diag_replacement(rescue, s0_labels, s1_labels):
    trigger = _diag_extra.get("trigger") or {}
    probe = _diag_extra.get("probe") or {}
    if rescue == "none":
        replaced = False
        within = _diag_extra.get("within_split") or {}
        if within.get("status") == "ran":
            code = "pyannote_kept_within_speaker_same_voice"
            reason = (
                "Every Pyannote speaker was re-embedded on its own windows. "
                "No label contained two voices far enough to split, so Pyannote rows were kept."
            )
        elif within.get("status") == "failed":
            code = "pipeline_rows_kept_within_split_failed"
            reason = f"Within-speaker embedding split failed: {within.get('reason')}"
        elif not trigger.get("probe_ran"):
            code = "pipeline_rows_kept_probe_not_triggered"
            reason = "Pyannote returned at least the requested/minimum speakers; _probe_speakers was not run."
        elif probe.get("status") == "failed":
            code = "pipeline_rows_kept_probe_failed"
            reason = f"_probe_speakers failed: {probe.get('reason')}"
        else:
            code = "pipeline_rows_kept_probe_found_no_larger_split"
            reason = "_probe_speakers ran but did not find more speakers than Pyannote; Pyannote rows kept."
    elif rescue == "split_within":
        replaced = True
        code = "within_speaker_embedding_split"
        within = _diag_extra.get("within_split") or {}
        reason = (
            "Each Pyannote speaker was embedded on its own. A label was split only where its windows "
            f"were not the same voice (centroid cosine <= {DIAR_WITHIN_MAX_SIMILARITY}). splits={within.get('splits')}."
        )
    elif rescue == "num_speakers":
        replaced = True
        code = "pipeline_recount_with_forced_speakers"
        reason = f"Pyannote was called a second time forcing {probe.get('retry_target')} speakers; S1 is that second call's annotation."
    elif rescue == "windows":
        replaced = True
        code = "window_labels_from_embedding_clustering"
        reason = "The recount returned <2 speakers; S1 is built from embedding windows (_annotation_from_spans). Gaps between windows are not filled."
    else:
        replaced = None
        code = "unknown_rescue_value"
        reason = f"unrecognised rescue value: {rescue}"
    return {
        "replaced_pyannote_rows": replaced,
        "reason_code": code,
        "reason": reason,
        "s0_labels": s0_labels,
        "s1_labels": s1_labels,
        "s1_labels_differ_from_s0": (s0_labels != s1_labels) if s0_labels is not None else None,
    }


def _diag_warnings(rescue, speakers):
    extra = _diag_extra
    probe = extra.get("probe") or {}
    hook = _diag_runs[0] if _diag_runs else {}
    found = []
    if extra.get("audio_cleanup_error"):
        found.append(f"audio cleanup for diarization was skipped: {extra.get('audio_cleanup_error')}")
    if hook.get("clustering_input") is not None and hook.get("clustering_input") < 2:
        found.append("fewer than 2 embeddings reached Pyannote's clustering; Pyannote can only return one speaker")
    if hook.get("embeddings_nan"):
        found.append(f"{hook.get('embeddings_nan')} Pyannote embedding rows were NaN")
    if probe.get("status") == "failed":
        found.append(f"_probe_speakers failed: {probe.get('reason')}")
    before = probe.get("windows_before_cap")
    embedded = probe.get("windows_embedded")
    cap = probe.get("window_cap")
    if before is not None and cap is not None and before > cap:
        found.append(
            f"probe windows were sub-sampled {before} -> {cap}; "
            "S1 rows (only when rescue == 'windows') cover just the sampled windows and leave gaps"
        )
    if before is not None and embedded is not None and embedded < min(before, cap or before):
        found.append(f"{min(before, cap or before) - embedded} probe windows were dropped before clustering (too short, quiet or non-finite)")
    s0 = extra.get("s0") or {}
    if rescue != "none" and s0.get("labels") is not None and s0.get("labels") != len(speakers):
        found.append(f"S1 differs from Pyannote: {s0.get('labels')} -> {len(speakers)} speakers (rescue={rescue})")
    if s0.get("selected") == "exclusive":
        found.append("S0 uses Pyannote's exclusive annotation (overlap removed); the inclusive one is in s0.alternative")
    return found


def _build_diarization_debug(rescue, speakers, segments, rows, job_id=None, duration_s=None):
    extra = _diag_extra
    fingerprint = hashlib.sha256(str(job_id).encode("utf-8")).hexdigest()[:12] if job_id not in (None, "") else None
    s0 = extra.get("s0") or {}
    words_total, words_tagged = _count_words(segments)
    binding = {
        "status": "ok" if words_total else "no_word_level_entries",
        "words_total": words_total,
        "tagged_final": words_tagged,
    }
    try:
        binding.update(_binding_stats(segments, rows))
    except Exception as exc:
        binding["stats_error"] = f"{type(exc).__name__}: {exc}"
    probe = dict(extra.get("probe") or {})
    return {
        "schema_version": 1,
        "status": "ok",
        "worker_code": {"base_commit": DIAG_BASE_COMMIT, "diag_patch": DIAG_PATCH_VERSION},
        "computed_in_this_job": True,
        "job_fingerprint_sha256_12": fingerprint,
        "audio_duration_s": duration_s,
        "replacement": _diag_replacement(rescue, s0.get("labels"), len(speakers)),
        "warnings": _diag_warnings(rescue, speakers),
        "models": {
            "diarization": DIARIZATION_MODEL,
            "asr": WHISPER_MODEL,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
        },
        "libraries": _library_versions(),
        "requested_bounds": extra.get("bounds"),
        "effective_settings": extra.get("effective_settings"),
        "configured_constants": {
            "threshold": DIAR_THRESHOLD,
            "Fa": DIAR_FA,
            "Fb": DIAR_FB,
            "min_active_ratio": DIAR_MIN_ACTIVE_RATIO,
            "min_duration_off": DIAR_MIN_DURATION_OFF,
            "split_distance_euclidean": DIAR_SPLIT_DISTANCE,
            "max_split_similarity": DIAR_MAX_SPLIT_SIMILARITY,
            "within_split_distance_euclidean": DIAR_WITHIN_SPLIT_DISTANCE,
            "within_max_similarity": DIAR_WITHIN_MAX_SIMILARITY,
            "within_window_s": DIAR_WITHIN_WINDOW_S,
            "within_hop_s": DIAR_WITHIN_HOP_S,
            "within_min_speaker_s": DIAR_WITHIN_MIN_SPEAKER_S,
        },
        "audio_prep": extra.get("audio_prep"),
        "trigger": extra.get("trigger"),
        "word_voice": extra.get("word_voice"),
        "s0": {
            "selected": s0.get("selected"),
            "labels": s0.get("labels"),
            "inclusive_labels": s0.get("inclusive_labels"),
            "exclusive_labels": s0.get("exclusive_labels"),
            "rows_key": "diarization_raw",
            "alternative": {
                "kind": s0.get("alternative_kind"),
                "available": s0.get("alternative_rows") is not None,
                **_capped_rows(s0.get("alternative_rows")),
            },
        },
        "s1": {
            "rescue": rescue,
            "speakers": len(speakers),
            "speaker_labels": list(speakers),
            "rows_key": "diarization",
            "probe": probe,
            "word_binding": binding,
        },
        "notes": {
            "diarization_raw": "S0: rows of the annotation _best_annotation selected from the first Pyannote call (rows shorter than 0.05 s dropped, as for S1).",
            "diarization": "S1: final rows. Equal to S0 when s1.rescue == 'none'.",
            "exclusive_diarization": "Legacy key; an exact copy of 'diarization', not Pyannote's exclusive annotation.",
            "timings.diarize_debug[].overlap_s": (
                "Pyannote's own estimate: seconds where the pre-clustering speaker-count track is >= 2 "
                "(sum(count >= 2) * speaker_count_step_s). Not derived from the output rows."
            ),
            "timings.diarize_debug": "One entry per Pyannote call; a second entry exists only when rescue == 'num_speakers'.",
        },
    }


def _diag_output_fields(rescue, speakers, segments, rows, job_id, duration_s):
    fields = {}
    try:
        s0 = _diag_extra.get("s0")
        if s0 is not None:
            fields["diarization_raw"] = list(s0.get("rows") or [])
        fields["diarization_debug"] = _build_diarization_debug(
            rescue, speakers, segments, rows, job_id=job_id, duration_s=duration_s
        )
    except Exception as exc:  # diagnostics must never break a transcription
        print(f"[diarize] debug payload failed: {type(exc).__name__}: {exc}", flush=True)
        fields["diarization_debug"] = {
            "schema_version": 1,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return fields


def _fill_word_speakers(segments, rows):
    if not rows:
        return segments
    for segment in segments:
        for word in segment.get("words") or []:
            if word.get("speaker") or word.get("start") is None:
                continue
            start = float(word["start"])
            end = float(word["end"]) if word.get("end") is not None else start
            mid = (start + end) / 2.0
            best = None
            best_overlap = 0.0
            nearest = None
            nearest_distance = 1e9
            for row in rows:
                overlap = min(end, row["end"]) - max(start, row["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = row["speaker"]
                if row["start"] <= mid <= row["end"]:
                    distance = 0.0
                else:
                    distance = min(abs(mid - row["start"]), abs(mid - row["end"]))
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = row["speaker"]
            chosen = best or nearest
            if chosen:
                word["speaker"] = chosen
    return segments


def _split_segments_by_speaker(segments):
    split = []
    for segment in segments:
        words = list(segment.get("words") or [])
        tagged = [word.get("speaker") for word in words if word.get("speaker")]
        if len(set(tagged)) <= 1:
            updated = dict(segment)
            if tagged:
                updated["speaker"] = tagged[0]
            split.append(updated)
            continue
        group = []
        group_speaker = None

        def flush():
            if not group:
                return
            start = next((word.get("start") for word in group if word.get("start") is not None), segment.get("start"))
            end = next((word.get("end") for word in reversed(group) if word.get("end") is not None), segment.get("end"))
            text = " ".join(
                str(word.get("word") or word.get("text") or "").strip() for word in group
            ).strip()
            piece = dict(segment)
            piece["start"] = start
            piece["end"] = end
            piece["text"] = text or segment.get("text", "")
            piece["speaker"] = group_speaker
            piece["words"] = list(group)
            split.append(piece)

        for word in words:
            speaker = word.get("speaker") or group_speaker
            if group and speaker != group_speaker:
                flush()
                group = []
                group_speaker = speaker
            if not group:
                group_speaker = speaker
            group.append(word)
        flush()
    return split


def _collect_speakers(segments, rows):
    found = set()
    for row in rows:
        if row.get("speaker"):
            found.add(str(row["speaker"]))
    for segment in segments:
        if segment.get("speaker"):
            found.add(str(segment["speaker"]))
        for word in segment.get("words") or []:
            if word.get("speaker"):
                found.add(str(word["speaker"]))
    return sorted(found)


def _bind_words(segments, rows):
    """Keep a word with the speaker who started it.

    whisperx's max-overlap rule gives the last words of a turn to the next person
    when the timestamp spills a little past the cut. A word moves to the other
    speaker only when that speaker clearly owns most of the word. A word sitting
    in a short gap stays with the speaker who just finished, not the one about to start.
    """
    if not rows:
        return segments
    previous_speaker = None
    previous_end = None
    for segment in segments:
        for word in segment.get("words") or []:
            if word.get("start") is None:
                continue
            start = float(word["start"])
            end = float(word["end"]) if word.get("end") is not None else start
            if end < start:
                end = start
            duration = max(end - start, 0.05)
            overlaps = []
            starters = []
            for row in rows:
                overlap = min(end, float(row["end"])) - max(start, float(row["start"]))
                if overlap > 0:
                    overlaps.append((overlap, row))
                if float(row["start"]) <= start < float(row["end"]):
                    starters.append(row)
            chosen = None
            if starters:
                starter = max(
                    starters,
                    key=lambda row: min(end, float(row["end"])) - max(start, float(row["start"])),
                )
                starter_overlap = min(end, float(starter["end"])) - max(start, float(starter["start"]))
                chosen = starter["speaker"]
                if overlaps:
                    best_overlap, best_row = max(overlaps, key=lambda item: item[0])
                    if (
                        best_row["speaker"] != chosen
                        and best_overlap >= starter_overlap + 0.18
                        and best_overlap >= 0.55 * duration
                    ):
                        chosen = best_row["speaker"]
            elif overlaps:
                best_overlap, best_row = max(overlaps, key=lambda item: item[0])
                if best_overlap >= 0.35 * duration:
                    chosen = best_row["speaker"]
            if (
                chosen is None
                and previous_speaker is not None
                and previous_end is not None
                and 0.0 <= start - previous_end <= 0.40
            ):
                chosen = previous_speaker
            if chosen:
                word["speaker"] = str(chosen)
                previous_speaker = str(chosen)
                previous_end = end
            elif previous_end is not None and start - previous_end > 0.40:
                previous_speaker = None
                previous_end = None
    return segments


def _rows_from_word_speakers(segments):
    rows = []
    for segment in segments or []:
        for word in segment.get("words") or []:
            speaker = word.get("speaker")
            if not speaker or word.get("start") is None:
                continue
            start = float(word["start"])
            end = float(word["end"]) if word.get("end") is not None else start
            if end < start:
                end = start
            if rows and rows[-1]["speaker"] == speaker and start <= rows[-1]["end"] + 0.25:
                rows[-1]["end"] = round(max(rows[-1]["end"], end), 3)
            else:
                rows.append({"start": round(start, 3), "end": round(end, 3), "speaker": str(speaker)})
    return rows


def _next_speaker_name(used):
    number = 0
    while True:
        candidate = f"SPEAKER_{number:02d}"
        number += 1
        if candidate not in used:
            used.add(candidate)
            return candidate


def _robust_center(rows):
    """Centroid of the words that agree with each other, so one foreign word cannot drag it."""
    matrix = _normalize_rows(np.vstack(rows))
    if len(matrix) == 1:
        return matrix[0]
    center = matrix.mean(axis=0)
    center = center / max(float(np.linalg.norm(center)), 1e-8)
    scores = matrix.dot(center)
    keep = matrix[scores >= float(np.median(scores))]
    if len(keep) == 0:
        keep = matrix
    center = keep.mean(axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-8)


def _word_clip(audio, start_s, end_s, target_s):
    """Audio of this word only. Short words are repeated, never padded with silence.

    Silence padding makes one voice look like several prints. Repeating the word
    keeps its spectrum and gives the embedding model a stable length.
    """
    start = max(0, int(float(start_s) * SAMPLE_RATE))
    end = min(len(audio), int(float(end_s) * SAMPLE_RATE))
    if end <= start:
        return None
    clip = np.ascontiguousarray(audio[start:end], dtype=np.float32)
    minimum = int(0.30 * SAMPLE_RATE)
    if clip.size < minimum:
        return None
    target = int(target_s * SAMPLE_RATE)
    if clip.size >= target:
        extra = clip.size - target
        offset = extra // 2
        clip = clip[offset:offset + target]
    else:
        repeats = int(np.ceil(target / float(clip.size)))
        clip = np.tile(clip, repeats)[:target]
    rms = float(np.sqrt(np.mean(np.square(clip.astype(np.float64)))))
    if rms < 0.008:
        return None
    return clip


def _embed_word_clips(embedding_model, clips):
    win = clips[0].shape[0]
    batch = []
    vectors = []

    def flush():
        if not batch:
            return
        stacked = np.stack(batch).astype(np.float32, copy=False)
        tensor = torch.from_numpy(stacked[:, None, :])
        embedded = np.asarray(embedding_model(tensor), dtype=np.float32)
        if embedded.ndim == 1:
            embedded = embedded.reshape(1, -1)
        vectors.append(embedded)
        batch.clear()

    for clip in clips:
        if clip.shape[0] != win:
            clip = clip[:win] if clip.shape[0] > win else np.pad(clip, (0, win - clip.shape[0]), mode="edge")
        batch.append(clip)
        if len(batch) >= 8:
            flush()
    flush()
    if not vectors:
        return None
    return np.vstack(vectors)


def _word_duration(word):
    start = float(word["start"])
    end = float(word["end"]) if word.get("end") is not None else start
    return start, max(end, start), max(0.0, max(end, start) - start)


def _indexes_duration(words, indexes):
    return float(sum(_word_duration(words[index])[2] for index in indexes))


def _mean_cosine(matrix, indexes, center):
    if not indexes:
        return 0.0
    return float(np.mean(matrix[list(indexes)].dot(center)))


def _retie_words_by_voice(embedding_model, audio, segments):
    """Give every word the nearest sustained voice print.

    A print is opened from a run of words that agree with each other and sit
    away from prints already found. One odd word, or a shout shorter than the
    run, stays with the nearest known print. Words too short to measure follow
    the nearest measured word. The previous label is not evidence.
    """
    if embedding_model is None or audio is None or len(audio) == 0:
        return {"status": "skipped", "reason": "no_embedding_model"}
    words = []
    reliable = []
    clips = []
    for segment in segments:
        for word in segment.get("words") or []:
            if word.get("start") is None or not word.get("speaker"):
                continue
            words.append(word)
            start, end, _duration = _word_duration(word)
            clip = _word_clip(audio, start, end, DIAR_WORD_WINDOW_S)
            if clip is None:
                continue
            reliable.append(len(words) - 1)
            clips.append(clip)
    if len(reliable) < 4:
        return {"status": "skipped", "reason": "too_few_words", "words": len(reliable)}
    embedded = _embed_word_clips(embedding_model, clips)
    if embedded is None or len(embedded) != len(reliable):
        return {"status": "skipped", "reason": "embedding_failed", "words": len(reliable)}
    finite = np.all(np.isfinite(embedded), axis=1)
    if not np.all(finite):
        reliable = [index for index, ok in zip(reliable, finite) if ok]
        embedded = embedded[finite]
    if len(reliable) < 4:
        return {"status": "skipped", "reason": "too_few_words", "words": len(reliable)}
    original = [str(words[index].get("speaker")) for index in reliable]
    matrix = _normalize_rows(embedded)
    labels, _similarity = _cluster_embeddings(
        matrix, DIAR_PRINT_CLUSTER_DISTANCE, min_clusters=1, max_clusters=min(16, len(reliable))
    )
    buckets = {}
    for label, index in zip(_compact_labels(labels), range(len(reliable))):
        buckets.setdefault(int(label), []).append(index)
    order = sorted(buckets.values(), key=lambda indexes: _indexes_duration(words, [reliable[i] for i in indexes]), reverse=True)
    prints = []

    def nearest_print(center):
        if not prints:
            return None, -1.0
        scores = [float(np.dot(center, item["center"])) for item in prints]
        best = int(np.argmax(scores))
        return best, scores[best]

    def one_run(indexes):
        spans = sorted(_word_duration(words[reliable[index]])[:2] for index in indexes)
        for (_start, end), (nxt, _end) in zip(spans, spans[1:]):
            if nxt - end > 0.60:
                return False
        return True

    def accept(indexes):
        groups = [indexes]
        split = _bisect_reliable(matrix, words, reliable, indexes)
        if split:
            groups = split
        for group in groups:
            if len(group) < 2:
                continue
            duration = _indexes_duration(words, [reliable[index] for index in group])
            if duration < 0.70:
                continue
            center = _robust_center(matrix[group])
            own = _mean_cosine(matrix, group, center)
            host, score = nearest_print(center)
            if host is None:
                prints.append({"center": center, "seed": list(group)})
                continue
            if score >= DIAR_PRINT_JOIN:
                continue
            margin = own - _mean_cosine(matrix, group, prints[host]["center"])
            # Two turns of a new voice may have a pause between them.
            # Two scattered blips under a second do not.
            unmistakable = score <= DIAR_PRINT_OBVIOUS and len(group) >= 2 and (
                (one_run(group) and duration + 1e-3 >= 0.80)
                or duration + 1e-3 >= 1.0
            )
            clear = (
                score <= DIAR_PRINT_FAR
                and duration + 1e-3 >= DIAR_PRINT_MIN_SECONDS
                and len(group) >= DIAR_PRINT_MIN_WORDS
            )
            border = (
                score <= DIAR_PRINT_BORDER
                and margin >= DIAR_PRINT_MARGIN
                and own >= DIAR_PRINT_SELF
                and duration + 1e-3 >= 1.2
                and len(group) >= DIAR_PRINT_MIN_WORDS
            )
            if unmistakable or clear or border:
                prints.append({"center": center, "seed": list(group)})

    for indexes in order:
        accept(indexes)
    if not prints:
        return {"status": "skipped", "reason": "no_print", "words": len(reliable)}
    merged = []
    for item in prints:
        host = None
        for existing in merged:
            if float(np.dot(item["center"], existing["center"])) >= 0.90:
                host = existing
                break
        if host is None:
            merged.append(item)
    prints = merged
    centers = [item["center"] for item in prints]
    bank = np.vstack(centers)
    choice = [int(np.argmax(bank.dot(row))) for row in matrix]
    # Pull a single short word back when both neighbours are the same print
    # and this word is not a clear match to somebody else.
    order_index = sorted(range(len(reliable)), key=lambda index: float(words[reliable[index]]["start"]))
    position = {index: place for place, index in enumerate(order_index)}
    for index in order_index:
        _start, _end, duration = _word_duration(words[reliable[index]])
        if duration >= 0.55:
            continue
        place = position[index]
        neighbours = []
        if place > 0:
            neighbours.append(order_index[place - 1])
        if place + 1 < len(order_index):
            neighbours.append(order_index[place + 1])
        if len(neighbours) < 2 or choice[neighbours[0]] != choice[neighbours[1]]:
            continue
        if choice[neighbours[0]] == choice[index]:
            continue
        mine = float(np.dot(matrix[index], centers[choice[index]]))
        other = float(np.dot(matrix[index], centers[choice[neighbours[0]]]))
        if mine >= 0.88 and mine >= other + 0.10:
            continue
        if other >= mine - 0.08:
            choice[index] = choice[neighbours[0]]
    first_at = []
    for print_index in range(len(centers)):
        times = [
            float(words[reliable[index]]["start"])
            for index, chosen in enumerate(choice)
            if chosen == print_index
        ]
        first_at.append(min(times) if times else 1e9)
    ranking = sorted(range(len(centers)), key=lambda print_index: (first_at[print_index], print_index))
    names = {print_index: f"SPEAKER_{place:02d}" for place, print_index in enumerate(ranking)}
    for index, chosen in enumerate(choice):
        words[reliable[index]]["speaker"] = names[chosen]
    measured = []
    for index, chosen in enumerate(choice):
        start, end, _duration = _word_duration(words[reliable[index]])
        measured.append((start, end, names[chosen]))
    measured.sort()
    reliable_ids = set(reliable)
    for word_index, word in enumerate(words):
        if word_index in reliable_ids:
            continue
        start, end, _duration = _word_duration(word)
        previous = None
        following = None
        for sample_start, sample_end, name in measured:
            if sample_end <= start + 0.02:
                gap = start - sample_end
                if previous is None or gap < previous[0]:
                    previous = (gap, name)
            elif sample_start >= end - 0.02:
                gap = sample_start - end
                if following is None or gap < following[0]:
                    following = (gap, name)
        if previous is not None and previous[0] <= 0.50:
            word["speaker"] = previous[1]
        elif following is not None and following[0] <= 0.30:
            word["speaker"] = following[1]
        elif previous is not None and previous[0] <= 0.80:
            word["speaker"] = previous[1]
    final_names = {str(words[index].get("speaker")) for index in reliable}
    moved = sum(
        1 for index, name in enumerate(original) if str(words[reliable[index]].get("speaker")) != name
    )
    return {
        "status": "ran",
        "words": len(reliable),
        "splits": int(max(0, len(final_names) - len(set(original)))),
        "moved": int(moved),
        "prints": len(final_names),
        "match_min_cosine": DIAR_PRINT_JOIN,
        "new_speaker_max_cosine": DIAR_PRINT_BORDER,
    }


def _bisect_reliable(matrix, words, reliable, indexes):
    """indexes are positions in the reliable matrix, not in words."""
    if len(indexes) < 6:
        return None
    local_words = [words[reliable[index]] for index in indexes]
    if _indexes_duration(local_words, range(len(local_words))) < 2.2:
        return None
    labels = _kmeans_labels(matrix[list(indexes)], 2)
    sides = {}
    for index, label in zip(indexes, labels):
        sides.setdefault(int(label), []).append(index)
    if len(sides) < 2:
        return None
    groups = list(sides.values())
    if any(len(group) < DIAR_PRINT_MIN_WORDS or _indexes_duration(words, [reliable[i] for i in group]) < DIAR_PRINT_MIN_SECONDS for group in groups):
        return None
    centers = [_robust_center(matrix[group]) for group in groups]
    if float(np.dot(centers[0], centers[1])) > DIAR_PRINT_BORDER:
        return None
    for group, center, other in ((groups[0], centers[0], centers[1]), (groups[1], centers[1], centers[0])):
        own = _mean_cosine(matrix, group, center)
        away = _mean_cosine(matrix, group, other)
        if own < DIAR_PRINT_SELF or own - away < DIAR_PRINT_MARGIN:
            return None
    return groups


def _assign_speakers(result, annotation, audio=None, embedding_model=None):
    rows = _rows_from_annotation(annotation)
    if not rows or not result.get("segments"):
        return result, rows
    segments = _bind_words(result.get("segments") or [], rows)
    if audio is not None and embedding_model is not None:
        try:
            _diag_note("word_voice", **_retie_words_by_voice(embedding_model, audio, segments))
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            print(f"[diarize] word voice skipped: {type(exc).__name__}: {exc}", flush=True)
            _diag_note("word_voice", status="failed", reason=f"{type(exc).__name__}: {exc}")
    voiced = _rows_from_word_speakers(segments)
    if voiced:
        rows = voiced
    result["segments"] = _split_segments_by_speaker(segments)
    return result, rows


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
            audio, audio_info = _load_mono_16k(_materialize_audio(source, workdir))
            timings["load_audio_s"] = round(time.time() - t, 2)
            timings["audio"] = audio_info
        duration_s = round(len(audio) / SAMPLE_RATE, 2)

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
        diarization_rows = []
        diag_fields = {}
        if diarization and not result["segments"]:
            diag_fields = {
                "diarization_debug": {
                    "schema_version": 1,
                    "status": "skipped",
                    "reason": "transcription returned no segments",
                }
            }
        if diarization and result["segments"]:
            t = time.time()
            pipeline = _get_diarize_pipeline(hf_token)
            annotation, rescue = _separate_speakers(
                pipeline,
                audio,
                num_speakers=_optional_int(job_input, "num_speakers"),
                min_speakers=_optional_int(job_input, "min_speakers"),
                max_speakers=_optional_int(job_input, "max_speakers"),
            )
            prepared = _prepare_diarization_audio(audio)
            embedding_model = getattr(getattr(pipeline, "model", None), "_embedding", None)
            result, diarization_rows = _assign_speakers(result, annotation, prepared, embedding_model)
            speakers = _collect_speakers(result.get("segments") or [], diarization_rows)
            diag_fields = _diag_output_fields(
                rescue, speakers, result.get("segments") or [], diarization_rows, job.get("id"), duration_s
            )
            timings["diarize_s"] = round(time.time() - t, 2)
            timings["diarize_rescue"] = rescue
            timings["diarize_speakers"] = len(speakers)
            timings["diarize_threshold"] = DIAR_THRESHOLD
            timings["diarize_fb"] = DIAR_FB
            timings["diarize_active_ratio"] = DIAR_MIN_ACTIVE_RATIO
            timings["diarize_debug"] = list(_diag_runs)

        segments = result.get("segments", [])
        return _clean(
            {
                "segments": segments,
                "text": " ".join(s.get("text", "").strip() for s in segments).strip(),
                "detected_language": detected_language,
                "language_probability": None,
                "speakers": speakers,
                "diarization": diarization_rows,
                "exclusive_diarization": diarization_rows,
                "duration_s": duration_s,
                "timings": timings,
                "model": WHISPER_MODEL,
                **diag_fields,
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

import base64
import gc
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

# Speaker identity is cosine distance in embedding space only. No pause, duration,
# or temporal smoothing is used to merge or split voices.
# community-1 VBx AHC uses euclidean distance on unit vectors: 0.12 ≈ cosine 0.993.
DIAR_THRESHOLD = float(os.environ.get("DIARIZATION_THRESHOLD", "0.12"))
DIAR_FA = float(os.environ.get("DIARIZATION_FA", "0.50"))
DIAR_FB = float(os.environ.get("DIARIZATION_FB", "0.01"))
DIAR_MIN_DURATION_OFF = 0.0
DIAR_MIN_ACTIVE_RATIO = float(os.environ.get("DIARIZATION_MIN_ACTIVE_RATIO", "0.01"))
# sklearn cosine distance = 1 - cosine similarity. 0.05 keeps a cluster only when
# embeddings are almost the same vector; anything looser becomes a new SPEAKER.
DIAR_COSINE_DISTANCE = float(os.environ.get("DIARIZATION_COSINE_DISTANCE", "0.05"))
DIAR_WINDOW_S = float(os.environ.get("DIARIZATION_WINDOW_S", "0.50"))
DIAR_HOP_S = float(os.environ.get("DIARIZATION_HOP_S", "0.25"))
DIAR_MIN_EMBED_S = float(os.environ.get("DIARIZATION_MIN_EMBED_S", "0.40"))
DIAR_MAX_WINDOWS = int(os.environ.get("DIARIZATION_MAX_WINDOWS", "240"))
DIAR_SEGMENTATION_STEP = float(os.environ.get("DIARIZATION_SEGMENTATION_STEP", "0.04"))
DIAR_MAX_AUTO_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_AUTO_SPEAKERS", "8"))
DIAR_DEFAULT_MIN_SPEAKERS = int(os.environ.get("DIARIZATION_MIN_SPEAKERS", "1"))
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


def _cluster_embeddings(matrix, distance_threshold, min_clusters=1, max_clusters=8):
    """Agglomerative clustering on speaker embeddings only (cosine, complete linkage).

    A new speaker is created as soon as cosine distance exceeds the threshold.
    Time, duration, and pauses are not used.
    """
    from sklearn.cluster import AgglomerativeClustering

    matrix = _normalize_rows(matrix)
    count = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if count == 0:
        return np.zeros(0, dtype=int), 1.0
    max_clusters = int(max(1, min(int(max_clusters), count)))
    min_clusters = int(max(1, min(int(min_clusters), max_clusters)))
    if count == 1 or max_clusters == 1:
        return np.zeros(count, dtype=int), 1.0

    kwargs = dict(n_clusters=None, linkage="complete", distance_threshold=float(distance_threshold))
    try:
        model = AgglomerativeClustering(metric="cosine", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="cosine", **kwargs)
    labels = _compact_labels(model.fit_predict(matrix).astype(int))
    if len(set(int(value) for value in labels)) > max_clusters:
        labels = _merge_to_max(matrix, labels, max_clusters)
    return _compact_labels(labels), _min_centroid_cosine(matrix, labels)


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


def _install_fine_segmentation(model, step_ratio):
    """Shorter hops between pyannote windows so embeddings track fast speaker changes."""
    model.segmentation_step = step_ratio
    inference = getattr(model, "_segmentation", None)
    duration = getattr(inference, "duration", None)
    if inference is None or not duration:
        return
    hop = float(duration) * float(step_ratio)
    if hasattr(inference, "step"):
        inference.step = hop
    window = getattr(inference, "sliding_window", None)
    if window is not None and hasattr(window, "step"):
        try:
            window.step = hop
        except Exception:
            pass
    print(f"[diarize] embedding windows duration={duration:.2f}s hop={hop:.2f}s", flush=True)


def _configure_pipeline(model):
    params = {
        "segmentation": {"min_duration_off": 0.0},
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
    _assign_param(clustering, "method", "complete")
    _assign_param(segmentation, "min_duration_off", 0.0)
    _assign_param(segmentation, "min_duration_on", 0.0)
    _install_active_ratio(clustering, DIAR_MIN_ACTIVE_RATIO)
    _install_fine_segmentation(model, DIAR_SEGMENTATION_STEP)
    print(
        "[diarize] embedding clustering "
        f"ahc_threshold={getattr(clustering, 'threshold', None)} "
        f"cosine_distance={DIAR_COSINE_DISTANCE} "
        f"Fa={getattr(clustering, 'Fa', None)} "
        f"Fb={getattr(clustering, 'Fb', None)} "
        f"min_duration_on=0 min_duration_off=0 "
        f"min_active_ratio={DIAR_MIN_ACTIVE_RATIO}",
        flush=True,
    )


_diag_runs = []


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
    if exclusive is None or len(exclusive) == 0:
        return inclusive
    if inclusive is None or _label_count(exclusive) >= _label_count(inclusive):
        return exclusive
    return inclusive


def _speech_spans(audio, annotation, window_s):
    win = max(int(window_s * SAMPLE_RATE), int(DIAR_MIN_EMBED_S * SAMPLE_RATE))
    hop = max(int(DIAR_HOP_S * SAMPLE_RATE), win // 2)
    minimum = int(DIAR_MIN_EMBED_S * SAMPLE_RATE)
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
        if end - start <= win:
            spans.append((start, end))
            continue
        cursor = start
        while cursor < end:
            stop = min(cursor + win, end)
            if stop - cursor >= minimum:
                spans.append((cursor, stop))
            if stop >= end:
                break
            cursor += hop
    if len(spans) > DIAR_MAX_WINDOWS:
        chosen = np.linspace(0, len(spans) - 1, DIAR_MAX_WINDOWS).astype(int)
        spans = [spans[int(index)] for index in chosen]
    return spans


def _embed_spans(embedding_model, audio, spans, window_s):
    win = int(window_s * SAMPLE_RATE)
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
        if clip.size < int(DIAR_MIN_EMBED_S * SAMPLE_RATE):
            continue
        if not np.any(np.isfinite(clip)) or float(np.max(np.abs(clip))) < 1e-6:
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
        start, end = float(start), float(end)
        # Adjacent same-embedding windows may touch; never fill a gap in time.
        if merged and merged[-1][2] == label and start <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])
    order = []
    for _start, _end, label in merged:
        if label not in order:
            order.append(label)
    names = {label: f"SPEAKER_{index:02d}" for index, label in enumerate(order)}
    annotation = Annotation()
    for index, (start, end, label) in enumerate(merged):
        if end <= start:
            continue
        annotation[Segment(start, end), index] = names[label]
    return annotation


def _probe_speakers(model, audio, annotation, num_speakers, min_speakers, max_speakers):
    embedding_model = getattr(model, "_embedding", None)
    if embedding_model is None:
        return None, 1.0, None, None
    window_s = DIAR_WINDOW_S
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

    ceiling = num_speakers or max_speakers or DIAR_MAX_AUTO_SPEAKERS
    ceiling = max(1, min(int(ceiling), len(kept), DIAR_MAX_AUTO_SPEAKERS))

    labels, similarity = _cluster_embeddings(
        matrix,
        DIAR_COSINE_DISTANCE,
        min_clusters=1,
        max_clusters=ceiling,
    )
    labels = _compact_labels(labels)
    found = len(set(int(value) for value in labels))
    print(
        f"[diarize] embedding agglomerative windows={len(kept)} speakers={found} "
        f"cosine_distance={DIAR_COSINE_DISTANCE} centroid_similarity={round(similarity, 4)}",
        flush=True,
    )
    target = found if found >= 1 else None
    return target, similarity, kept, labels


def _separate_speakers(pipeline, audio, num_speakers, min_speakers, max_speakers):
    model = pipeline.model
    _diag_runs.clear()
    num_speakers, min_speakers, max_speakers = _speaker_bounds(num_speakers, min_speakers, max_speakers)
    prepared = _prepare_diarization_audio(audio)
    _configure_pipeline(model)
    output = _call_pipeline(model, prepared, num_speakers, min_speakers, max_speakers)
    annotation = _best_annotation(output)
    found = _label_count(annotation)
    rescue = "pipeline"
    similarity = None
    try:
        _target, similarity, spans, labels = _probe_speakers(
            model, prepared, annotation, num_speakers, min_speakers, max_speakers
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        print(f"[diarize] embedding clustering skipped: {type(exc).__name__}: {exc}", flush=True)
        spans, labels = None, None
    if spans is not None and labels is not None and len(labels) > 0:
        annotation = _annotation_from_spans(spans, labels)
        rescue = "embeddings"
    final = _label_count(annotation)
    print(
        f"[diarize] speakers pipeline={found} embeddings={final} source={rescue} "
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
        if end <= start or speaker is None or speaker == "":
            continue
        rows.append({"start": round(start, 3), "end": round(end, 3), "speaker": str(speaker)})
    return rows


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


def _assign_speakers(result, annotation):
    rows = _rows_from_annotation(annotation)
    if not rows or not result.get("segments"):
        return result, rows
    frame = pd.DataFrame(rows)
    assigned = whisperx.assign_word_speakers(frame, result, fill_nearest=True)
    assigned["segments"] = _split_segments_by_speaker(
        _fill_word_speakers(assigned.get("segments") or [], rows)
    )
    return assigned, rows


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
        if diarization and result["segments"]:
            t = time.time()
            annotation, rescue = _separate_speakers(
                _get_diarize_pipeline(hf_token),
                audio,
                num_speakers=_optional_int(job_input, "num_speakers"),
                min_speakers=_optional_int(job_input, "min_speakers"),
                max_speakers=_optional_int(job_input, "max_speakers"),
            )
            result, diarization_rows = _assign_speakers(result, annotation)
            speakers = _collect_speakers(result.get("segments") or [], diarization_rows)
            timings["diarize_s"] = round(time.time() - t, 2)
            timings["diarize_rescue"] = rescue
            timings["diarize_speakers"] = len(speakers)
            timings["diarize_threshold"] = DIAR_THRESHOLD
            timings["diarize_cosine_distance"] = DIAR_COSINE_DISTANCE
            timings["diarize_window_s"] = DIAR_WINDOW_S
            timings["diarize_hop_s"] = DIAR_HOP_S
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

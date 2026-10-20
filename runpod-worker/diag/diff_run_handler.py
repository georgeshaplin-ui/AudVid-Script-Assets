"""Differential run: BASE handler.py (c99803c) vs PATCHED handler.py on identical fake inputs.

Usage: python diff_run_handler.py BASE_handler.py PATCHED_handler.py

Both files are really executed (the whole module, including handler()), with GPU/model
dependencies replaced by deterministic stubs (torch, whisperx, pyannote.core, pandas, runpod,
requests). numpy, scipy and scikit-learn are the REAL libraries, so the real preprocessing,
embedding-window code and clustering of the algorithm run in both versions.

For every scenario the check is:
  * output(PATCHED) minus {diarization_raw, diarization_debug} == output(BASE), exactly
    (timings compared without wall-clock *_s values, and without the one new hook key);
  * diarization_raw (S0) == the rows of the annotation the BASE _best_annotation selects from
    the first fake Pyannote call (computed independently with the BASE module);
  * diarization (S1) is identical in both versions;
  * the fake Pyannote annotation objects were not modified by the diagnostics;
  * rescue / replacement reason codes are consistent with what actually happened.
"""
import copy
import importlib.util
import json
import os
from importlib.machinery import SourceFileLoader
import sys
import types

import numpy as np

BASE_PATH, PATCHED_PATH = sys.argv[1], sys.argv[2]
SR = 16000


# ----------------------------------------------------------------------------- stubs ---------
class FakeTensor:
    def __init__(self, array):
        self.arr = np.asarray(array)

    def reshape(self, *shape):
        return FakeTensor(self.arr.reshape(*shape))

    @property
    def shape(self):
        return self.arr.shape


class OOM(RuntimeError):
    pass


class Segment:
    def __init__(self, start, end):
        self.start, self.end = float(start), float(end)


class Annotation:
    def __init__(self):
        self._t = []

    def __setitem__(self, key, label):
        segment, track = key
        self._t.append((segment, track, label))

    def itertracks(self, yield_label=False):
        for segment, track, label in sorted(self._t, key=lambda t: (t[0].start, t[0].end, str(t[1]))):
            yield (segment, track, label) if yield_label else (segment, track)

    def labels(self):
        return sorted({label for _, _, label in self._t})

    def __len__(self):
        return len(self._t)


def snapshot(annotation):
    if annotation is None:
        return None
    return [(s.start, s.end, str(t), l) for s, t, l in annotation.itertracks(yield_label=True)]


def make_annotation(tracks):
    annotation = Annotation()
    for index, (start, end, label) in enumerate(tracks):
        annotation[Segment(start, end), index] = label
    return annotation


class PyOut:
    def __init__(self, inclusive, exclusive):
        self.speaker_diarization = inclusive
        self.exclusive_speaker_diarization = exclusive


def install_stubs():
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda array: FakeTensor(array)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, OutOfMemoryError=OOM, empty_cache=lambda: None
    )
    torch.__version__ = "stub"
    sys.modules["torch"] = torch

    def assign_word_speakers(frame, result, fill_nearest=False):
        rows = list(frame)
        for segment in result.get("segments", []):
            votes = {}
            for word in segment.get("words", []):
                if word.get("start") is None:
                    continue
                start, end = word["start"], word.get("end", word["start"])
                best, best_overlap = None, 0.0
                for row in rows:
                    overlap = min(end, row["end"]) - max(start, row["start"])
                    if overlap > best_overlap:
                        best, best_overlap = row["speaker"], overlap
                if best is None and fill_nearest and rows:
                    mid = (start + end) / 2.0
                    best = min(rows, key=lambda r: min(abs(mid - r["start"]), abs(mid - r["end"])))["speaker"]
                if best is not None:
                    word["speaker"] = best
                    votes[best] = votes.get(best, 0) + 1
            if votes:
                segment["speaker"] = max(sorted(votes), key=lambda k: votes[k])
        return result

    class ASR:
        def transcribe(self, audio, batch_size=16, language=None, task="transcribe"):
            return copy.deepcopy(ASR_RESULT)

    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = lambda *a, **k: ASR()
    whisperx.load_audio = lambda path: AUDIO.copy()
    whisperx.load_align_model = lambda language_code=None, device=None: (None, {})
    whisperx.align = lambda segments, model, metadata, audio, device, return_char_alignments=False: {
        "segments": copy.deepcopy(segments)
    }
    whisperx.assign_word_speakers = assign_word_speakers
    diarize = types.ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = object
    whisperx.diarize = diarize
    sys.modules["whisperx"] = whisperx
    sys.modules["whisperx.diarize"] = diarize

    pandas = types.ModuleType("pandas")
    class DataFrame(list):  # a real type: scikit-learn calls isinstance(x, pandas.DataFrame)
        def __init__(self, rows=()):
            super().__init__(rows)

    class Series(list):
        pass

    pandas.DataFrame, pandas.Series = DataFrame, Series
    pandas.__version__ = "2.0.0"  # scikit-learn inspects it when KMeans is used
    sys.modules["pandas"] = pandas
    sys.modules["requests"] = types.ModuleType("requests")
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(start=lambda config: None)
    sys.modules["runpod"] = runpod
    pyannote = types.ModuleType("pyannote")
    core = types.ModuleType("pyannote.core")
    core.Annotation, core.Segment = Annotation, Segment
    pyannote.core = core
    sys.modules["pyannote"] = pyannote
    sys.modules["pyannote.core"] = core


# ----------------------------------------------------------------------------- data -----------
def build_audio():
    """12 s mono: alternating 200 Hz / 600 Hz blocks of 3 s (two 'voices')."""
    t = np.arange(12 * SR) / SR
    audio = np.zeros_like(t, dtype=np.float32)
    for index in range(4):
        sl = slice(index * 3 * SR, (index + 1) * 3 * SR)
        freq = 200.0 if index % 2 == 0 else 600.0
        audio[sl] = (0.3 * np.sin(2 * np.pi * freq * t[sl])).astype(np.float32)
    rng = np.random.default_rng(0)
    return (audio + 0.002 * rng.standard_normal(audio.shape)).astype(np.float32)


def build_asr_result():
    segments = []
    for block in range(4):
        words = [
            {"word": f"w{block}_{i}", "start": block * 3.0 + i * 0.5, "end": block * 3.0 + i * 0.5 + 0.4}
            for i in range(6)
        ]
        segments.append(
            {"start": block * 3.0, "end": block * 3.0 + 3.0, "text": " ".join(w["word"] for w in words), "words": words}
        )
    return {"segments": segments, "language": "en"}


AUDIO = build_audio()
ASR_RESULT = build_asr_result()

TWO = [(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01"), (6, 9, "SPEAKER_00"), (9, 12, "SPEAKER_01")]
ONE = [(0, 12, "SPEAKER_00")]
THREE_INCL = [(0, 4, "SPEAKER_00"), (3, 8, "SPEAKER_01"), (8, 12, "SPEAKER_02")]
TWO_EXCL = [(0, 4, "SPEAKER_00"), (4, 12, "SPEAKER_01")]


class FakeEmbedding:
    def __init__(self, fail=False):
        self.fail = fail

    def __call__(self, tensor):
        if self.fail:
            raise ValueError("embedding backend exploded (test)")
        clips = tensor.arr[:, 0, :]
        out = []
        for clip in clips:
            spec = np.abs(np.fft.rfft(clip.astype(np.float64)))
            n = clip.shape[0]
            e200 = spec[int(round(200 * n / SR))]
            e600 = spec[int(round(600 * n / SR))]
            out.append([e200, e600, 1.0])
        return np.asarray(out, dtype=np.float32)


class FakeClustering:
    threshold, Fa, Fb = 0.6, 0.07, 0.8

    def filter_embeddings(self, embeddings, segmentations=None, min_active_ratio=0.2):
        return (np.zeros((4, 8)),)


class FakeSegmentation:
    min_duration_on = 0.1
    min_duration_off = 0.1


class FakeModel:
    def __init__(self, responses, embedding_fail=False):
        self.responses = responses  # list of (inclusive_tracks, exclusive_tracks) | Exception | Annotation tracks
        self.calls = []
        self.produced = []
        self.clustering = FakeClustering()
        self.segmentation = FakeSegmentation()
        self._embedding = FakeEmbedding(embedding_fail)

    def instantiate(self, params):
        return self

    def __call__(self, inputs, num_speakers=None, min_speakers=None, max_speakers=None, hook=None):
        index = len(self.calls)
        self.calls.append((num_speakers, min_speakers, max_speakers))
        response = self.responses[min(index, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        inclusive_tracks, exclusive_tracks = response
        inclusive = make_annotation(inclusive_tracks)
        exclusive = make_annotation(exclusive_tracks) if exclusive_tracks is not None else None
        self.produced.append((inclusive, exclusive, snapshot(inclusive), snapshot(exclusive)))
        if hook is not None:
            counting = types.SimpleNamespace(
                data=np.array([[1], [2], [1], [0]]), sliding_window=types.SimpleNamespace(step=0.5)
            )
            hook("speaker_counting", counting, file=None)
            hook("embeddings", np.zeros((3, 2, 4)), file=None)
            self.clustering.filter_embeddings(np.zeros((3, 2, 4)), None)
        return PyOut(inclusive, exclusive)


class FakePipeline:
    def __init__(self, model):
        self.model = model


# ----------------------------------------------------------------------------- loading --------
def load(path, name):
    loader = SourceFileLoader(name, path)  # explicit loader: works for any file extension
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    module._materialize_audio = lambda source, workdir: "stub-audio"
    module._load_mono_16k = lambda path: (AUDIO.copy(), {"decoder": "stub", "source_sr": SR, "source_channels": 1})
    return module


install_stubs()
BASE = load(BASE_PATH, "handler_base")
PATCHED = load(PATCHED_PATH, "handler_patched")


def run(module, responses, job_extra=None, embedding_fail=False):
    model = FakeModel(copy.deepcopy(responses), embedding_fail=embedding_fail)
    module._get_diarize_pipeline = lambda token: FakePipeline(model)
    # the BASE/PATCHED module keeps global per-run state; start every run from the same state
    module._diag_runs.clear()
    if hasattr(module, "_diag_extra"):
        module._diag_extra.clear()
    job_input = {"audio_file": "x", "diarization": True, "huggingface_access_token": "TEST-PLACEHOLDER"}
    job_input.update(job_extra or {})
    output = module.handler({"id": "job-test-1", "input": job_input})
    return output, model


def normalise(output, drop_diag):
    out = copy.deepcopy(output)
    if drop_diag:
        out.pop("diarization_raw", None)
        out.pop("diarization_debug", None)
    timings = out.get("timings")
    if isinstance(timings, dict):
        for key in [k for k in timings if k.endswith("_s") and not isinstance(timings[k], (list, dict))]:
            timings.pop(key)
        for entry in timings.get("diarize_debug", []) or []:
            if drop_diag:
                entry.pop("speaker_count_step_s", None)
    return json.loads(json.dumps(out, sort_keys=True, default=str))


SCENARIOS = [
    # name, responses, job_extra, expected rescue, embedding_fail, expected reason_code
    ("A_pyannote_ok_no_probe", [(TWO, TWO)], {}, "none", False, "pipeline_rows_kept_probe_not_triggered"),
    ("B_collapsed_recount_2", [(ONE, ONE), (TWO, TWO)], {}, "num_speakers", False, "pipeline_recount_with_forced_speakers"),
    ("C_collapsed_recount_1_windows", [(ONE, ONE), (ONE, ONE)], {}, "windows", False, "window_labels_from_embedding_clustering"),
    ("D_inclusive_wins", [(THREE_INCL, TWO_EXCL)], {}, "none", False, "pipeline_rows_kept_probe_not_triggered"),
    ("E_exclusive_missing", [(TWO, None)], {}, "none", False, "pipeline_rows_kept_probe_not_triggered"),
    ("F_probe_fails", [(ONE, ONE)], {}, "none", True, "pipeline_rows_kept_probe_failed"),
    ("G_forced_num_speakers", [(ONE, ONE), (TWO, TWO)], {"num_speakers": 2}, "num_speakers", False, "pipeline_recount_with_forced_speakers"),
    ("H_pyannote_raises", [RuntimeError("pyannote exploded (test)")], {}, None, False, None),
]

failures = []
ONLY = os.environ.get("ONLY")  # optional: run a single scenario by name prefix (debugging)
for name, responses, extra, expected_rescue, emb_fail, expected_code in SCENARIOS:
    if ONLY and not name.startswith(ONLY):
        continue
    base_out, base_model = run(BASE, responses, extra, emb_fail)
    new_out, new_model = run(PATCHED, responses, extra, emb_fail)

    # 1) everything the app consumed before is identical
    if normalise(base_out, False) != normalise(new_out, True):
        failures.append(f"{name}: output differs from BASE beyond the two diagnostic keys")
        print(f"[{name}] DIFFERENT")
        continue

    # 2) fake Pyannote annotations untouched by the diagnostics
    for (inc, exc, snap_inc, snap_exc) in new_model.produced:
        if snapshot(inc) != snap_inc or snapshot(exc) != snap_exc:
            failures.append(f"{name}: a Pyannote annotation object was modified")
    # and the model saw identical calls
    if base_model.calls != new_model.calls:
        failures.append(f"{name}: Pyannote call arguments differ {base_model.calls} vs {new_model.calls}")

    if "error" in base_out:
        if "diarization_raw" in new_out or "diarization_debug" in new_out:
            failures.append(f"{name}: error output must not gain diagnostic keys")
        print(f"[{name}] OK (error path identical): {base_out['error']}")
        continue

    # 3) S0 equals the rows of the annotation BASE selects from the first call
    first_inclusive, first_exclusive, _, _ = base_model.produced[0]
    expected_s0 = BASE._rows_from_annotation(BASE._best_annotation(PyOut(first_inclusive, first_exclusive)))
    if new_out.get("diarization_raw") != expected_s0:
        failures.append(f"{name}: diarization_raw != independently computed S0")
    # 4) S1 equals BASE's diarization
    if new_out["diarization"] != base_out["diarization"]:
        failures.append(f"{name}: S1 differs from BASE")
    # 5) rescue and replacement semantics
    rescue = new_out["timings"]["diarize_rescue"]
    debug = new_out["diarization_debug"]
    if rescue != expected_rescue:
        failures.append(f"{name}: unexpected rescue {rescue} (expected {expected_rescue})")
    if debug["status"] != "ok" or debug["replacement"]["reason_code"] != expected_code:
        failures.append(f"{name}: reason_code {debug['replacement']['reason_code']} (expected {expected_code})")
    if debug["replacement"]["replaced_pyannote_rows"] != (rescue != "none"):
        failures.append(f"{name}: replaced_pyannote_rows inconsistent with rescue")
    if (new_out["diarization_raw"] == new_out["diarization"]) != (rescue == "none"):
        # rows may legitimately coincide for other rescues; only 'none' is required to be equal
        if rescue == "none":
            failures.append(f"{name}: rescue 'none' but S0 != S1")
    if debug["job_fingerprint_sha256_12"] is None or debug["worker_code"]["base_commit"] != "c99803c":
        failures.append(f"{name}: fingerprint/worker_code missing")
    json.dumps(new_out)  # must serialise
    probe_info = debug["s1"]["probe"]
    print(f"    probe={json.dumps(probe_info, sort_keys=True)}")
    print(f"    warnings={debug['warnings']}")
    print(
        f"[{name}] OK rescue={rescue} S0_rows={len(new_out['diarization_raw'])} "
        f"S1_rows={len(new_out['diarization'])} speakers={new_out['speakers']} code={debug['replacement']['reason_code']}"
    )

if failures:
    print("\nDIFFERENTIAL_CHECK_FAILED")
    for message in failures:
        print(" -", message)
    sys.exit(1)
print("\nDIFFERENTIAL_CHECK_OK: patched output == base output (minus diarization_raw/diarization_debug) in all scenarios.")

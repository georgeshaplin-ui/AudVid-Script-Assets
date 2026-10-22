"""Behaviour of the within-speaker split. No GPU, no network, no real Pyannote.

The embedding stub turns a window into [energy at 200 Hz, energy at 600 Hz, 1].
Two pure tones are far apart after L2 normalisation. Loud and quiet copies of the
same tone stay close. Pyannote's own threshold constants are not changed.
"""
import copy
import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "runpod-worker/handler.py"
SR = 16000
WORDS = {"items": []}


class FakeTensor:
    def __init__(self, array):
        self.arr = np.asarray(array)

    def reshape(self, *shape):
        return FakeTensor(self.arr.reshape(*shape))

    @property
    def shape(self):
        return self.arr.shape


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


class PyOut:
    def __init__(self, inclusive, exclusive):
        self.speaker_diarization = inclusive
        self.exclusive_speaker_diarization = exclusive


def make_annotation(tracks):
    annotation = Annotation()
    for index, (start, end, label) in enumerate(tracks):
        annotation[Segment(start, end), index] = label
    return annotation


def tone(seconds, freq, amplitude=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def install_stubs(audio):
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda array: FakeTensor(array)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        OutOfMemoryError=type("OOM", (RuntimeError,), {}),
        empty_cache=lambda: None,
    )
    torch.__version__ = "stub"
    sys.modules["torch"] = torch

    def assign_word_speakers(frame, result, fill_nearest=False):
        rows = list(frame)
        for segment in result.get("segments", []):
            votes = {}
            for word in segment.get("words", []):
                start, end = word["start"], word.get("end", word["start"])
                best, best_overlap = None, 0.0
                for row in rows:
                    overlap = min(end, row["end"]) - max(start, row["start"])
                    if overlap > best_overlap:
                        best, best_overlap = row["speaker"], overlap
                if best is not None and best_overlap > 0:
                    word["speaker"] = best
                    votes[best] = votes.get(best, 0) + 1
            if votes:
                segment["speaker"] = max(sorted(votes), key=lambda k: votes[k])
        return result

    class ASR:
        def transcribe(self, audio, batch_size=16, language=None, task="transcribe"):
            words = copy.deepcopy(WORDS["items"])
            end = words[-1]["end"] if words else 0.0
            return {"segments": [{"start": 0.0, "end": end, "text": "x", "words": words}], "language": "en"}

    whisperx = types.ModuleType("whisperx")
    whisperx.load_model = lambda *a, **k: ASR()
    whisperx.load_audio = lambda path: audio.copy()
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

    class DataFrame(list):
        def __init__(self, rows=()):
            super().__init__(rows)

    class Series(list):
        pass

    pandas.DataFrame, pandas.Series = DataFrame, Series
    pandas.__version__ = "2.0.0"
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


class FakeEmbedding:
    def __init__(self, mode="tone"):
        self.mode = mode

    def __call__(self, tensor):
        clips = tensor.arr[:, 0, :]
        if self.mode == "constant":
            return np.ones((len(clips), 3), dtype=np.float32)
        out = []
        for clip in clips:
            spec = np.abs(np.fft.rfft(clip.astype(np.float64)))
            n = clip.shape[0]
            out.append([
                spec[int(round(200 * n / SR))],
                spec[int(round(600 * n / SR))],
                1.0,
            ])
        return np.asarray(out, dtype=np.float32)


class FakeClustering:
    threshold, Fa, Fb = 0.6, 0.07, 0.8

    def filter_embeddings(self, embeddings, segmentations=None, min_active_ratio=0.2):
        return (np.zeros((4, 8)),)


class FakeSegmentation:
    min_duration_on = 0.1
    min_duration_off = 0.1


class FakeModel:
    def __init__(self, responses, embedding="tone"):
        self.responses = responses
        self.calls = []
        self.clustering = FakeClustering()
        self.segmentation = FakeSegmentation()
        self._embedding = FakeEmbedding(embedding)

    def instantiate(self, params):
        return self

    def __call__(self, inputs, num_speakers=None, min_speakers=None, max_speakers=None, hook=None):
        index = len(self.calls)
        self.calls.append((num_speakers, min_speakers, max_speakers))
        inclusive_tracks, exclusive_tracks = self.responses[min(index, len(self.responses) - 1)]
        inclusive = make_annotation(inclusive_tracks)
        exclusive = make_annotation(exclusive_tracks) if exclusive_tracks is not None else None
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


def load(path, audio):
    loader = SourceFileLoader("handler_under_test", path)
    spec = importlib.util.spec_from_loader("handler_under_test", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    module._materialize_audio = lambda source, workdir: "stub-audio"
    module._load_mono_16k = lambda path: (audio.copy(), {"decoder": "stub", "source_sr": SR, "source_channels": 1})
    return module


def run(module, responses, audio_words, embedding="tone"):
    model = FakeModel(responses, embedding=embedding)
    module._get_diarize_pipeline = lambda token: FakePipeline(model)
    module._diag_runs.clear()
    module._diag_extra.clear()
    # The stub ASR returns no words; the real handler aligns whatever transcribe returned.
    # Inject words by wrapping align.
    words = audio_words
    WORDS["items"] = words

    def align(segments, model_a, metadata, audio, device, return_char_alignments=False):
        return {"segments": [{"start": 0.0, "end": words[-1]["end"], "text": "x", "words": copy.deepcopy(words)}]}

    sys.modules["whisperx"].align = align
    output = module.handler({"id": "job-split", "input": {
        "audio_file": "x", "diarization": True, "huggingface_access_token": "TEST-PLACEHOLDER",
    }})
    return output, model


def speakers_covering(rows, t):
    return {row["speaker"] for row in rows if row["start"] <= t < row["end"]}


def main():
    failures = []

    two_voices = np.concatenate([tone(3, 200), tone(3, 600), tone(3, 200), tone(3, 600)])
    one_voice = np.concatenate([tone(6, 200, 0.1), tone(6, 200, 0.8)])
    install_stubs(two_voices)
    module = load(PATH, two_voices)

    if module.DIAR_THRESHOLD != 0.35 or module.DIAR_FB != 0.02:
        failures.append(f"VBx constants changed: threshold={module.DIAR_THRESHOLD} Fb={module.DIAR_FB}")
    if module.DIAG_PATCH_VERSION < 2:
        failures.append(f"diag patch version {module.DIAG_PATCH_VERSION} does not include the split")

    inclusive = make_annotation([(0, 6, "SPEAKER_00"), (5, 12, "SPEAKER_01")])
    exclusive = make_annotation([(0, 5, "SPEAKER_00"), (5, 12, "SPEAKER_01")])
    chosen = module._best_annotation(PyOut(inclusive, exclusive))
    if chosen is not inclusive:
        failures.append("inclusive annotation was not preferred when both exist")

    words = [
        {"word": "a", "start": 0.4, "end": 0.9},
        {"word": "b", "start": 3.4, "end": 3.9},
        {"word": "c", "start": 6.4, "end": 6.9},
        {"word": "d", "start": 9.4, "end": 9.9},
    ]

    # Pyannote glued two tones into one label. The windows must come apart.
    merged = [([(0, 12, "SPEAKER_00")], [(0, 12, "SPEAKER_00")])]
    out, model = run(module, merged, words, embedding="tone")
    debug = out.get("diarization_debug") or {}
    replacement = debug.get("replacement") or {}
    labels = {row["speaker"] for row in out.get("diarization") or []}
    early = speakers_covering(out.get("diarization") or [], 0.5)
    late = speakers_covering(out.get("diarization") or [], 3.5)
    if replacement.get("reason_code") != "within_speaker_embedding_split":
        failures.append(f"merged voices were not split: {replacement}")
    if len(labels) < 2 or early == late or not early or not late:
        failures.append(f"split did not separate 200 Hz from 600 Hz: labels={labels} early={early} late={late}")
    if len(model.calls) != 1:
        failures.append(f"split called Pyannote again: {model.calls}")

    # The same tone, quiet then loud, is not split by the within-speaker check.
    # Called directly so the older "pyannote returned 1 speaker" recount does not run.
    module._load_mono_16k = lambda path: (one_voice.copy(), {"decoder": "stub", "source_sr": SR, "source_channels": 1})
    same_ann = make_annotation([(0, 12, "SPEAKER_00")])
    model_same = FakeModel(merged, embedding="tone")
    prepared = module._prepare_diarization_audio(one_voice)
    split_ann, n_splits, notes = module._split_within_speakers(model_same, prepared, same_ann)
    if n_splits != 0 or split_ann is not None:
        failures.append(f"one voice was split: splits={n_splits} notes={notes}")

    # Identical embeddings are one voice. Pyannote's two labels do not count.
    module._load_mono_16k = lambda path: (two_voices.copy(), {"decoder": "stub", "source_sr": SR, "source_channels": 1})
    kept = [([(0, 6, "SPEAKER_00"), (6, 12, "SPEAKER_01")], [(0, 6, "SPEAKER_00"), (6, 12, "SPEAKER_01")])]
    out_keep, _model = run(module, kept, words, embedding="constant")
    kept_labels = {row["speaker"] for row in out_keep.get("diarization") or []}
    kept_code = (out_keep.get("diarization_debug") or {}).get("replacement", {}).get("reason_code")
    if kept_labels != {"SPEAKER_00"} or kept_code == "within_speaker_embedding_split":
        failures.append(f"identical voice print stayed split: {kept_labels} {kept_code}")

    def level_split(low_vector):
        class LevelEmbedding:
            def __call__(self, tensor):
                clips = tensor.arr[:, 0, :]
                out = []
                for clip in clips:
                    out.append([1.0, 0.0, 0.0] if float(np.mean(clip)) > 0.15 else low_vector)
                return np.asarray(out, dtype=np.float32)

        audio = np.concatenate([
            np.full(3 * SR, 0.30, dtype=np.float32),
            np.full(3 * SR, 0.05, dtype=np.float32),
        ])
        model = types.SimpleNamespace(_embedding=LevelEmbedding())
        annotation = make_annotation([(0, 6, "SPEAKER_00")])
        return module._split_within_speakers(model, audio, annotation)

    different, different_splits, different_notes = level_split([0.0, 1.0, 0.0])
    if different_splits < 1 or different is None:
        failures.append(f"clearly different voices stayed merged: {different_notes}")
    close, close_splits, close_notes = level_split([0.85, 0.526782, 0.0])
    if close_splits != 0 or close is not None:
        failures.append(f"cosine 0.85 same-voice variation was split: splits={close_splits} notes={close_notes}")
    same_print, same_splits, same_notes = level_split([0.995, 0.100, 0.0])
    if same_splits != 0 or same_print is not None:
        failures.append(f"near-copy voice was split: splits={same_splits} notes={same_notes}")

    rows = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
    bound = module._bind_words([{
        "words": [
            {"word": "last", "start": 1.60, "end": 2.08},
            {"word": "held", "start": 1.55, "end": 1.75},
            {"word": "next", "start": 2.15, "end": 2.40},
        ]
    }], rows)
    tags = [word["speaker"] for word in bound[0]["words"]]
    if tags != ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]:
        failures.append(f"boundary words moved to the next speaker: {tags}")

    class MeanEmbedding:
        def __call__(self, tensor):
            clips = tensor.arr[:, 0, :]
            out = []
            for clip in clips:
                if float(np.mean(np.abs(clip))) > 0.15:
                    out.append([1.0, 0.0, 0.0])
                else:
                    out.append([0.20, 0.98, 0.0])
            return np.asarray(out, dtype=np.float32)

    def word(start, end, speaker):
        return {"word": "w", "start": start, "end": end, "speaker": speaker}

    swallowed_audio = np.concatenate([
        np.full(3 * SR, 0.30, dtype=np.float32),
        np.full(4 * SR, 0.05, dtype=np.float32),
    ])
    swallowed = [{"words": [
        word(0.3, 0.7, "SPEAKER_00"),
        word(0.9, 1.3, "SPEAKER_00"),
        word(1.5, 1.9, "SPEAKER_00"),
        word(3.4, 3.8, "SPEAKER_00"),
        word(4.2, 4.6, "SPEAKER_00"),
        word(5.0, 5.4, "SPEAKER_00"),
        word(5.8, 6.2, "SPEAKER_00"),
    ]}]
    swallowed_info = module._retie_words_by_voice(MeanEmbedding(), swallowed_audio, swallowed)
    host = swallowed[0]["words"][0]["speaker"]
    guest = {item["speaker"] for item in swallowed[0]["words"] if item["start"] >= 3.4}
    if swallowed_info.get("status") != "ran" or host in guest or len(guest) != 1:
        failures.append(f"swallowed voice was not recovered: {swallowed_info} host={host} guest={guest}")

    boundary_audio = np.concatenate([
        np.full(2 * SR, 0.30, dtype=np.float32),
        np.full(4 * SR, 0.05, dtype=np.float32),
    ])
    boundary = [{"words": [
        word(0.4, 0.8, "SPEAKER_00"),
        word(1.0, 1.4, "SPEAKER_00"),
        word(2.1, 2.5, "SPEAKER_00"),
        word(3.0, 3.4, "SPEAKER_01"),
        word(3.6, 4.0, "SPEAKER_01"),
        word(4.4, 4.8, "SPEAKER_01"),
    ]}]
    boundary_info = module._retie_words_by_voice(MeanEmbedding(), boundary_audio, boundary)
    if boundary[0]["words"][2]["speaker"] != "SPEAKER_01":
        failures.append(
            f"turn-start word stayed with the previous speaker: "
            f"{boundary[0]['words'][2]['speaker']} {boundary_info}"
        )

    same_audio = np.full(4 * SR, 0.30, dtype=np.float32)
    same = [{"words": [
        word(0.3, 0.7, "SPEAKER_00"),
        word(0.9, 1.3, "SPEAKER_00"),
        word(1.6, 2.0, "SPEAKER_00"),
        word(2.4, 2.8, "SPEAKER_00"),
        word(3.1, 3.5, "SPEAKER_00"),
        word(3.6, 3.7, "SPEAKER_00"),
    ]}]
    same_info = module._retie_words_by_voice(MeanEmbedding(), same_audio, same)
    same_labels = {item["speaker"] for item in same[0]["words"]}
    if same_labels != {"SPEAKER_00"} or same_info.get("splits"):
        failures.append(f"one voice was split by the word pass: {same_labels} {same_info}")
    extracted_prints = [item for item in (same_info.get("word_prints") or []) if item.get("extracted")]
    missed_prints = [item for item in (same_info.get("word_prints") or []) if not item.get("extracted")]
    if len(extracted_prints) != 5 or len(missed_prints) != 1:
        failures.append(
            f"word prints should be 5 extracted and 1 too short, got {len(extracted_prints)} and {len(missed_prints)}"
        )
    elif len(extracted_prints[0].get("vector") or []) != 3 or "SPEAKER_00" not in (extracted_prints[0].get("cosine_to_prints") or {}):
        failures.append(f"extracted word print is missing its vector: {extracted_prints[0]}")
    elif missed_prints[0].get("reason") != "too_short_or_quiet":
        failures.append(f"short word was not marked unmeasured: {missed_prints[0]}")

    class MildEmbedding:
        def __call__(self, tensor):
            clips = tensor.arr[:, 0, :]
            out = []
            for clip in clips:
                if float(np.mean(np.abs(clip))) > 0.2:
                    out.append([1.0, 0.0, 0.0])
                else:
                    out.append([0.85, 0.526782, 0.0])
            return np.asarray(out, dtype=np.float32)

    mild_audio = np.concatenate([
        np.full(3 * SR, 0.30, dtype=np.float32),
        np.full(3 * SR, 0.12, dtype=np.float32),
    ])
    mild = [{"words": [
        word(0.3, 0.8, "SPEAKER_00"),
        word(1.0, 1.5, "SPEAKER_00"),
        word(1.7, 2.2, "SPEAKER_00"),
        word(2.4, 2.9, "SPEAKER_00"),
        word(3.3, 3.8, "SPEAKER_00"),
        word(4.0, 4.5, "SPEAKER_00"),
        word(4.7, 5.2, "SPEAKER_00"),
        word(5.4, 5.9, "SPEAKER_00"),
    ]}]
    mild_info = module._retie_words_by_voice(MildEmbedding(), mild_audio, mild)
    mild_labels = {item["speaker"] for item in mild[0]["words"]}
    if mild_labels != {"SPEAKER_00"} or mild_info.get("splits"):
        failures.append(f"unchanged voice print was split: {mild_labels} {mild_info}")

    class CloseVoice:
        def __call__(self, tensor):
            clips = tensor.arr[:, 0, :]
            out = []
            for clip in clips:
                if float(np.mean(np.abs(clip))) > 0.2:
                    out.append([1.0, 0.0, 0.0])
                else:
                    out.append([0.83, 0.55776, 0.0])
            return np.asarray(out, dtype=np.float32)

    close_audio = np.concatenate([
        np.full(3 * SR, 0.30, dtype=np.float32),
        np.full(3 * SR, 0.10, dtype=np.float32),
    ])
    close_words = [{"words": [
        word(0.3, 0.8, "SPEAKER_00"),
        word(1.0, 1.5, "SPEAKER_00"),
        word(1.7, 2.2, "SPEAKER_00"),
        word(2.4, 2.9, "SPEAKER_00"),
        word(3.3, 3.8, "SPEAKER_00"),
        word(4.0, 4.5, "SPEAKER_00"),
        word(4.7, 5.2, "SPEAKER_00"),
        word(5.4, 5.9, "SPEAKER_00"),
    ]}]
    close_info = module._retie_words_by_voice(CloseVoice(), close_audio, close_words)
    close_early = {item["speaker"] for item in close_words[0]["words"] if item["start"] < 3.0}
    close_late = {item["speaker"] for item in close_words[0]["words"] if item["start"] >= 3.0}
    if close_info.get("status") != "ran" or close_early == close_late or len(close_early) != 1 or len(close_late) != 1:
        failures.append(f"sustained nearby voice was swallowed: {close_info} early={close_early} late={close_late}")

    class BlipVoice:
        def __call__(self, tensor):
            clips = tensor.arr[:, 0, :]
            out = []
            for clip in clips:
                level = float(np.mean(np.abs(clip)))
                if 0.12 < level < 0.22:
                    out.append([0.0, 1.0, 0.0])
                else:
                    out.append([1.0, 0.0, 0.0])
            return np.asarray(out, dtype=np.float32)

    blip_audio = np.concatenate([
        np.full(3 * SR, 0.30, dtype=np.float32),
        np.full(int(0.5 * SR), 0.16, dtype=np.float32),
        np.full(2 * SR, 0.30, dtype=np.float32),
    ])
    blip_words = [{"words": [
        word(0.3, 0.8, "SPEAKER_00"),
        word(1.0, 1.5, "SPEAKER_00"),
        word(1.7, 2.2, "SPEAKER_00"),
        word(3.05, 3.45, "SPEAKER_02"),
        word(3.8, 4.3, "SPEAKER_00"),
        word(4.5, 5.0, "SPEAKER_00"),
    ]}]
    blip_info = module._retie_words_by_voice(BlipVoice(), blip_audio, blip_words)
    blip_labels = [item["speaker"] for item in blip_words[0]["words"]]
    if len(set(blip_labels)) != 1 or blip_info.get("prints") != 1:
        failures.append(f"one odd word opened a speaker: {blip_labels} {blip_info}")

    if failures:
        print("WITHIN_SPLIT_FAILED")
        for message in failures:
            print(" -", message)
        sys.exit(1)
    print("WITHIN_SPLIT_OK")


if __name__ == "__main__":
    main()

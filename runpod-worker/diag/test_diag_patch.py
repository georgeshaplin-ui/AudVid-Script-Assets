"""Static + behavioural checks for the diagnostics-only patch of runpod-worker/handler.py.

Runs without GPU, models, network or RunPod: handler.py is NOT imported (importing it loads
Whisper). The file is parsed with `ast`, and only the diagnostic helpers are executed against
fake Pyannote-like objects.

Usage: python test_diag_patch.py [path/to/handler.py]
Needs: numpy only.
"""
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "runpod-worker/handler.py"
TEXT = open(SRC, encoding="utf-8").read()
TREE = ast.parse(TEXT, filename=SRC)  # 1) syntax check of the whole file

FUNCS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}

REQUIRED = {
    "_diag_note", "_diag_record_audio", "_diag_record_settings", "_diag_record_s0",
    "_diag_record_probe", "_diag_replacement", "_diag_warnings", "_diag_output_fields",
    "_build_diarization_debug", "_binding_stats", "_count_words", "_rows_from_annotation",
    "_label_count", "_clean", "_separate_speakers", "handler",
}
missing = REQUIRED - set(FUNCS)
assert not missing, f"missing functions: {sorted(missing)}"

# 2) the response must carry both diagnostic keys
assert "diarization_raw" in TEXT and "diarization_debug" in TEXT, "diagnostic keys not found"
handler_src = ast.get_source_segment(TEXT, FUNCS["handler"])
assert "**diag_fields" in handler_src, "handler does not merge diag_fields into the response"
assert '"diarization": diarization_rows' in handler_src, "legacy 'diarization' key missing"
assert '"exclusive_diarization": diarization_rows' in handler_src, "legacy 'exclusive_diarization' key missing"

# 3) the algorithm must call the diagnostics only through guarded entry points
sep_src = ast.get_source_segment(TEXT, FUNCS["_separate_speakers"])
for needle in ("_diag_record_s0(output, annotation)", "_diag_record_probe(", "_diag_note("):
    assert needle in sep_src, f"expected diagnostic call not found: {needle}"

# 4) execute only the helpers (and plain constants) in an isolated namespace
WANTED_FUNCS = REQUIRED - {"_separate_speakers", "handler"} | {
    "_as_number", "_effective_settings", "_library_versions", "_capped_rows", "_worker_identity",
    "_job_pyannote_mode", "_bounded_number", "_bounded_count", "_parse_pyannote_raw",
    "_public_raw_controls",
}
body = []
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCS:
        body.append(node)
    elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
        name = node.targets[0].id
        if name.startswith(("DIAR_", "DIARIZATION_", "DIAG_", "_RAW_")) or name in {
            "ROWS_CAP", "WHISPER_MODEL", "SAMPLE_RATE", "_diag_extra", "_diag_runs", "IMAGE_TAG",
        }:
            body.append(node)
# DEVICE / COMPUTE_TYPE are derived from torch in handler.py; fixed here (no torch needed).
ns = {
    "np": np, "math": math, "hashlib": hashlib, "importlib": importlib, "os": os,
    "socket": __import__("socket"),
    "DEVICE": "cpu", "COMPUTE_TYPE": "int8",
}
exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)


class Seg:
    def __init__(self, start, end):
        self.start, self.end = start, end


class Ann:
    def __init__(self, tracks):
        self._tracks = list(tracks)

    def itertracks(self, yield_label=False):
        for start, end, label in self._tracks:
            yield Seg(start, end), None, label

    def labels(self):
        return sorted({label for _, _, label in self._tracks})

    def __len__(self):
        return len(self._tracks)


class Out:
    def __init__(self, inclusive, exclusive):
        self.speaker_diarization = inclusive
        self.exclusive_speaker_diarization = exclusive


inclusive = Ann([(0.0, 2.0, "A"), (1.5, 3.0, "B"), (3.0, 4.0, "A")])
exclusive = Ann([(0.0, 1.5, "A"), (1.5, 3.0, "B"), (3.0, 4.0, "A")])
out = Out(inclusive, exclusive)
before_inc = list(inclusive.itertracks(yield_label=True))
before_exc = list(exclusive.itertracks(yield_label=True))
before_inc_t = [(s.start, s.end, l) for s, _, l in before_inc]
before_exc_t = [(s.start, s.end, l) for s, _, l in before_exc]

# S0 is recorded without touching the annotations
ns["_diag_extra"].clear()
ns["_diag_record_s0"](out, exclusive)
s0 = ns["_diag_extra"]["s0"]
assert s0["selected"] == "exclusive" and s0["alternative_kind"] == "inclusive", s0
assert [(r["start"], r["end"], r["speaker"]) for r in s0["rows"]] == [(0.0, 1.5, "A"), (1.5, 3.0, "B"), (3.0, 4.0, "A")]
assert [(s.start, s.end, l) for s, _, l in inclusive.itertracks(yield_label=True)] == before_inc_t
assert [(s.start, s.end, l) for s, _, l in exclusive.itertracks(yield_label=True)] == before_exc_t

# the algorithm's own rows and the S0 list are independent objects
algo_rows = ns["_rows_from_annotation"](exclusive)
assert algo_rows == s0["rows"] and algo_rows is not s0["rows"] and algo_rows[0] is not s0["rows"][0]

# the full output fields: JSON-safe, and the transcription result is not mutated
segments = [
    {"start": 0.0, "end": 4.0, "text": "x y z",
     "words": [{"word": "x", "start": 0.1, "end": 0.4, "speaker": "A"},
               {"word": "y", "start": 1.6, "end": 1.9, "speaker": "B"},
               {"word": "z", "start": 3.1, "end": 3.4}]},
]
segments_before = copy.deepcopy(segments)
rows_before = copy.deepcopy(algo_rows)
ns["_diag_note"]("trigger", collapsed=False, short=False, probe_ran=False, s0_labels=2)
fields = ns["_diag_output_fields"]("none", ["A", "B"], segments, algo_rows, "job-1", 4.0)
assert segments == segments_before, "diagnostics mutated the transcription segments"
assert algo_rows == rows_before, "diagnostics mutated the S1 rows"
assert fields["diarization_raw"] == s0["rows"]
debug = fields["diarization_debug"]
assert debug["status"] == "ok" and debug["computed_in_this_job"] is True
assert debug["replacement"]["reason_code"] == "pipeline_rows_kept_probe_not_triggered"
assert debug["replacement"]["replaced_pyannote_rows"] is False
assert debug["job_fingerprint_sha256_12"] == hashlib.sha256(b"job-1").hexdigest()[:12]
assert debug["s1"]["word_binding"]["words_total"] == 3 and debug["s1"]["word_binding"]["tagged_final"] == 2
assert debug["worker_code"]["base_commit"] and debug["worker_code"]["diag_patch"] >= 10
assert debug["worker_code"]["assignment"] == "three_pass_word"
assert debug["worker_code"]["pyannote_mode"] == "programmed"
assert debug["worker_code"]["diag_patch"] >= 13
raw_identity = ns["_worker_identity"]("raw")
assert raw_identity["assignment"] == "pyannote_raw" and raw_identity["pyannote_mode"] == "raw"
assert "pyannote_raw" not in raw_identity
raw_part = sep_src.split('if mode == "raw":', 1)[1].split('return annotation, "pyannote_raw"', 1)[0]
assert "_apply_pyannote_raw" in raw_part
assert "_prepare_diarization_audio" not in raw_part
assert "_configure_pipeline" not in raw_part
assert "_split_within_speakers" not in raw_part
assert "raw_controls" in handler_src and "pyannote_raw" in handler_src
stock = ns["_parse_pyannote_raw"](None)
assert stock["stock"] is True and stock["threshold"] is None and stock["num_speakers"] is None
assert ns["_parse_pyannote_raw"]({})["stock"] is True
clamped = ns["_parse_pyannote_raw"]({
    "threshold": 0.1, "Fa": 9, "Fb": 0, "min_duration_off": 3,
    "prompt": "split the speakers", "min_active_ratio": 0.02,
})
assert clamped["threshold"] == 0.5 and clamped["Fa"] == 0.5
assert clamped["Fb"] == 0.01 and clamped["min_duration_off"] == 1.0
assert clamped["stock"] is False and "prompt" not in clamped and "min_active_ratio" not in clamped
exact = ns["_parse_pyannote_raw"]({
    "num_speakers": 3, "min_speakers": 1, "max_speakers": 8, "threshold": 0.55,
})
assert exact["num_speakers"] == 3 and exact["min_speakers"] is None and exact["max_speakers"] is None
assert exact["threshold"] == 0.55
swapped = ns["_parse_pyannote_raw"]({"min_speakers": 5, "max_speakers": 2})
assert swapped["min_speakers"] == 2 and swapped["max_speakers"] == 5 and swapped["min_max_swapped"] is True
assert ns["_parse_pyannote_raw"]({"threshold": "nope", "num_speakers": True})["stock"] is True
assert ns["_parse_pyannote_raw"]({"num_speakers": 21})["num_speakers"] == 20
assert ns["_parse_pyannote_raw"]({"num_speakers": 0})["stock"] is True
assert ns["_public_raw_controls"](stock) == {"stock": True}
assert ns["_public_raw_controls"](exact)["threshold"] == 0.55
assert ns["_public_raw_controls"](exact)["num_speakers"] == 3
assert "min_speakers" not in ns["_public_raw_controls"](exact)
json.dumps(ns["_clean"](fields))  # must serialise (NaN/inf/numpy handled by _clean)

# rescue codes
for rescue, code, replaced in (
    ("num_speakers", "pipeline_recount_with_forced_speakers", True),
    ("windows", "window_labels_from_embedding_clustering", True),
    ("split_within", "within_speaker_embedding_split", True),
):
    info = ns["_diag_replacement"](rescue, 1, 2)
    assert info["reason_code"] == code and info["replaced_pyannote_rows"] is replaced, info

# failure isolation: broken input must degrade to a "failed" debug block, never raise
broken = ns["_diag_output_fields"]("none", ["A"], None, None, "job-2", 1.0)
assert broken["diarization_debug"]["status"] == "failed", broken

# a failing record helper must be silent
ns["_diag_record_s0"](object(), object())
ns["_diag_record_probe"](object(), object(), object(), object())

print("DIAG_PATCH_CHECKS_OK", SRC)

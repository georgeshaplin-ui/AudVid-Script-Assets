"""Bake model weights into the image so cold starts never hit the network for them.

The gated pyannote diarization pipeline is NOT baked in: it needs a Hugging Face token
and is fetched once per worker at runtime (see handler.py).

The Arabic dialect model is baked as CTranslate2 weights. A job must not download it.
"""
import argparse
import os
import shutil

import nltk
from huggingface_hub import snapshot_download

ARABIC_DIALECT_REPO = "oddadmix/whisper-large-v3-arabic-dialectal-v2"

FASTER_WHISPER_REPOS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v2": "Systran/faster-whisper-large-v2",
    "medium": "Systran/faster-whisper-medium",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", default="large-v3")
    parser.add_argument("--align", nargs="*", default=["en"])
    args = parser.parse_args()

    whisper_dir = os.path.join(os.environ["WHISPER_MODEL_DIR"], args.whisper)
    snapshot_download(FASTER_WHISPER_REPOS[args.whisper], local_dir=whisper_dir)
    print("WHISPER_OK", whisper_dir)

    from whisperx.alignment import load_align_model

    for lang in args.align:
        load_align_model(language_code=lang, device="cpu")
        print("ALIGN_OK", lang)

    nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], quiet=True)
    print("NLTK_OK")
    _bake_arabic_dialect()


def _bake_arabic_dialect() -> None:
    """Convert the multi-dialect Arabic fine-tune once, at image build."""
    import inspect

    from ctranslate2.converters import TransformersConverter

    dest = os.path.join(os.environ["WHISPER_MODEL_DIR"], "ar-dialect")
    source = "/tmp/ar-dialect-src"
    cache = "/tmp/ar-dialect-cache"
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(source, exist_ok=True)
    snapshot_download(
        ARABIC_DIALECT_REPO,
        local_dir=source,
        cache_dir=cache,
        allow_patterns=[
            "*.safetensors",
            "*.bin",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges.txt",
        ],
    )
    present = [
        name
        for name in ("tokenizer.json", "preprocessor_config.json", "vocabulary.json", "tokenizer_config.json")
        if os.path.isfile(os.path.join(source, name))
    ]
    if "tokenizer.json" not in present:
        raise RuntimeError(f"{ARABIC_DIALECT_REPO} did not include tokenizer.json")
    kwargs = {"copy_files": present}
    signature = inspect.signature(TransformersConverter.__init__)
    if "load_as_float16" in signature.parameters:
        kwargs["load_as_float16"] = True
    if "low_cpu_mem_usage" in signature.parameters:
        kwargs["low_cpu_mem_usage"] = True
    shutil.rmtree(dest, ignore_errors=True)
    TransformersConverter(source, **kwargs).convert(dest, quantization="float16", force=True)
    weights = os.path.join(dest, "model.bin")
    if not os.path.isfile(weights) or os.path.getsize(weights) < 500_000_000:
        raise RuntimeError(f"arabic dialect conversion did not write model.bin in {dest}")
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    print("ARABIC_DIALECT_OK", dest, os.path.getsize(weights))


if __name__ == "__main__":
    main()

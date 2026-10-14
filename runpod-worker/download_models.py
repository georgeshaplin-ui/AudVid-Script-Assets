"""Bake model weights into the image so cold starts never hit the network for them.

The gated pyannote diarization pipeline is NOT baked in: it needs a Hugging Face token
and is fetched once per worker at runtime (see handler.py).
"""
import argparse
import os

import nltk
from huggingface_hub import snapshot_download

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


if __name__ == "__main__":
    main()

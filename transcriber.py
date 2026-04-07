"""
transcriber.py — mlx-whisper transcription module

Loads mlx-community/whisper-large-v3-turbo once and caches the model
so repeated calls don't re-fetch weights from HuggingFace.
"""

import numpy as np

_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
_model_cache = {}  # keyed by model id — holds the loaded model object


def preload_model():
    """
    Explicitly load and cache the model weights.
    Call this once at startup so the first audio chunk isn't delayed.
    """
    if _MODEL_ID not in _model_cache:
        import mlx_whisper
        print(f"[transcriber] Loading model '{_MODEL_ID}'...")
        # Run a silent dummy transcription to force weight loading into memory
        dummy = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        mlx_whisper.transcribe(dummy, path_or_hf_repo=_MODEL_ID, verbose=False)
        _model_cache[_MODEL_ID] = mlx_whisper
        print("[transcriber] Model ready.")
    return _model_cache[_MODEL_ID]


def transcribe_chunk(audio_np: np.ndarray, language: str = None) -> dict:
    """
    Transcribe a numpy float32 audio array using mlx-whisper.

    Args:
        audio_np:  float32 numpy array at 16 kHz sample rate
        language:  ISO 639-1 language code (e.g. 'fr', 'en') or None for auto-detect

    Returns:
        dict with keys:
            text     — full transcription string
            language — detected/used language code
            segments — list of segment dicts from whisper
    """
    mlx_whisper = preload_model()

    kwargs = {
        "path_or_hf_repo": _MODEL_ID,
        "word_timestamps": False,
        "verbose": False,
    }

    if language and language.lower() != "auto":
        kwargs["language"] = language.lower()

    result = mlx_whisper.transcribe(audio_np, **kwargs)

    detected_lang = result.get("language", language or "unknown")
    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    return {
        "text": text,
        "language": detected_lang,
        "segments": segments,
    }

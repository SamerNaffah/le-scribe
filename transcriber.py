"""
transcriber.py — Transcription + speaker diarization module

Pipeline per chunk:
  1. Diarization  (pyannote) — who spoke and when
  2. Transcription (mlx-whisper) — what was said, per speaker segment
  3. Speaker tracking — consistent labels across chunks (SPEAKER_00, SPEAKER_01, ...)

Speaker labels can be mapped to real names via set_speaker_name().
"""

import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_whisper = None          # mlx_whisper module, loaded once
_diarizer = None         # pyannote Pipeline, loaded once

# Speaker identity tracking across chunks
# Maps pyannote's per-chunk label → global session label
_speaker_embeddings: list[tuple[str, np.ndarray]] = []  # [(global_label, embedding), ...]
_speaker_names: dict[str, str] = {}   # global_label → human name e.g. {"SPEAKER_00": "Samer"}
_EMBEDDING_THRESHOLD = 0.75           # cosine similarity to consider same speaker


# ---------------------------------------------------------------------------
# Public: speaker name mapping
# ---------------------------------------------------------------------------

def set_speaker_name(label: str, name: str):
    """Map a speaker label (e.g. 'SPEAKER_00') to a human name."""
    _speaker_names[label] = name


def get_speaker_names() -> dict[str, str]:
    return dict(_speaker_names)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def preload_model():
    """Load Whisper + pyannote once. Call after admission to warm up."""
    global _whisper, _diarizer

    if _whisper is None:
        import mlx_whisper
        print("[transcriber] Loading Whisper model...")
        dummy = np.zeros(_SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(dummy, path_or_hf_repo=_WHISPER_MODEL_ID, verbose=False)
        _whisper = mlx_whisper
        print("[transcriber] Whisper ready.")

    if _diarizer is None:
        import warnings
        import huggingface_hub
        huggingface_hub.login(token=_HF_TOKEN, add_to_git_credential=False)
        print("[transcriber] Loading diarization model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyannote.audio import Pipeline
            _diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        print("[transcriber] Diarization model ready.")

    return _whisper, _diarizer


# ---------------------------------------------------------------------------
# Speaker tracking helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _resolve_speaker(local_label: str, embedding: np.ndarray) -> str:
    """
    Match a per-chunk speaker embedding to a global session speaker label.
    Creates a new global label if no close match found.
    """
    best_sim = -1.0
    best_label = None
    for global_label, stored_emb in _speaker_embeddings:
        sim = _cosine_sim(embedding, stored_emb)
        if sim > best_sim:
            best_sim = sim
            best_label = global_label

    if best_sim >= _EMBEDDING_THRESHOLD and best_label is not None:
        # Update stored embedding with running average
        idx = next(i for i, (lbl, _) in enumerate(_speaker_embeddings) if lbl == best_label)
        old_emb = _speaker_embeddings[idx][1]
        _speaker_embeddings[idx] = (best_label, (old_emb + embedding) / 2)
        return best_label
    else:
        new_label = "SPEAKER_" + str(len(_speaker_embeddings)).zfill(2)
        _speaker_embeddings.append((new_label, embedding))
        print("[transcriber] New speaker detected: " + new_label)
        return new_label


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------

def transcribe_chunk(audio_np: np.ndarray, language: str = None) -> list[dict]:
    """
    Transcribe a chunk with speaker diarization.

    Returns a list of dicts (one per speaker segment):
        [{"speaker": "SPEAKER_00", "text": "...", "language": "fr"}, ...]

    Falls back to a single entry without speaker if diarization fails.
    """
    mlx_whisper, diarizer = preload_model()

    # --- Diarization ---
    try:
        # pyannote needs a file-like object or dict with waveform+sample_rate
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
        diarization = diarizer({"waveform": waveform, "sample_rate": _SAMPLE_RATE})

        # Extract segments: [(start, end, local_speaker_label), ...]
        segments = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Get speaker embeddings for identity tracking
        try:
            from pyannote.audio import Inference
            embedding_model = Inference("pyannote/embedding", window="whole")
            local_to_global = {}
            for _, _, local_label in segments:
                if local_label not in local_to_global:
                    # Extract embedding for this speaker's audio
                    speaker_audio = np.zeros(0, dtype=np.float32)
                    for start, end, lbl in segments:
                        if lbl == local_label:
                            s = int(start * _SAMPLE_RATE)
                            e = int(end * _SAMPLE_RATE)
                            speaker_audio = np.concatenate([speaker_audio, audio_np[s:e]])
                    if len(speaker_audio) > 0:
                        wf = torch.from_numpy(speaker_audio).unsqueeze(0)
                        emb = embedding_model({"waveform": wf, "sample_rate": _SAMPLE_RATE})
                        local_to_global[local_label] = _resolve_speaker(local_label, np.array(emb))
                    else:
                        local_to_global[local_label] = local_label
        except Exception:
            # Embedding extraction failed — use simple incremental labels
            known = {}
            for _, _, local_label in segments:
                if local_label not in known:
                    known[local_label] = "SPEAKER_" + str(len(known)).zfill(2)
            local_to_global = known

    except Exception as e:
        print("[transcriber] Diarization failed, falling back to single speaker: " + str(e))
        segments = [(0.0, len(audio_np) / _SAMPLE_RATE, "SPEAKER_00")]
        local_to_global = {"SPEAKER_00": "SPEAKER_00"}

    # --- Transcribe each speaker segment ---
    results = []
    whisper_kwargs = {
        "path_or_hf_repo": _WHISPER_MODEL_ID,
        "word_timestamps": False,
        "verbose": False,
    }
    if language and language.lower() != "auto":
        whisper_kwargs["language"] = language.lower()

    # Merge consecutive segments from same speaker (avoid micro-chunks)
    merged = []
    for start, end, local_label in segments:
        global_label = local_to_global.get(local_label, local_label)
        display = _speaker_names.get(global_label, global_label)
        if merged and merged[-1]["speaker"] == display:
            merged[-1]["end"] = end
        else:
            merged.append({"speaker": display, "start": start, "end": end, "local": local_label})

    for seg in merged:
        s = int(seg["start"] * _SAMPLE_RATE)
        e = int(seg["end"] * _SAMPLE_RATE)
        seg_audio = audio_np[s:e]

        if len(seg_audio) < _SAMPLE_RATE * 0.5:  # skip segments under 0.5s
            continue

        try:
            result = mlx_whisper.transcribe(seg_audio, **whisper_kwargs)
            text = result.get("text", "").strip()
            lang = result.get("language", language or "unknown")
            if text:
                results.append({
                    "speaker": seg["speaker"],
                    "text": text,
                    "language": lang,
                })
        except Exception as e:
            print("[transcriber] Whisper error on segment: " + str(e))

    # If nothing came out, fall back to full-chunk transcription
    if not results:
        try:
            result = mlx_whisper.transcribe(audio_np, **whisper_kwargs)
            text = result.get("text", "").strip()
            lang = result.get("language", language or "unknown")
            if text:
                results.append({"speaker": "SPEAKER_00", "text": text, "language": lang})
        except Exception:
            pass

    return results

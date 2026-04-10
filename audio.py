"""
audio.py — Audio capture with VAD-based chunking

Captures two streams simultaneously and mixes them:
  - BlackHole 2ch: remote participants (via bot's Chrome)
  - Bose mic (or MacBook mic): local user's own voice

Uses Silero VAD to detect speech boundaries and only yields chunks
when someone finishes speaking — no fixed timer, no mid-sentence cuts.
"""

import time
import queue
import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


BLACKHOLE_DEVICE_NAME = "BlackHole 2ch"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SILENCE_THRESHOLD = 0.005

# VAD settings
VAD_SAMPLE_RATE = 16000       # silero-vad requires 16kHz
MIN_SPEECH_SECONDS = 1.0      # ignore speech bursts shorter than this
MAX_CHUNK_SECONDS = 15.0      # force a cut if speech runs longer than this
SILENCE_PAD_SECONDS = 0.3     # silence padding to keep at end of each chunk


def find_blackhole_device() -> int:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if BLACKHOLE_DEVICE_NAME.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            print("[audio] Found BlackHole device: index=" + str(i) + ", name='" + dev["name"] + "'")
            return i
    available = "\n".join(
        "  [" + str(i) + "] " + d["name"] + " (in=" + str(d["max_input_channels"]) + ")"
        for i, d in enumerate(devices)
    )
    raise RuntimeError("BlackHole 2ch audio device not found.\nAvailable devices:\n" + available)


def find_mic_device() -> tuple[int, int] | None:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "bose" in dev["name"].lower() and dev["max_input_channels"] > 0:
            sr = int(dev["default_samplerate"])
            print("[audio] Found Bose mic: index=" + str(i) + ", name='" + dev["name"] + "', sr=" + str(sr))
            return i, sr
    for i, dev in enumerate(devices):
        if ("macbook" in dev["name"].lower() or "micro" in dev["name"].lower()) and dev["max_input_channels"] > 0:
            sr = int(dev["default_samplerate"])
            print("[audio] Found MacBook mic: index=" + str(i) + ", name='" + dev["name"] + "', sr=" + str(sr))
            return i, sr
    return None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    new_len = int(len(audio) * to_sr / from_sr)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _make_callback(q: queue.Queue):
    def _cb(indata, frames, time_info, status):
        if status:
            print("[audio] stream status: " + str(status))
        mono = indata.mean(axis=1).astype(np.float32)
        q.put(mono.copy())
    return _cb


def stream_chunks(
    device_index: int,
    chunk_seconds: int = 10,        # unused — kept for API compatibility
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    overlap_seconds: int = 0,       # VAD handles boundaries, no overlap needed
):
    """
    Generator yielding (audio_chunk: np.ndarray[float32], elapsed: float).

    Uses Silero VAD to detect when speech ends and yield complete utterances
    rather than fixed-size chunks.
    """
    print("[audio] Loading VAD model...")
    vad_model = load_silero_vad()
    print("[audio] VAD model ready.")

    bh_q: queue.Queue[np.ndarray] = queue.Queue()
    mic_q: queue.Queue[np.ndarray] = queue.Queue()

    bh_info = sd.query_devices(device_index)
    bh_native_sr = int(bh_info["default_samplerate"])
    mic_result = find_mic_device()

    bh_stream = sd.InputStream(
        device=device_index,
        channels=min(2, int(bh_info["max_input_channels"])),
        samplerate=bh_native_sr,
        dtype="float32",
        blocksize=int(bh_native_sr * 0.1),
        callback=_make_callback(bh_q),
    )

    mic_stream = None
    mic_native_sr = sample_rate
    if mic_result:
        mic_idx, mic_native_sr = mic_result
        mic_info = sd.query_devices(mic_idx)
        mic_stream = sd.InputStream(
            device=mic_idx,
            channels=1,
            samplerate=mic_native_sr,
            dtype="float32",
            blocksize=int(mic_native_sr * 0.1),
            callback=_make_callback(mic_q),
        )
        print("[audio] Capturing BlackHole (" + str(bh_native_sr) + "Hz) + mic (" + str(mic_native_sr) + "Hz)")
    else:
        print("[audio] Capturing BlackHole only — mic not found")

    # Rolling buffer at target 16kHz
    mixed_buf = np.zeros(0, dtype=np.float32)
    # Small secondary buffer for mic to stay in sync
    mic_buf = np.zeros(0, dtype=np.float32)

    start_time = time.time()
    min_frames = int(MIN_SPEECH_SECONDS * sample_rate)
    max_frames = int(MAX_CHUNK_SECONDS * sample_rate)
    pad_frames = int(SILENCE_PAD_SECONDS * sample_rate)

    print("[audio] Stream started. Listening for speech...")

    with bh_stream:
        if mic_stream:
            mic_stream.__enter__()
        try:
            while True:
                # Drain BlackHole
                while not bh_q.empty():
                    block = bh_q.get_nowait()
                    resampled = _resample(block, bh_native_sr, sample_rate)
                    mixed_buf = np.concatenate([mixed_buf, resampled])

                # Drain mic and keep in sync buffer
                while not mic_q.empty():
                    block = mic_q.get_nowait()
                    resampled = _resample(block, mic_native_sr, sample_rate)
                    mic_buf = np.concatenate([mic_buf, resampled])

                # Mix mic into mixed_buf
                if mic_stream and len(mic_buf) > 0:
                    mix_len = min(len(mic_buf), len(mixed_buf))
                    if mix_len > 0:
                        mixed_buf[-mix_len:] = np.clip(
                            mixed_buf[-mix_len:] + mic_buf[:mix_len], -1.0, 1.0
                        )
                        mic_buf = mic_buf[mix_len:]

                # Need at least min_frames to run VAD
                if len(mixed_buf) < min_frames:
                    time.sleep(0.05)
                    continue

                # Run VAD on the full buffer
                audio_tensor = torch.from_numpy(mixed_buf)
                timestamps = get_speech_timestamps(
                    audio_tensor,
                    vad_model,
                    sampling_rate=sample_rate,
                    min_silence_duration_ms=600,   # silence gap to consider speech ended
                    min_speech_duration_ms=int(MIN_SPEECH_SECONDS * 1000),
                )

                if not timestamps:
                    # No speech detected — if buffer is getting large, trim it
                    if len(mixed_buf) > max_frames:
                        mixed_buf = mixed_buf[-pad_frames:]
                    time.sleep(0.05)
                    continue

                last_ts = timestamps[-1]
                speech_end = last_ts["end"]
                buf_end = len(mixed_buf)

                # Check if speech has ended (silence after last speech segment)
                silence_after = buf_end - speech_end
                force_cut = buf_end >= max_frames

                if silence_after >= pad_frames or force_cut:
                    # Yield from start of buffer to end of last speech + padding
                    chunk_end = min(speech_end + pad_frames, buf_end)
                    chunk = mixed_buf[:chunk_end]

                    rms = _rms(chunk)
                    elapsed = time.time() - start_time

                    if rms >= silence_threshold:
                        duration = len(chunk) / sample_rate
                        print("[audio] Speech chunk: " + str(round(duration, 1)) + "s, "
                              "elapsed=" + str(round(elapsed, 1)) + "s, RMS=" + str(round(rms, 4)))
                        yield chunk, elapsed
                    else:
                        print("[audio] Skipping low-energy chunk (RMS=" + str(round(rms, 5)) + ")")

                    # Trim buffer — keep a small tail for context
                    mixed_buf = mixed_buf[chunk_end:]

                time.sleep(0.05)

        finally:
            if mic_stream:
                mic_stream.__exit__(None, None, None)

"""
audio.py — Audio capture module

Captures two streams and mixes them for transcription:
  - BlackHole 2ch (index 3, 48kHz): other participants via bot's Chrome
  - Bose QC45 mic (index 1, 16kHz): local user's own voice

Both are resampled to 16kHz and mixed before being sent to Whisper.
"""

import time
import queue
import numpy as np
import sounddevice as sd


BLACKHOLE_DEVICE_NAME = "BlackHole 2ch"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 10
DEFAULT_SILENCE_THRESHOLD = 0.005


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
    """
    Find the best available mic. Preference order:
      1. Bose QC45 (input only)
      2. MacBook mic
    Returns (device_index, native_sample_rate) or None.
    """
    devices = sd.query_devices()
    # Look for Bose input first
    for i, dev in enumerate(devices):
        if "bose" in dev["name"].lower() and dev["max_input_channels"] > 0:
            sr = int(dev["default_samplerate"])
            print("[audio] Found Bose mic: index=" + str(i) + ", name='" + dev["name"] + "', sr=" + str(sr))
            return i, sr
    # Fall back to MacBook mic
    for i, dev in enumerate(devices):
        if "macbook" in dev["name"].lower() or "micro" in dev["name"].lower():
            if dev["max_input_channels"] > 0:
                sr = int(dev["default_samplerate"])
                print("[audio] Found MacBook mic: index=" + str(i) + ", name='" + dev["name"] + "', sr=" + str(sr))
                return i, sr
    return None


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear resample from from_sr to to_sr."""
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def stream_chunks(
    device_index: int,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    overlap_seconds: int = 1,
):
    """
    Generator yielding (audio_chunk: np.ndarray[float32], elapsed: float).

    Captures BlackHole (remote participants) + mic (local user) simultaneously,
    resamples both to 16kHz, mixes them, and yields chunks for transcription.
    """
    chunk_frames = int(chunk_seconds * sample_rate)
    overlap_frames = int(overlap_seconds * sample_rate)

    bh_q: queue.Queue[np.ndarray] = queue.Queue()
    mic_q: queue.Queue[np.ndarray] = queue.Queue()

    # BlackHole native sr
    bh_info = sd.query_devices(device_index)
    bh_native_sr = int(bh_info["default_samplerate"])

    mic_result = find_mic_device()

    def make_callback(q):
        def _cb(indata, frames, time_info, status):
            if status:
                print("[audio] stream status: " + str(status))
            mono = indata.mean(axis=1).astype(np.float32)
            q.put(mono.copy())
        return _cb

    bh_blocksize = int(bh_native_sr * 0.1)
    bh_stream = sd.InputStream(
        device=device_index,
        channels=min(2, int(bh_info["max_input_channels"])),
        samplerate=bh_native_sr,
        dtype="float32",
        blocksize=bh_blocksize,
        callback=make_callback(bh_q),
    )

    mic_stream = None
    mic_native_sr = sample_rate
    if mic_result:
        mic_idx, mic_native_sr = mic_result
        mic_info = sd.query_devices(mic_idx)
        mic_blocksize = int(mic_native_sr * 0.1)
        mic_stream = sd.InputStream(
            device=mic_idx,
            channels=1,
            samplerate=mic_native_sr,
            dtype="float32",
            blocksize=mic_blocksize,
            callback=make_callback(mic_q),
        )
        print("[audio] Capturing BlackHole (" + str(bh_native_sr) + "Hz) + mic (" + str(mic_native_sr) + "Hz) → mixed at " + str(sample_rate) + "Hz")
    else:
        print("[audio] Capturing BlackHole only — mic not found")

    # Rolling buffers (in target 16kHz frames)
    bh_buf = np.zeros(0, dtype=np.float32)
    mic_buf = np.zeros(0, dtype=np.float32)
    prev_tail = np.zeros(0, dtype=np.float32)
    start_time = time.time()

    with bh_stream:
        if mic_stream:
            mic_stream.__enter__()
        try:
            print("[audio] Stream started. Buffering " + str(chunk_seconds) + "s chunks...")
            while True:
                # Drain BlackHole queue → resample → append to buffer
                while not bh_q.empty():
                    block = bh_q.get_nowait()
                    resampled = _resample(block, bh_native_sr, sample_rate)
                    bh_buf = np.concatenate([bh_buf, resampled])

                # Drain mic queue → resample → append to buffer
                while not mic_q.empty():
                    block = mic_q.get_nowait()
                    resampled = _resample(block, mic_native_sr, sample_rate)
                    mic_buf = np.concatenate([mic_buf, resampled])

                if len(bh_buf) < chunk_frames:
                    time.sleep(0.05)
                    continue

                # Slice chunk from BlackHole
                bh_chunk = bh_buf[:chunk_frames]
                bh_buf = bh_buf[chunk_frames - overlap_frames:]

                # Mix in mic if we have enough
                if mic_stream and len(mic_buf) >= chunk_frames:
                    mic_chunk = mic_buf[:chunk_frames]
                    mic_buf = mic_buf[chunk_frames - overlap_frames:]
                    mixed = np.clip(bh_chunk + mic_chunk, -1.0, 1.0)
                else:
                    mixed = bh_chunk
                    # Drop excess mic buffer if it drifts too far ahead
                    if len(mic_buf) > chunk_frames * 3:
                        mic_buf = mic_buf[-chunk_frames:]

                # Prepend overlap tail
                if len(prev_tail) > 0:
                    chunk_with_overlap = np.concatenate([prev_tail, mixed])
                else:
                    chunk_with_overlap = mixed

                prev_tail = mixed[-overlap_frames:] if overlap_frames > 0 else np.zeros(0, dtype=np.float32)

                rms = _rms(chunk_with_overlap)
                elapsed = time.time() - start_time

                if rms < silence_threshold:
                    print("[audio] Skipping silent chunk (RMS=" + str(round(rms, 5)) + ")")
                    continue

                print("[audio] Yielding chunk: elapsed=" + str(round(elapsed, 1)) + "s, RMS=" + str(round(rms, 4)))
                yield chunk_with_overlap, elapsed

        finally:
            if mic_stream:
                mic_stream.__exit__(None, None, None)

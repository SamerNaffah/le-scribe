"""
bot.py — Google Meet transcription bot

Usage:
    python bot.py --meet https://meet.google.com/xxx-xxxx-xxx --lang auto
    python bot.py --meet https://meet.google.com/xxx-xxxx-xxx --lang fr

Joins Google Meet using an existing Chrome profile (no credentials needed),
captures audio via BlackHole 2ch, transcribes with mlx-whisper, and saves
the full session transcript to ~/Documents/transcriptions/.
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import sys
import tempfile

# Force line-buffered stdout so server.py sees transcript lines immediately
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1, closefd=False)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1, closefd=False)
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Transcript storage
# ---------------------------------------------------------------------------

transcript_entries: list[dict] = []
transcript_lock = threading.Lock()
session_start: datetime.datetime = datetime.datetime.now()


def format_elapsed(seconds: float) -> str:
    """Convert elapsed seconds to HH:MM:SS string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def add_entry(elapsed: float, lang: str, text: str):
    entry = {
        "time": format_elapsed(elapsed),
        "lang": lang,
        "text": text,
    }
    with transcript_lock:
        transcript_entries.append(entry)
    print(f"[{entry['time']}] ({lang}) {text}")


def save_transcript():
    """Save transcript to ~/Documents/transcriptions/ as .txt and .json."""
    out_dir = Path.home() / "Documents" / "transcriptions"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = session_start.strftime("%Y-%m-%d_%H-%M")
    base = out_dir / f"{timestamp}_meet"

    with transcript_lock:
        entries = list(transcript_entries)

    # Plain text
    txt_path = base.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Google Meet Transcript — {session_start.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        for e in entries:
            f.write(f"[{e['time']}] ({e['lang']}) {e['text']}\n")

    # JSON
    json_path = base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\n[bot] Transcript saved:")
    print(f"      {txt_path}")
    print(f"      {json_path}")
    return str(txt_path), str(json_path)


# ---------------------------------------------------------------------------
# Signal handling — Ctrl+C always saves before exit
# ---------------------------------------------------------------------------

_driver = None  # global reference for cleanup
_tmp_profile = None  # temp Chrome profile dir to clean up on exit


def _graceful_exit(signum, frame):
    print("\n[bot] Interrupted — saving transcript before exit...")
    save_transcript()
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass
    if _tmp_profile and os.path.exists(_tmp_profile):
        try:
            shutil.rmtree(_tmp_profile, ignore_errors=True)
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)


# ---------------------------------------------------------------------------
# Chrome / Meet joining
# ---------------------------------------------------------------------------

def build_driver():
    """
    Launch Chrome as a guest (no profile/login needed).
    Uses a fresh temp profile each run so there's no conflict with the
    user's running Chrome instance.
    """
    global _tmp_profile
    import undetected_chromedriver as uc

    # Fresh throwaway profile — no login required
    tmp_dir = tempfile.mkdtemp(prefix="meet_transcriber_profile_")
    _tmp_profile = tmp_dir

    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={tmp_dir}")

    # Auto-grant camera/mic permissions so Meet doesn't prompt
    options.add_argument("--use-fake-ui-for-media-stream")
    # Keep audio playing so BlackHole can capture it
    options.add_argument("--mute-audio=false")

    # Prevent Chrome from suspending audio in background tabs
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")

    # NOT headless — Google Meet requires a visible window
    # options.add_argument("--headless")  # intentionally omitted

    driver_path = str(Path.home() / "Library" / "Application Support" / "meet_transcriber" / "chromedriver")
    print(f"[bot] Starting ChromeDriver: {driver_path}")
    driver = uc.Chrome(options=options, driver_executable_path=driver_path, version_main=146)
    driver.set_window_size(1280, 800)
    return driver


def enter_name_as_guest(driver, name: str = "Transcription Bot", timeout: int = 15):
    """
    On the Meet pre-join screen, type a guest name into the name input field.
    Google Meet shows this when joining without a Google account.
    """
    from selenium.webdriver.common.by import By

    print(f"[bot] Looking for guest name field to enter '{name}'...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Meet renders a text input with placeholder "Your name" for guests
        selectors = [
            'input[placeholder*="name" i]',
            'input[aria-label*="name" i]',
            'input[type="text"]',
        ]
        for sel in selectors:
            try:
                fields = driver.find_elements(By.CSS_SELECTOR, sel)
                for field in fields:
                    if field.is_displayed() and field.is_enabled():
                        field.click()
                        field.clear()
                        field.send_keys(name)
                        print(f"[bot] Entered guest name: '{name}'")
                        return True
            except Exception:
                pass
        time.sleep(1)

    print("[bot] WARNING: Could not find guest name field — joining without name.")
    return False


def dismiss_media_prompts(driver):
    """Click through mic/cam permission dialogs if they appear."""
    from selenium.webdriver.common.by import By

    time.sleep(2)
    selectors_to_click_off = [
        'button[aria-label*="microphone"]',
        'button[data-is-muted="false"][data-tooltip*="microphone"]',
    ]
    for sel in selectors_to_click_off:
        try:
            btns = driver.find_elements(By.CSS_SELECTOR, sel)
            for btn in btns:
                if btn.is_displayed():
                    btn.click()
                    time.sleep(0.5)
        except Exception:
            pass


def click_join_button(driver, timeout: int = 15) -> bool:
    """
    Find and click 'Ask to join' or 'Join now' button on the pre-join screen.
    Returns True if clicked, False if not found.
    """
    from selenium.webdriver.common.by import By

    join_texts = ["Ask to join", "Join now", "Join", "Demander à rejoindre", "Rejoindre"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        for text in join_texts:
            try:
                # Use XPath to find button containing this text (case-insensitive via translate)
                xpath = (
                    f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                    f"'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"
                )
                btns = driver.find_elements(By.XPATH, xpath)
                for btn in btns:
                    if btn.is_displayed() and btn.is_enabled():
                        btn.click()
                        print(f"[bot] Clicked join button: '{text}'")
                        return True
            except Exception:
                pass
        time.sleep(1)

    print("[bot] WARNING: Could not find join button within timeout.")
    return False


def wait_for_admission(driver, timeout: int = 300) -> bool:
    """
    Poll the DOM until we are admitted to the meeting or timeout.
    Detects admission by checking for elements that only appear inside the call.
    Returns True when admitted, False on timeout.
    """
    from selenium.webdriver.common.by import By

    print("[bot] Waiting for admission to the meeting...")
    deadline = time.time() + timeout
    dots = 0

    while time.time() < deadline:
        # Check for "waiting to be let in" message — still pending
        waiting_selectors = [
            '[data-call-ended]',               # meeting ended indicator
        ]
        # Indicators that we're INSIDE the call
        admitted_selectors = [
            'button[aria-label*="Leave call"]',
            'button[aria-label*="Quitter"]',
            'div[data-allocation-index]',       # participant tiles
            '[data-ssrc]',                      # video stream elements
            'button[aria-label*="microphone"]', # in-call controls
        ]

        for sel in admitted_selectors:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                if elems and any(e.is_displayed() for e in elems):
                    print(f"\n[bot] Admitted to meeting! (detected: {sel})")
                    return True
            except Exception:
                pass

        dots = (dots + 1) % 4
        print(f"\r[bot] Waiting to be admitted{'.' * (dots + 1)}   ", end="", flush=True)
        time.sleep(3)

    print(f"\n[bot] WARNING: Admission timeout after {timeout}s.")
    return False


def detect_meeting_ended(driver) -> bool:
    """Check if the meeting has ended (DOM signals)."""
    from selenium.webdriver.common.by import By

    end_signals = [
        # "You've left the meeting" / "The meeting has ended"
        'div[data-meeting-ended]',
        '[data-call-ended]',
    ]
    text_signals = ["left the meeting", "meeting has ended", "réunion s'est terminée"]

    for sel in end_signals:
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                return True
        except Exception:
            pass

    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
        for sig in text_signals:
            if sig in body_text:
                return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Google Meet transcription bot using mlx-whisper (fully local)"
    )
    parser.add_argument(
        "--meet", required=True,
        help="Google Meet URL, e.g. https://meet.google.com/xxx-xxxx-xxx"
    )
    parser.add_argument(
        "--lang", default="auto",
        help="Language code (e.g. 'fr', 'en') or 'auto' for auto-detection (default: auto)"
    )
    parser.add_argument(
        "--bot-name", default="Transcription Bot",
        help="Display name the bot uses to join the meeting (default: 'Transcription Bot')"
    )
    parser.add_argument(
        "--chunk-seconds", type=int, default=10,
        help="Audio chunk size in seconds for transcription (default: 10)"
    )
    parser.add_argument(
        "--silence-threshold", type=float, default=0.005,
        help="RMS threshold below which chunks are treated as silence (default: 0.005)"
    )
    parser.add_argument(
        "--admission-timeout", type=int, default=300,
        help="Seconds to wait for meeting admission before giving up (default: 300)"
    )
    return parser.parse_args()


def transcription_worker(device_index: int, chunk_seconds: int, silence_threshold: float, lang: str):
    """Background thread: capture audio → transcribe → store entries."""
    from audio import stream_chunks
    from transcriber import transcribe_chunk

    language = None if lang.lower() == "auto" else lang.lower()

    try:
        for audio_chunk, elapsed in stream_chunks(
            device_index=device_index,
            chunk_seconds=chunk_seconds,
            silence_threshold=silence_threshold,
        ):
            try:
                result = transcribe_chunk(audio_chunk, language=language)
                text = result["text"]
                detected_lang = result["language"]

                if text:
                    add_entry(elapsed, detected_lang, text)
            except Exception as e:
                print(f"[transcriber] Error transcribing chunk: {e}")
    except Exception as e:
        print(f"[audio] Stream error: {e}")


def main():
    global _driver

    args = parse_args()

    print("=" * 60)
    print("  Google Meet Transcription Bot")
    print(f"  Meet URL : {args.meet}")
    print(f"  Language : {args.lang}")
    print(f"  Bot name : {args.bot_name}")
    print("=" * 60)

    # Step 1: Find BlackHole device early — fail fast before launching Chrome
    from audio import find_blackhole_device
    device_index = find_blackhole_device()

    # Step 2: Launch Chrome (fresh guest profile, no login needed)
    print(f"\n[bot] Launching Chrome as guest...")
    _driver = build_driver()

    try:
        print(f"[bot] Navigating to {args.meet}")
        _driver.get(args.meet)
        time.sleep(4)  # allow page to load and render name field

        # Step 3: Enter guest name, dismiss media prompts, then click join
        enter_name_as_guest(_driver, name=args.bot_name)
        time.sleep(1)
        dismiss_media_prompts(_driver)
        time.sleep(1)
        click_join_button(_driver)

        # Step 4: Wait for admission
        admitted = wait_for_admission(_driver, timeout=args.admission_timeout)
        if not admitted:
            print("[bot] Could not get admitted. Saving empty transcript and exiting.")
            save_transcript()
            _driver.quit()
            return

        print("[bot] In the meeting. Pre-loading transcription model...")
        from transcriber import preload_model
        preload_model()
        print("[bot] Starting audio capture and transcription...")
        print("[bot] Press Ctrl+C at any time to stop and save transcript.\n")

        # Step 5: Start transcription in background thread
        t = threading.Thread(
            target=transcription_worker,
            args=(device_index, args.chunk_seconds, args.silence_threshold, args.lang),
            daemon=True,
        )
        t.start()

        # Step 6: Monitor meeting end in main thread
        while True:
            time.sleep(5)
            try:
                if detect_meeting_ended(_driver):
                    print("\n[bot] Meeting has ended.")
                    break
            except Exception:
                # Driver may have been closed
                break

        # Allow final chunk to finish transcribing
        print("[bot] Waiting for final transcription chunk...")
        time.sleep(15)

    finally:
        save_transcript()
        try:
            _driver.quit()
        except Exception:
            pass
        if _tmp_profile and os.path.exists(_tmp_profile):
            shutil.rmtree(_tmp_profile, ignore_errors=True)
        print("[bot] Done.")
        # Suppress chromedriver mutex crash on exit (harmless uc cleanup bug)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)


if __name__ == "__main__":
    main()

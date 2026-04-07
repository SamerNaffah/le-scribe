"""
server.py — Flask web UI for meet-transcriber
"""

import io
import json
import queue
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.process: subprocess.Popen | None = None
        self.running = False
        self.entries: list[dict] = []
        self.log_lines: list[str] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.meet_url: str = ""
        self.language: str = "auto"
        # Each SSE client gets its own queue registered here
        self._client_queues: list[queue.Queue] = []
        self._lock = threading.Lock()
        # Replay buffer: every event ever pushed this session
        self._event_log: list[str] = []

    # -- fan-out to all connected SSE clients --
    def push(self, event: str, data: dict):
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        with self._lock:
            self._event_log.append(msg)
            for q in self._client_queues:
                q.put(msg)

    def add_client(self) -> queue.Queue:
        """Register a new SSE client and replay all past events to it."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            # Replay everything that happened before this client connected
            for msg in self._event_log:
                q.put(msg)
            self._client_queues.append(q)
        return q

    def remove_client(self, q: queue.Queue):
        with self._lock:
            try:
                self._client_queues.remove(q)
            except ValueError:
                pass


session = Session()
session_lock = threading.Lock()   # guards session.running / session.process

# ---------------------------------------------------------------------------
# Bot output parsing
# ---------------------------------------------------------------------------

TRANSCRIPT_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s+\((\w+)\)\s+(.*)")


def _read_bot_output(proc: subprocess.Popen):
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line:
            continue

        m = TRANSCRIPT_RE.match(line)
        if m:
            entry = {"time": m.group(1), "lang": m.group(2), "text": m.group(3)}
            with session_lock:
                session.entries.append(entry)
            session.push("transcript", entry)
        else:
            with session_lock:
                session.log_lines.append(line)
            session.push("log", {"message": line})

    proc.wait()
    _finalize()


def _finalize():
    with session_lock:
        session.running = False
        session.end_time = datetime.now()
        entries = list(session.entries)
        start = session.start_time
        end = session.end_time

    duration = int((end - start).total_seconds()) if start and end else 0
    word_count = sum(len(e["text"].split()) for e in entries)
    langs = list(dict.fromkeys(e["lang"] for e in entries))

    session.push("done", {
        "duration": _fmt(duration),
        "segments": len(entries),
        "words": word_count,
        "languages": langs,
    })


def _fmt(seconds: int) -> str:
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preflight")
def api_preflight():
    """Check system requirements before starting a session."""
    issues = []

    # Check BlackHole
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        found = any(
            "blackhole 2ch" in d["name"].lower() and d["max_input_channels"] > 0
            for d in devices
        )
        if not found:
            available = [d["name"] for d in devices]
            issues.append({
                "type": "blackhole",
                "message": "BlackHole 2ch audio device not found.",
                "detail": (
                    "Run: brew install blackhole-2ch, then reboot.\n"
                    "Then open Audio MIDI Setup → create a Multi-Output Device with "
                    "BlackHole 2ch + your speakers → set it as system output."
                ),
                "available_devices": available,
            })
    except Exception as e:
        issues.append({"type": "sounddevice", "message": str(e)})

    # Check mlx_whisper importable
    try:
        import mlx_whisper  # noqa
    except ImportError:
        issues.append({
            "type": "mlx_whisper",
            "message": "mlx-whisper not installed.",
            "detail": "Run: pip install mlx-whisper",
        })

    return jsonify({"ok": len(issues) == 0, "issues": issues})


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.json or {}
    meet_url = (data.get("meet_url") or "").strip()
    language = (data.get("language") or "auto").strip()

    if not meet_url:
        return jsonify({"error": "Meet URL is required"}), 400

    with session_lock:
        if session.running:
            return jsonify({"error": "A session is already running"}), 409

        # Reset session (preserves connected SSE clients by rebuilding)
        old_clients = session._client_queues[:]
        session.reset()
        # Re-register existing SSE clients into the new session
        with session._lock:
            session._client_queues = old_clients

        session.running = True
        session.meet_url = meet_url
        session.language = language
        session.start_time = datetime.now()

    bot_path = Path(__file__).parent / "bot.py"
    cmd = [sys.executable, "-u", str(bot_path), "--meet", meet_url, "--lang", language]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    with session_lock:
        session.process = proc

    # Push started AFTER process is created so replay works correctly
    session.push("started", {"meet_url": meet_url, "language": language})

    threading.Thread(target=_read_bot_output, args=(proc,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with session_lock:
        proc = session.process
        running = session.running

    if proc is None:
        return jsonify({"error": "No active session"}), 400

    # SIGINT triggers graceful shutdown + transcript save in bot.py
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass  # already dead

    return jsonify({"status": "stopping"})


@app.route("/api/stream")
def api_stream():
    """SSE endpoint — one persistent connection per browser tab."""
    client_q = session.add_client()

    def generate():
        try:
            while True:
                try:
                    msg = client_q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            session.remove_client(client_q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/status")
def api_status():
    with session_lock:
        return jsonify({
            "running": session.running,
            "meet_url": session.meet_url,
            "language": session.language,
            "segments": len(session.entries),
            "start_time": session.start_time.isoformat() if session.start_time else None,
        })


@app.route("/api/download/pdf")
def download_pdf():
    with session_lock:
        entries = list(session.entries)
        meet_url = session.meet_url
        start_time = session.start_time
        end_time = session.end_time or datetime.now()

    if not entries:
        return jsonify({"error": "No transcript to export"}), 400

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm, bottomMargin=2.5*cm,
        )
        styles = getSampleStyleSheet()

        title_s = ParagraphStyle("T", parent=styles["Heading1"],
                                 fontSize=18, spaceAfter=6,
                                 textColor=colors.HexColor("#1a1a2e"))
        meta_s  = ParagraphStyle("M", parent=styles["Normal"],
                                 fontSize=9, textColor=colors.HexColor("#666666"), spaceAfter=4)
        time_s  = ParagraphStyle("Ti", parent=styles["Normal"],
                                 fontSize=9, textColor=colors.HexColor("#5b6af0"),
                                 fontName="Helvetica-Bold")
        lang_s  = ParagraphStyle("L", parent=styles["Normal"],
                                 fontSize=9, textColor=colors.HexColor("#888888"))
        text_s  = ParagraphStyle("Tx", parent=styles["Normal"],
                                 fontSize=11, leading=15, spaceAfter=8)

        duration = int((end_time - start_time).total_seconds()) if start_time else 0
        word_count = sum(len(e["text"].split()) for e in entries)
        langs = list(dict.fromkeys(e["lang"] for e in entries))

        story = [
            Paragraph("Meet Transcript", title_s),
            Paragraph(
                f"Date: {start_time.strftime('%B %d, %Y — %H:%M') if start_time else '—'}"
                f" &nbsp;·&nbsp; Duration: {_fmt(duration)}"
                f" &nbsp;·&nbsp; Segments: {len(entries)}"
                f" &nbsp;·&nbsp; Words: {word_count}"
                f" &nbsp;·&nbsp; Languages: {', '.join(langs) or '—'}",
                meta_s,
            ),
        ]
        if meet_url:
            story.append(Paragraph(f"Meeting: {meet_url}", meta_s))
        story += [Spacer(1, 0.4*cm),
                  HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")),
                  Spacer(1, 0.4*cm)]

        for e in entries:
            row = Table(
                [[Paragraph(e["time"], time_s),
                  Paragraph(f'({e["lang"]})', lang_s),
                  Paragraph(e["text"], text_s)]],
                colWidths=[2*cm, 1.2*cm, None],
            )
            row.setStyle(TableStyle([
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                ("TOPPADDING",    (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ]))
            story.append(row)

        doc.build(story)
        buf.seek(0)
        fname = f"transcript_{start_time.strftime('%Y-%m-%d_%H-%M') if start_time else 'export'}.pdf"
        return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)

    except ImportError:
        return jsonify({"error": "reportlab not installed — run: pip install reportlab"}), 500


@app.route("/api/download/txt")
def download_txt():
    with session_lock:
        entries = list(session.entries)
        meet_url = session.meet_url
        start_time = session.start_time

    if not entries:
        return jsonify({"error": "No transcript to export"}), 400

    buf = io.StringIO()
    buf.write("Google Meet Transcript\n")
    buf.write(f"Date: {start_time.strftime('%Y-%m-%d %H:%M') if start_time else '—'}\n")
    buf.write(f"URL:  {meet_url}\n")
    buf.write("=" * 60 + "\n\n")
    for e in entries:
        buf.write(f"[{e['time']}] ({e['lang']}) {e['text']}\n")

    fname = f"transcript_{start_time.strftime('%Y-%m-%d_%H-%M') if start_time else 'export'}.txt"
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=fname,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = 5050
    print(f"\n  Meet Transcriber UI → http://localhost:{port}\n")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)

"""Live dictation: the worker the supervisor runs while the microphone is open.

It records, and while you speak it transcribes the part you have not finished
yet about once a second, so the words appear as you say them. whisper.cpp is not
a streaming model, so "live" is done by window: speech that is followed by a
pause is transcribed once and committed, and only the open tail after it is
transcribed again on each tick.

It shares the GPU with the coding model, and yields to it:
  * the speech model is loaded once, not on every use;
  * nothing is transcribed while the coding model is reading or generating, so
    dictating over a running answer costs it nothing (the text arrives when it
    finishes, or when you stop);
  * each tick waits three times as long as the last one took (up to 2.5 s),
    so dictation normally uses a quarter of the GPU at most;
  * silence is never transcribed.

Protocol, one JSON object per line on stdout:
  {"type": "recording"}                    the microphone is open
  {"type": "partial", "text": "..."}       everything heard so far
  {"type": "final", "text": "..."}         after stdin closes
  {"type": "error", "message": "..."}

Run inside the voice runtime:  python voice_live.py <speech model> [--wav FILE]
`--wav` replays a 16 kHz mono file at real-time pace instead of opening the
microphone; the tests use it.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import wave

RATE = 16000
TICK_SECONDS = 0.9            # how often the open tail is re-transcribed, at best
YIELD_FACTOR = 3.0            # wait this many times the last transcription before the next
MAX_WAIT_SECONDS = 2.5        # but never leave the words more than this far behind
MIN_SPEECH_SECONDS = 1.0      # do not transcribe less than this: a clip that short is mostly guessed
PAUSE_SECONDS = 0.7           # a pause this long ends a phrase
MAX_TAIL_SECONDS = 12.0       # an unbroken tail longer than this is cut at its quietest point
FRAME = RATE // 50            # 20 ms
SILENCE_RMS = 350.0           # int16; room noise sits well under this


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def frame_rms(samples):
    """RMS of each 20 ms frame of an int16 array."""
    import numpy as np
    n = len(samples) // FRAME
    if n == 0:
        return np.zeros(0)
    frames = samples[: n * FRAME].astype("float32").reshape(n, FRAME)
    return np.sqrt((frames * frames).mean(axis=1))


def trailing_silence(samples) -> float:
    """Seconds of silence at the end."""
    rms = frame_rms(samples)
    quiet = 0
    for value in rms[::-1]:
        if value >= SILENCE_RMS:
            break
        quiet += 1
    return quiet * FRAME / RATE


def has_speech(samples) -> bool:
    rms = frame_rms(samples)
    return bool(len(rms)) and int((rms >= SILENCE_RMS).sum()) * FRAME / RATE >= 0.2


def quietest_cut(samples) -> int:
    """Where to cut an over-long tail: the quietest 200 ms in its last four
    seconds, leaving at least half a second after it."""
    import numpy as np
    rms = frame_rms(samples)
    lo, hi = max(0, len(rms) - 200), len(rms) - 25
    if hi - lo < 10:
        return len(samples)
    window = np.convolve(rms[lo:hi], np.ones(10) / 10, mode="valid")
    return (lo + int(window.argmin()) + 5) * FRAME


def join(committed: str, tail: str) -> str:
    return " ".join(part for part in (committed.strip(), tail.strip()) if part)


class CodingModel:
    """Whether the coding model is using the GPU right now, from the
    supervisor's own progress report. Unknown counts as idle."""

    def __init__(self) -> None:
        self.url = os.environ.get("LOCALCODE_CONTROL_URL", "").rstrip("/")
        self.token = os.environ.get("LOCALCODE_CONTROL_TOKEN", "")

    def busy(self) -> bool:
        if not self.url:
            return False
        try:
            req = urllib.request.Request(self.url + "/progress", headers={"X-Localcode-Token": self.token})
            with urllib.request.urlopen(req, timeout=0.4) as r:
                return json.load(r).get("phase") in ("reading", "generating")
        except Exception:  # noqa: BLE001
            return False


class Audio:
    """Everything captured so far, as int16 chunks."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chunks: list[bytes] = []
        self.done = threading.Event()

    def add(self, data: bytes) -> None:
        with self.lock:
            self.chunks.append(data)

    def samples(self):
        import numpy as np
        with self.lock:
            data = b"".join(self.chunks)
        return np.frombuffer(data, dtype="int16")


def replay(audio: Audio, path: str, stop: threading.Event) -> None:
    """Feed a wav file at the pace a microphone would."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("--wav needs 16 kHz mono 16-bit audio")
        step = RATE // 10
        while not stop.is_set():
            data = w.readframes(step)
            if not data:
                break
            audio.add(data)
            time.sleep(0.1)
    audio.done.set()


def main() -> int:
    model_path = sys.argv[1]
    wav = sys.argv[sys.argv.index("--wav") + 1] if "--wav" in sys.argv else os.environ.get("LOCALCODE_VOICE_TEST_WAV")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from localcode.voice import _clean_transcript, _silence_native_stderr

    audio, stop = Audio(), threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), stop.set()), daemon=True).start()

    stream = None
    if wav:
        threading.Thread(target=replay, args=(audio, wav, stop), daemon=True).start()
    else:
        import sounddevice as sd
        stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", callback=lambda data, n, t, status: audio.add(bytes(data)))
        stream.start()
    emit({"type": "recording"})

    from pywhispercpp.model import Model
    with _silence_native_stderr():
        model = Model(model_path, n_threads=4, print_realtime=False, print_progress=False, print_timestamps=False,
                      **({"context_params": {"use_gpu": False}} if os.environ.get("LOCALCODE_VOICE_GPU") == "0" else {}))

    def transcribe(samples) -> str:
        with _silence_native_stderr():
            segments = model.transcribe(samples.astype("float32") / 32768.0)
        return _clean_transcript(" ".join(s.text for s in segments).strip())

    coding_model = CodingModel()
    committed, start, last_partial, took = "", 0, "", 0.0
    while not stop.is_set() and not audio.done.is_set():
        stop.wait(max(TICK_SECONDS, min(took * YIELD_FACTOR, MAX_WAIT_SECONDS)))
        if stop.is_set():
            break
        tail = audio.samples()[start:]
        if len(tail) < MIN_SPEECH_SECONDS * RATE or coding_model.busy():
            continue
        if not has_speech(tail):
            start += max(0, len(tail) - RATE // 2)          # nothing said: drop the silence, keep half a second of lead-in
            continue
        paused = trailing_silence(tail) >= PAUSE_SECONDS
        too_long = len(tail) > MAX_TAIL_SECONDS * RATE
        cut = len(tail) if paused or not too_long else quietest_cut(tail)
        began = time.monotonic()
        text = transcribe(tail[:cut])
        took = time.monotonic() - began
        if paused or too_long:
            committed, start, text = join(committed, text), start + cut, ""
        partial = join(committed, text)
        if partial != last_partial:
            last_partial = partial
            emit({"type": "partial", "text": partial})

    if stream is not None:
        stream.stop(); stream.close()
    tail = audio.samples()[start:]
    text = transcribe(tail) if len(tail) >= RATE // 5 and has_speech(tail) else ""
    emit({"type": "final", "text": join(committed, text)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # noqa: BLE001 - the supervisor shows this to the user
        emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        raise SystemExit(1)

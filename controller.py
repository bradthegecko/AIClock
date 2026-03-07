import json
import subprocess
import threading
import time
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import requests
from faster_whisper import WhisperModel
from openwakeword.model import Model

from tts import speak

STATUS_FILE = "status.json"
PORT = 8000
CLOCK_URL = f"http://127.0.0.1:{PORT}/clock_display.html"

MIC_DEVICE = "plughw:CARD=Device,DEV=0"
OLLAMA_MODEL = "llama3.2:3b"

INPUT_RATE = 44100
TARGET_RATE = 16000
TARGET_CHUNK = 1280
INPUT_CHUNK = int(TARGET_CHUNK * INPUT_RATE / TARGET_RATE)

WAKEWORD_THRESHOLD = 0.65
REQUIRED_CONSECUTIVE_HITS = 2
WAKE_DEBOUNCE_SECONDS = 6
POST_ASSISTANT_COOLDOWN_SECONDS = 6
WAKE_STREAM_WARMUP_SECONDS = 2.0

WHISPER_MODEL = WhisperModel("tiny", compute_type="int8")

trigger_event = threading.Event()
assistant_busy = threading.Event()
shutdown_event = threading.Event()

wake_proc = None
wake_model = None
last_trigger_time = 0.0
wake_stream_started_at = 0.0


def set_status(text: str) -> None:
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"assistant_status": text}, f)


class ClockHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        global last_trigger_time

        if self.path == "/trigger":
            now = time.time()
            if not assistant_busy.is_set() and (now - last_trigger_time) > 1:
                last_trigger_time = now
                trigger_event.set()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_web_server():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), ClockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def open_clock():
    subprocess.Popen([
        "chromium",
        "--start-fullscreen",
        CLOCK_URL
    ])
    time.sleep(3)


def find_alexa_model() -> str:
    local_candidates = [
        Path("wakewords/alexa.onnx"),
        Path("wakewords/alexa_v0.1.onnx"),
        Path("wakewords/alexa.tflite"),
    ]

    for path in local_candidates:
        if path.exists():
            return str(path)

    import openwakeword
    package_dir = Path(openwakeword.__file__).resolve().parent
    matches = list(package_dir.rglob("*alexa*.onnx")) + list(package_dir.rglob("*alexa*.tflite"))
    if matches:
        return str(matches[0])

    raise FileNotFoundError("Could not find Alexa wake-word model.")


def start_wake_stream():
    global wake_proc, wake_stream_started_at

    if wake_proc is not None:
        return

    wake_proc = subprocess.Popen(
        [
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(INPUT_RATE),
            "-c", "1",
            "-t", "raw"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    wake_stream_started_at = time.time()


def stop_wake_stream():
    global wake_proc

    if wake_proc is None:
        return

    wake_proc.terminate()
    try:
        wake_proc.wait(timeout=1)
    except Exception:
        wake_proc.kill()

    wake_proc = None


def wakeword_worker():
    global wake_model, last_trigger_time

    alexa_model_path = find_alexa_model()
    print("Using wake word model:", alexa_model_path)

    wake_model = Model(wakeword_model_paths=[alexa_model_path])

    print("Wake word listener ready. Say 'Alexa'.")
    start_wake_stream()

    bytes_per_sample = 2
    bytes_per_chunk = INPUT_CHUNK * bytes_per_sample
    counter = 0
    consecutive_hits = 0

    while not shutdown_event.is_set():
        if assistant_busy.is_set():
            consecutive_hits = 0
            time.sleep(0.05)
            continue

        if wake_proc is None:
            consecutive_hits = 0
            start_wake_stream()
            time.sleep(0.1)
            continue

        audio_data = wake_proc.stdout.read(bytes_per_chunk)
        if len(audio_data) < bytes_per_chunk:
            consecutive_hits = 0
            continue

        pcm = np.frombuffer(audio_data, dtype=np.int16)

        old_x = np.linspace(0, len(pcm) - 1, num=len(pcm))
        new_x = np.linspace(0, len(pcm) - 1, num=TARGET_CHUNK)
        pcm_16k = np.interp(new_x, old_x, pcm).astype(np.int16)

        prediction = wake_model.predict(pcm_16k)

        score = 0.0
        for key, value in prediction.items():
            if "alexa" in key.lower():
                score = float(value)
                break

        counter += 1
        if counter % 20 == 0:
            print(f"Alexa score: {score:.3f}")

        now = time.time()

        # Ignore a short window after re-opening the mic
        if (now - wake_stream_started_at) < WAKE_STREAM_WARMUP_SECONDS:
            consecutive_hits = 0
            continue

        # Require multiple consecutive hits
        if score >= WAKEWORD_THRESHOLD:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        if (
            consecutive_hits >= REQUIRED_CONSECUTIVE_HITS
            and not assistant_busy.is_set()
            and (now - last_trigger_time) > WAKE_DEBOUNCE_SECONDS
        ):
            print(f"Wake word detected: {score:.3f}")
            last_trigger_time = now
            consecutive_hits = 0
            trigger_event.set()
            time.sleep(1.0)


def wait_for_trigger():
    trigger_event.clear()
    trigger_event.wait()
    trigger_event.clear()


def record_audio():
    subprocess.run([
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", "5",
        "speech.wav"
    ], check=True)


def transcribe() -> str:
    segments, _ = WHISPER_MODEL.transcribe("speech.wav", language="en")
    return "".join(segment.text for segment in segments).strip()


def ask_ai(prompt: str) -> str:
    system_prompt = """
You are a voice assistant running on a Raspberry Pi AI clock.
Respond conversationally and very briefly.
Use one or two short sentences maximum.
Do not give long explanations unless specifically asked.
Avoid lists, markdown, and tangents.
""".strip()

    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    global last_trigger_time

    print("AI Clock Controller Started")

    web_server = start_web_server()
    open_clock()
    set_status('Press or say "Alexa" to activate')

    wake_thread = threading.Thread(target=wakeword_worker, daemon=True)
    wake_thread.start()

    try:
        while True:
            wait_for_trigger()
            assistant_busy.set()

            try:
                stop_wake_stream()

                set_status("Listening")
                print("Recording...")
                record_audio()

                set_status("Thinking")
                print("Transcribing...")
                user_text = transcribe()
                print("You said:", user_text)

                if not user_text:
                    set_status('Press or say "Alexa" to activate')
                    continue

                print("Querying Ollama...")
                reply = ask_ai(user_text)
                print("AI:", reply)

                if not reply:
                    set_status('Press or say "Alexa" to activate')
                    continue

                set_status("Speaking")
                speak(reply)

                set_status('Press or say "Alexa" to activate')

            except Exception as e:
                print("Error:", e)
                set_status('Press or say "Alexa" to activate')
                time.sleep(1)

            finally:
                last_trigger_time = time.time()
                time.sleep(POST_ASSISTANT_COOLDOWN_SECONDS)
                assistant_busy.clear()
                start_wake_stream()

    except KeyboardInterrupt:
        print("\nStopping controller.")
        shutdown_event.set()
        stop_wake_stream()
        set_status('Press or say "Alexa" to activate')
    finally:
        web_server.shutdown()


if __name__ == "__main__":
    main()

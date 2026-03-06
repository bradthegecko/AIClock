import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyaudio
import requests
from faster_whisper import WhisperModel
from openwakeword.model import Model

from tts import speak

STATUS_FILE = "status.json"
PORT = 8000
CLOCK_URL = f"http://127.0.0.1:{PORT}/clock_display.html"

MIC_DEVICE = "plughw:CARD=Device,DEV=0"
OLLAMA_MODEL = "llama3.2:3b"


WAKEWORD_THRESHOLD = 0.25
WAKEWORD_TARGET_RATE = 16000
WAKEWORD_TARGET_CHUNK = 1280
PYAUDIO_INPUT_INDEX = 2
PYAUDIO_INPUT_RATE = 44100
PYAUDIO_INPUT_CHUNK = 3528


WHISPER_MODEL = WhisperModel("tiny", compute_type="int8")

def start_web_server():
    return subprocess.Popen(
        ["python", "-m", "http.server", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
def set_status(text):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump({"assistant_status": text}, f)


def open_clock():
    subprocess.Popen([
        "chromium",
        "--kiosk",
        CLOCK_URL
    ])
    time.sleep(3)


def find_alexa_model():
    local_candidates = [
        Path("wakewords/alexa.onnx"),
        Path("wakewords/alexa_v0.1.onnx"),
        Path("wakewords/alexa.tflite"),
    ]

    for path in local_candidates:
        if path.exists():
            return str(path)

    try:
        import openwakeword
        package_dir = Path(openwakeword.__file__).resolve().parent
        matches = list(package_dir.rglob("*alexa*.onnx")) + list(package_dir.rglob("*alexa*.tflite"))
        if matches:
            return str(matches[0])
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not find an Alexa wake word model. Put it in the wakewords folder, "
        "for example wakewords/alexa.onnx."
    )

def wait_for_wake_word():
    print("Loading wake word model...")
    alexa_model_path = find_alexa_model()
    print("Using wake word model:", alexa_model_path)

    oww = Model(wakeword_model_paths=[alexa_model_path])

    pa = pyaudio.PyAudio()

    print("Using input device index:", PYAUDIO_INPUT_INDEX)
    print("Wake word input sample rate:", PYAUDIO_INPUT_RATE)
    print("Wake word input chunk:", PYAUDIO_INPUT_CHUNK)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=PYAUDIO_INPUT_RATE,
        input=True,
        input_device_index=PYAUDIO_INPUT_INDEX,
        frames_per_buffer=PYAUDIO_INPUT_CHUNK
    )

    print("Wake word listener ready. Say 'Alexa'.")

    counter = 0

    try:
        while True:
            audio_data = stream.read(PYAUDIO_INPUT_CHUNK, exception_on_overflow=False)
            pcm = np.frombuffer(audio_data, dtype=np.int16)

            # Resample from 44.1 kHz to 16 kHz, ending with exactly 1280 samples
            old_x = np.linspace(0, len(pcm) - 1, num=len(pcm))
            new_x = np.linspace(0, len(pcm) - 1, num=WAKEWORD_TARGET_CHUNK)
            pcm_16k = np.interp(new_x, old_x, pcm).astype(np.int16)

            prediction = oww.predict(pcm_16k)

            score = 0.0
            for key, value in prediction.items():
                if "alexa" in key.lower():
                    score = float(value)
                    break

            counter += 1
            if counter % 10 == 0:
                print(f"Alexa score: {score:.3f}")

            if score >= WAKEWORD_THRESHOLD:
                print(f"Wake word detected: {score:.3f}")
                time.sleep(0.8)
                return

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
   


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


def transcribe():
    segments, _ = WHISPER_MODEL.transcribe("speech.wav", language="en")
    return "".join(segment.text for segment in segments).strip()


def ask_ai(prompt):
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
    print("AI Clock Controller Started")
    web_server = start_web_server()
    open_clock()
    set_status("Idle — waiting for wake word")

    while True:
        try:
            wait_for_wake_word()

            set_status("Listening")
            print("Recording...")
            record_audio()

            set_status("Thinking")
            print("Transcribing...")
            user_text = transcribe()
            print("You said:", user_text)

            if not user_text:
                set_status("Idle — waiting for wake word")
                continue

            print("Querying Ollama...")
            reply = ask_ai(user_text)
            print("AI:", reply)

            set_status("Speaking")
            speak(reply)

            set_status("Idle — waiting for wake word")

        except KeyboardInterrupt:
            print("\nStopping controller.")
            set_status("Idle — waiting for wake word")
            break

        except Exception as e:
            print("Error:", e)
            set_status("Idle — waiting for wake word")
            time.sleep(1)


if __name__ == "__main__":
    main()

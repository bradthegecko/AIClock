import json
import subprocess
import threading
import time
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote_plus
import re




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
REQUIRED_CONSECUTIVE_HITS = 1
WAKE_DEBOUNCE_SECONDS = 6
POST_ASSISTANT_COOLDOWN_SECONDS = 6
WAKE_STREAM_WARMUP_SECONDS = 2.0


CHAT_HISTORY = []
MAX_HISTORY_EXCHANGES = 3


CURRENT_INFO_KEYWORDS = [
    "current", "today", "latest", "recent", "now", "died", "attack", "war", "ruler",
    "president", "vice president", "governor", "mayor", 
    "prime minister", "emir", "king", "queen", "ceo", "shortest",
    "weather", "news", "score", "stock","movie","show", "biggest", "tallest", "price"
]



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
        "-d", "8",
        "speech.wav"
    ], check=True)


def transcribe() -> str:
    segments, _ = WHISPER_MODEL.transcribe("speech.wav", language="en")
    return "".join(segment.text for segment in segments).strip()

def build_history_prompt(user_text: str) -> str:
    history_lines = []

    for item in CHAT_HISTORY[-(MAX_HISTORY_EXCHANGES * 2):]:
        history_lines.append(f"{item['role']}: {item['content']}")

    history_lines.append(f"User: {user_text}")

    return "\n".join(history_lines)



def ask_ai(prompt: str) -> str:
    system_prompt = """
You are a voice assistant running on a Raspberry Pi AI clock.
Respond conversationally and very briefly.
Use one or two short sentences maximum.
Do not give long explanations unless specifically asked.
Avoid lists, markdown, and tangents.

If the user asks for current events, recent facts, live information, office holders,
news, weather, sports, or something that may have changed recently, or says the word current, or asks for who a president or political office holder is, or asks about a  president of a university, or says anyting that has to do with pop culture, or says anything that has to do with movies or tv shows, respond with exactly:
SEARCH_WEB

For stable general-knowledge questions, answer directly without using SEARCH_WEB.
""".strip()

    conversation_prompt = build_history_prompt(prompt)

    full_prompt = f"{system_prompt}\n\n{conversation_prompt}\nAssistant:"

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


def search_web(query: str, max_results: int = 3) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    html = response.text

    results = []

    blocks = re.findall(
        r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>.*?(?:<a class="result__snippet".*?>(.*?)</a>|<div class="result__snippet">(.*?)</div>)',
        html,
        flags=re.DOTALL
    )

    for block in blocks[:max_results]:
        link = block[0]
        title_html = block[1]
        snippet_html = block[2] or block[3] or ""

        title = re.sub(r"<.*?>", "", title_html)
        snippet = re.sub(r"<.*?>", "", snippet_html)

        title = re.sub(r"\s+", " ", title).strip()
        snippet = re.sub(r"\s+", " ", snippet).strip()

        # keep snippets short so Ollama doesn't choke
        snippet = snippet[:180]

        results.append({
            "title": title,
            "snippet": snippet,
            "link": link
        })

    return results

def format_search_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."

    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(f"{i}. {item['title']}")
        lines.append(f"Snippet: {item['snippet']}")
        lines.append("")

    return "\n".join(lines).strip()

def summarize_search_results(user_question: str, search_results_text: str) -> str:
    system_prompt = """
You are a voice assistant running on a Raspberry Pi AI clock.
Use the provided web search results to answer the user's question.
Summarize the results naturally and briefly.
Do not read links aloud.
Do not quote large passages.
Prefer the most likely correct and current answer from the search results.
If the search results are weak or conflicting, say so briefly.
Use one to three short sentences maximum.
""".strip()

    full_prompt = f"""
{system_prompt}

User question:
{user_question}

Web search results:
{search_results_text}

Assistant:
""".strip()

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


def should_force_web_search(user_text: str) -> bool:
    text = user_text.lower()
    return any(keyword in text for keyword in CURRENT_INFO_KEYWORDS)


def answer_question(user_text: str) -> str:
    global CHAT_HISTORY

    if should_force_web_search(user_text):
        print("Forcing web search for current info...")

        try:
            search_results = search_web(user_text, max_results=3)
        except Exception as e:
            print("Web search error:", e)
            return "I couldn't search the web right now."

        if not search_results:
            return "I couldn't find anything useful online."

        search_results_text = format_search_results(search_results)

        try:
            reply = summarize_search_results(user_text, search_results_text)
        except Exception as e:
            print("Search summary error:", e)
            top = search_results[0]
            snippet = top["snippet"].strip()
            reply = snippet if snippet else f"I found something relevant: {top['title']}."
    else:
        try:
            first_reply = ask_ai(user_text)
        except Exception as e:
            print("Primary AI error:", e)
            return "I ran into a problem answering that."

        if first_reply == "SEARCH_WEB":
            print("AI requested web search...")

            try:
                search_results = search_web(user_text, max_results=3)
            except Exception as e:
                print("Web search error:", e)
                return "I couldn't search the web right now."

            if not search_results:
                return "I couldn't find anything useful online."

            search_results_text = format_search_results(search_results)

            try:
                reply = summarize_search_results(user_text, search_results_text)
            except Exception as e:
                print("Search summary error:", e)
                top = search_results[0]
                snippet = top["snippet"].strip()
                reply = snippet if snippet else f"I found something relevant: {top['title']}."
        else:
            reply = first_reply

    CHAT_HISTORY.append({"role": "User", "content": user_text})
    CHAT_HISTORY.append({"role": "Assistant", "content": reply})
    CHAT_HISTORY = CHAT_HISTORY[-(MAX_HISTORY_EXCHANGES * 2):]

    return reply


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

                print("Answering question...")
                reply = answer_question(user_text)
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

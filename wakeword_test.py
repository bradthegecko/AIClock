import os
import subprocess
import time
import numpy as np
from openwakeword.model import Model

MIC_DEVICE = "plughw:2,0"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample

model = Model()

print("Available wakeword models:", flush=True)
print(model.models.keys(), flush=True)
print("\nListening only for: ALEXA", flush=True)
print("Press Ctrl+C to stop.\n", flush=True)

cmd = [
    "arecord",
    "-D", MIC_DEVICE,
    "-f", "S16_LE",
    "-r", str(SAMPLE_RATE),
    "-c", "1",
    "-t", "raw",
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    bufsize=0,
)

last_detect_time = 0
last_print_time = 0
cooldown_seconds = 2

buffer = b""

try:
    while True:
        data = os.read(proc.stdout.fileno(), 1024)
        if not data:
            continue

        buffer += data

        while len(buffer) >= CHUNK_BYTES:
            chunk = buffer[:CHUNK_BYTES]
            buffer = buffer[CHUNK_BYTES:]

            pcm = np.frombuffer(chunk, dtype=np.int16)
            prediction = model.predict(pcm)

            alexa_score = float(prediction.get("alexa", 0.0))
            mic_level = int(np.abs(pcm).mean())
            now = time.time()

            if now - last_print_time > 0.5:
                print(f"Mic level: {mic_level} | Alexa score: {alexa_score:.3f}", flush=True)
                last_print_time = now

            if alexa_score > 0.15 and (now - last_detect_time) > cooldown_seconds:
                print(f"\nWAKE WORD DETECTED: ALEXA  score={alexa_score:.3f}\n", flush=True)
                last_detect_time = now

except KeyboardInterrupt:
    print("\nStopping wakeword listener...", flush=True)

finally:
    proc.terminate()
    proc.wait()

import subprocess
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5
OUTPUT_FILE = "speech.wav"
MIC_DEVICE = "plughw:CARD=Device,DEV=0"

print("Recording for 5 seconds...")

subprocess.run([
    "arecord",
    "-D", MIC_DEVICE,
    "-f", "S16_LE",
    "-r", str(SAMPLE_RATE),
    "-c", "1",
    "-d", str(DURATION),
    OUTPUT_FILE
], check=True)

print("Loading speech model...")
model = WhisperModel("base", compute_type="int8")

print("Transcribing...")
segments, info = model.transcribe(OUTPUT_FILE, language="en")

print("\nYou said:")
for segment in segments:
    print(segment.text)

print("\nDone.")

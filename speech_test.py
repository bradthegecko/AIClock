import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import numpy as np

# Microphone device index from sounddevice query
MIC_DEVICE = 1

# Recording settings
SAMPLE_RATE = 48000
DURATION = 5

print("Recording for 5 seconds...")

audio = sd.rec(
    int(SAMPLE_RATE * DURATION),
    samplerate=SAMPLE_RATE,
    channels=1,
    device=MIC_DEVICE,
    dtype="int16",
)

sd.wait()

print("Saving audio...")
write("speech.wav", SAMPLE_RATE, audio)

print("Loading speech model...")
model = WhisperModel("base", compute_type="int8")

print("Transcribing...")

segments, info = model.transcribe("speech.wav")

print("\nYou said:")

for segment in segments:
    print(segment.text)

print("\nDone.")

import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import requests
import pyttsx3

MIC_DEVICE = 1
SAMPLE_RATE = 48000
DURATION = 5

model = WhisperModel("tiny", compute_type="int8")

tts = pyttsx3.init()

print("Say something...")

audio = sd.rec(
    int(SAMPLE_RATE * DURATION),
    samplerate=SAMPLE_RATE,
    channels=1,
    device=MIC_DEVICE,
    dtype="int16"
)

sd.wait()

write("speech.wav", SAMPLE_RATE, audio)

segments, info = model.transcribe("speech.wav")

text = ""
for segment in segments:
    text += segment.text

print("You said:", text)

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": text,
        "stream": False
    }
)

reply = response.json()["response"]

print("AI:", reply)

tts.say(reply)
tts.runAndWait()

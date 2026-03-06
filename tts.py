import subprocess

VOICE_MODEL = "voices/en_US-libritts_r-medium.onnx"
SPEAKER_ID = 643
AUDIO_DEVICE = "plughw:CARD=UACDemoV10,DEV=0"

def speak(text):
    cmd = [
        "piper",
        "--model", VOICE_MODEL,
        "--speaker", str(SPEAKER_ID),
        "--output-raw",
    ]

    p1 = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    p2 = subprocess.Popen(
        [
            "aplay",
            "-D", AUDIO_DEVICE,
            "-r", "22050",
            "-f", "S16_LE",
            "-t", "raw",
        ],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    p1.stdin.write(text.encode("utf-8"))
    p1.stdin.close()

    p2_out, p2_err = p2.communicate()
    p1.wait()
    p1_err = p1.stderr.read()

    if p1.returncode not in (0, None):
        print("Piper error:")
        print(p1_err.decode(errors="ignore"))

    if p2.returncode not in (0, None):
        print("aplay error:")
        print(p2_err.decode(errors="ignore"))

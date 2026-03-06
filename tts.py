import subprocess

VOICE_MODEL = "voices/en_US-libritts_r-medium.onnx"
SPEAKER_ID = 643


def speak(text):

    cmd = [
        "piper",
        "--model",
        VOICE_MODEL,
        "--speaker",
        str(SPEAKER_ID),
        "--output-raw"
    ]

    p1 = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    p2 = subprocess.Popen(
        ["aplay", "-D", "plughw:3,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=p1.stdout
    )

    p1.stdin.write(text.encode())
    p1.stdin.close()

    p2.communicate()

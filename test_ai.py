import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": "Say hello from the Raspberry Pi AI.",
        "stream": False
    },
    timeout=120
)

response.raise_for_status()
data = response.json()
print(data["response"])

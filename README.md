# AIClock ("Oliver")
A voice-activated AI smart clock powered by a Raspberry Pi that combines a touchscreen interface, wake-word detection (openWakeWord), speech recognition (Whisper), local AI (Ollama 3.2:3B), text-to-speech (Piper), and optional internet search fallback.

The clock displays time and weather while listening for a custom wake word. Once activated, it records speech, processes the request using a local AI model, optionally searches the internet for current information, and speaks the response.

---

# Demo

## Video Demo

To be inserted

## Images
### Clock Interface
To be inserted

### Hardware Setup
To be inserted

# Features
- Custom wake word detection
- Touchscreen interface (optimized for 800x480 display)
- Local AI assistant using Ollama
- Whisper speech transcription
- Piper text-to-speech
- Internet search fallback for current information
- Weather display with forecast
- Chat memory (3 interactions)
- Touch activation button
- Transparent UI widgets
- Custom backgrounds

# System Architecture

## Voice Pipeline:
Wake Word

   ↓
   
Audio Recording

   ↓
   
Speech Transcription (Whisper)

   ↓
   
AI Processing (Ollama)

   ↓
   
Optional Web Search

   ↓
   
AI Summary

   ↓
   
Text To Speech (Piper)

## Display Pipeline:
controller.py

   ↓
      
status.json

   ↓
      
clock_display.html

   ↓
      
Chromium fullscreen

# Hardware
## Required Hardware:
- Raspberry Pi 5 (8GB or more recommended)
- 800x480 touchscreen display 
- USB microphone
- USB Speaker
- Internet connection (recommended for internet fallbacks)
  
## Optional Hardware:

- Cooling fan (Highly Recomended)
- NVMe SSD Storage (Highly Recomended)

---

# Software Stack

| Component | Purpose |
|--------|--------|
| Python | Main controller |
| openWakeWord | Wake word detection |
| Whisper | Speech transcription |
| Ollama | Local AI assistant |
| Piper | Text to speech |
| Chromium | Fullscreen UI |
| HTML/CSS/JS | Clock interface |

---

# Project Structure

      AIClock/
      │
      ├── controller.py
      ├── clock_display.html
      ├── tts.py
      ├── status.json
      │
      ├── backgrounds/
      │ └── background1.jpg
      │
      ├── wakewords/
      │ └── custom_wakeword.onnx
      │
      ├── requirements.txt
      └── README.md

--- 

# Installation
## Clone the repository:
      git clone https://github.com/bradthegecko/AIClock.git
      cd AIClock
## Create a virtual environment:
      python -m venv venv
      source venv/bin/activate

## Install dependencies:
      pip install -r requirements.txt

## Install an Ollama model (llama3.2:3b):

      ollama pull llama3.2

---

# Running the Clock
## start the main controller:
      python controller.py

The display will automatically launch in fullscreen mode

---
# Wake word 
Wake word detection uses **openWakeWord**.
supported model format: .onnx

place wake word models in the 'wakewords/' directory and update the model path in `controller.py`.

---
# Weather System
Weather information is retrieved from an API and displayed on the interface.

Displayed data includes:

- current temperature
- today's high and low
- wind speed
- humidity
- precipitation chance
- tomorrow's forecast

---

# Internet Search Fallback

If the AI cannot answer a question using its local knowledge, the system performs a web search.

Workflow:

User question

↓

Local AI attempt

↓

If AI returns SEARCH_WEB

↓

DuckDuckGo search

↓

AI summarizes results


---

# User Interface

The interface is designed specifically for **800x480 touchscreen displays**.

Widgets include:

- clock display
- voice activation button
- current weather
- weather details
- tomorrow forecast

The UI uses semi-transparent panels over customizable background images.

---

# Troubleshooting

### Microphone or speaker not detected

Check available audio devices:

      arecord -l


Update the microphone and speaker device in `controller.py`.

---

### Wake word not triggering

Verify the wake word model path and microphone input levels.

---

### AI not responding

Ensure Ollama is running:
ollama serve


---
# Future Improvements

Potential upgrades:

- improved conversation memory
- Timer and Alarm support
- calendar integration
- home automation support
- music/video playback
- offline weather

---

# License

MIT License

---

# Author

Bradley Ayieko  
University of Virginia


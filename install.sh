#!/bin/bash

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "Virtual environment created successfully!"
else
  echo "Virtual environment already exists."
fi
source .venv/bin/activate
sudo apt update
sudo apt install portaudio19-dev python3-pyaudio espeak ffmpeg libespeak1
pip install pyaudio
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install --no-cache-dir 'transformers[torch]'
pip install -r requirements.txt
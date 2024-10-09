# onnx-aec
A playground for experimenting with acoustic echo cancellation using a microphone, speaker, and ONNX.

## 0. Environment
+ Ubuntu22.04
+ Python3.10

## 1. Test
```bash
sudo apt-get update \
&& sudo apt-get install libsndfile1 python3-dev

pip install \
SoundCard==0.4.3 \
soundfile==0.12.1 \
sounddevice==0.5.0
```

### 1-1. View a list of available audio devices
```bash
python aec.py

=== Available audio devices ===
=== Microphones ===
Index 0: Full HD webcam Mono
Index 1: Monitor of USB Microphone Analog Stereo
Index 2: USB Microphone Digital Stereo (IEC958)
Index 3: Monitor of Built-in Audio Digital Stereo (IEC958)
Index 4: Monitor of GA104 High Definition Audio Controller Digital Stereo (HDMI 2)
=== Speakers ===
Index 0: USB Microphone Analog Stereo
Index 1: Built-in Audio Digital Stereo (IEC958)
Index 2: GA104 High Definition Audio Controller Digital Stereo (HDMI 2)
==============================
Usage: aec.py [output_mode] [mic_index] [output_device_index]
output_mode: "wav" (default) or "speaker"
mic_index: Index of the microphone to use (optional, default is the system default microphone)
output_device_index: Index of the speaker used for outputting the echo-cancelled audio
```

### 1-2. Mic + Speaker (reference_speaker) -> WAV
```bash
python aec.py wav

ctrl + c
```

### 1-3. Mic + Speaker (reference_speaker) -> Speaker
```bash
python aec.py speaker 0 1

ctrl + c
```

### 1-4. Mic + Speaker (reference_speaker) -> Virtual Speaker
```bash
python aec.py speaker

ctrl + c
```

## 2. ONNX Runtime Web
https://github.com/microsoft/onnxruntime/tree/main/js/web#onnx-runtime-web

## 3. ONNX Runtime Web - Operators
https://github.com/microsoft/onnxruntime/tree/main/js/web#Operators

## 4. Cited
1. https://github.com/microsoft/AEC-Challenge
2. https://github.com/PINTO0309/speexdsp-python

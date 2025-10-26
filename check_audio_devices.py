#!/usr/bin/env python3
"""Quick script to check available audio devices."""

import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    
    print("Available Audio Devices:")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
        print(f"  Max Input Channels: {info['maxInputChannels']}")
        print(f"  Max Output Channels: {info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {info['defaultSampleRate']}")
        print()
    
    # Get default devices
    try:
        default_input = p.get_default_input_device_info()
        print(f"Default Input Device: {default_input['index']} - {default_input['name']}")
    except:
        print("No default input device found")
    
    try:
        default_output = p.get_default_output_device_info()
        print(f"Default Output Device: {default_output['index']} - {default_output['name']}")
    except:
        print("No default output device found")
    
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()

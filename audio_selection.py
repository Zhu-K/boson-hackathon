import pyaudio

p = pyaudio.PyAudio()

print("=== INPUT DEVICES (Microphones) ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0:
        print(f"Index {i}: {info['name']} - {info['maxInputChannels']} channels")

print("\n=== OUTPUT DEVICES (Speakers) ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxOutputChannels') > 0:
        print(f"Index {i}: {info['name']} - {info['maxOutputChannels']} channels")

print(f"\nDefault input device index: {p.get_default_input_device_info()['index']}")
print(f"Default output device index: {p.get_default_output_device_info()['index']}")

p.terminate()
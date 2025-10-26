from collections import deque
from typing import Tuple, List, Dict

import numpy as np
import pyaudio
import wave
import os


def record_microphone_segment(
    threshold: float = 500.0,
    rate: int = 16_000,
    chunk_size: int = 1024,
    pre_seconds: float = 1.0,
    post_seconds: float = 1.0,
    silence_tolerance: float = 1.5,
    max_duration: float = 30.0,
    input_device_index: int = 0,
    emit_callback=None,
) -> bytes:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index,
    )

    pre_max_chunks = int(pre_seconds * rate / chunk_size)
    post_max_chunks = int(post_seconds * rate / chunk_size)
    silence_max_chunks = int(silence_tolerance * rate / chunk_size)
    max_chunks = int(max_duration * rate / chunk_size)

    pre_buffer: deque = deque(maxlen=pre_max_chunks)
    frames: list[bytes] = []
    recording = False
    silent_chunks = 0
    total_chunks = 0

    try:
        if emit_callback:
            emit_callback('status', {'step': 'listening', 'message': '👂 Listening... Start speaking!'})

        while True:
            data = stream.read(chunk_size)
            total_chunks += 1

            if total_chunks >= max_chunks:
                if emit_callback:
                    emit_callback('status', {'step': 'max_duration', 'message': '⏱️ Maximum duration reached'})
                break

            pre_buffer.append(data)

            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))

            if recording:
                frames.append(data)
                if rms < threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                if silent_chunks >= silence_max_chunks:
                    if emit_callback:
                        emit_callback('status', {'step': 'silence_detected', 'message': '🤫 Silence detected, stopping...'})
                    for _ in range(post_max_chunks):
                        trailing = stream.read(chunk_size)
                        frames.append(trailing)
                    break
            else:
                if rms >= threshold:
                    recording = True
                    if emit_callback:
                        emit_callback('status', {'step': 'speech_detected', 'message': '🎤 Speech detected, recording...'})
                    frames.extend(list(pre_buffer))
                    frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return b"".join(frames)


def load_wav_file_as_pcm(path: str, target_rate: int = 24_000, volume: float = 0.25) -> Tuple[bytes, int]:
    if not os.path.exists(path):
        return b"", target_rate
    with wave.open(path, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
        scaled = np.clip(audio_array * volume, -32768, 32767).astype(np.int16)
        return scaled.tobytes(), sample_rate


def list_audio_devices() -> Dict:
    devices: List[Dict] = []
    default_index = None
    p = pyaudio.PyAudio()
    try:
        try:
            default_index = p.get_default_input_device_info().get('index')
        except Exception:
            default_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': info.get('name', f'Device {i}'),
                })
    finally:
        try:
            p.terminate()
        except Exception:
            pass
    return {'devices': devices, 'default_index': default_index}


from __future__ import annotations
import argparse
import base64
import io
import os
import wave
from typing import Iterable, Iterator
import time

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

import numpy as np

# Load environment variables
load_dotenv()


def b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def get_client() -> OpenAI:
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://hackathon.boson.ai/v1")


def translate_emotion(client: OpenAI, user_prompt: str, translator_prompt: str) -> str:
    """Run the translation into either angry or sarcastic text."""
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=[
            {"role": "system", "content": translator_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    return response.choices[0].message.content


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:
    """Transcribe the recorded audio bytes using Higgs Audio."""
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            wf.writeframes(audio_bytes)
        wav_data = buffer.getvalue()

    audio_b64 = base64.b64encode(wav_data).decode("utf-8")

    response = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe this audio for me."},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            },
        ],
        max_completion_tokens=256,
        temperature=0.0,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def tts_generate_streaming(client: OpenAI, text: str, tts_prompt: str, voice_prompt: str, ref_path: str) -> Iterator:
    """Generate TTS for the given text using Higgs Audio."""
    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": tts_prompt},
            {"role": "user", "content": voice_prompt},
            {
                "role": "assistant",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": b64(ref_path), "format": "wav"}},
                ],
            },
            {"role": "user", "content": f"[SPEAKER1] {text}"},
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=True,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )


def record_microphone_segment(threshold: float = 1500.0, rate: int = 16_000,
                              chunk_size: int = 1024, duration: float | None = None) -> bytes:
    """
    Record audio from the microphone when speech is detected.
    If 'duration' is provided, record for that many seconds instead of until silence.
    """
    # smoe comment
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=2,
        frames_per_buffer=chunk_size,
    )

    frames: list[bytes] = []
    print("🎙️ Speak now...")

    start_time = time.time()
    try:
        while True:
            data = stream.read(chunk_size)
            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))
            frames.append(data)

            # Stop if duration reached
            if duration and (time.time() - start_time >= duration):
                print(f"⏱️ Recording stopped after {duration} seconds.")
                break

            # Stop if silence detected (for threshold-based mode)
            if not duration and len(frames) > 0 and rms < threshold:
                break
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return b"".join(frames)


def play_streaming_audio(chunks: Iterable) -> None:
    """Play streaming PCM audio chunks from API."""
    p = pyaudio.PyAudio()
    audio_stream = None
    try:
        for chunk in chunks:
            delta = chunk.choices[0].delta
            audio_field = getattr(delta, "audio", None)
            if not audio_field:
                continue
            b64_data = audio_field.get("data") if isinstance(audio_field, dict) else getattr(audio_field, "data", None)
            if not b64_data:
                continue
            pcm_bytes = base64.b64decode(b64_data)
            if audio_stream is None:
                audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24_000, output=True)
            audio_stream.write(pcm_bytes)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        p.terminate()


def play_wav_file(path: str, volume: float = 1.0) -> None:
    """Play a WAV file at a given volume (replaces deprecated audioop)."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return

    with wave.open(path, "rb") as wf:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        chunk = 1024
        data = wf.readframes(chunk)
        while data:
            # Convert bytes to numpy array, scale, and clip
            audio_array = np.frombuffer(data, dtype=np.int16)
            scaled = np.clip(audio_array * volume, -32768, 32767).astype(np.int16)
            stream.write(scaled.tobytes())
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()


def main():
    parser = argparse.ArgumentParser(description="Higgs Audio Emotion Demo")
    parser.add_argument("--mode", choices=["angry", "sarcastic"], default="angry",
                        help="Choose emotional mode (angry or sarcastic)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Set maximum recording duration in seconds (optional)")
    args = parser.parse_args()

    # Import correct prompt file dynamically
    if args.mode == "angry":
        from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
        laugh_track = "./audios/sitcom_laugh_track.wav"
    else:
        from prompts_sarcastic import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT
        VOICE_REFERENCE_PATH = "./audios/sarcasm_clip.wav"
        laugh_track = "./audios/sitcom_laugh_track.wav"  # Optional alt track

    client = get_client()

    print(f"🎭 Mode selected: {args.mode.capitalize()}")
    if args.duration:
        print(f"🎧 Fixed recording duration: {args.duration} seconds")
    print("Listening for speech...")

    # Step 1: Record
    audio_bytes = record_microphone_segment(duration=args.duration)

    # Step 2: Transcribe
    captured_speech = transcribe_audio(client, audio_bytes)

    # Step 3: Translate
    emotional_text = translate_emotion(client, captured_speech, TRANSLATOR_SYSTEM_PROMPT)
    print(f"Rephrased text: {emotional_text}\nGenerating speech...")

    # Step 4: TTS
    stream_iter = tts_generate_streaming(client, emotional_text,
                                         TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH)
    play_streaming_audio(stream_iter)

    # Step 5: Add laugh track
    play_wav_file(laugh_track, volume=0.25)

    print("✅ Done.")


if __name__ == "__main__":
    main()

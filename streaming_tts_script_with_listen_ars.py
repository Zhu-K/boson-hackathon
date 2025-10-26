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

# Speaker tag constant for TTS prompts
SPEAKER_TAG = "[SPEAKER1]"


class VoiceReference:
    """Handle voice reference audio for TTS, supporting both file paths and raw audio bytes."""
    
    def __init__(self, audio: str | bytes, text: str) -> None:
        """
        Initialize voice reference.
        
        Args:
            audio: Either a file path (str) or raw audio bytes
            text: The reference text/prompt for this voice
        """
        if isinstance(audio, str):
            # File path - read and encode
            self.voice_audio = b64(audio)
        else:
            # Raw bytes - convert to WAV format first
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit samples
                    wf.setframerate(16_000)  # Must match record_microphone_segment()
                    wf.writeframes(audio)
                wav_data = buffer.getvalue()
            # Base64-encode the WAV for transmission
            self.voice_audio = base64.b64encode(wav_data).decode("utf-8")
        
        self.text = text


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


def tts_generate_streaming(client: OpenAI, text: str, tts_prompt: str, voice_ref: VoiceReference) -> Iterator:
    """Generate TTS for the given text using Higgs Audio.
    
    Args:
        client: OpenAI client
        text: Text to convert to speech
        tts_prompt: System prompt for TTS
        voice_ref: VoiceReference object containing audio and text
    """
    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": tts_prompt},
            {"role": "user", "content": voice_ref.text},
            {
                "role": "assistant",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": voice_ref.voice_audio, "format": "wav"}},
                ],
            },
            {"role": "user", "content": f"{SPEAKER_TAG} {text}"},
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


# Mode configuration mapping (for emotion/text translation)
MODE_CONFIG = {
    "angry": {
        "module": "prompts",
        "description": "Angry"
    },
    "sarcastic": {
        "module": "prompts_sarcastic",
        "description": "Sarcastic"
    },
    "roast": {
        "module": "prompts_roast",
        "description": "Roast"
    }
}

# Voice configuration mapping (for TTS voice characteristics)
VOICE_CONFIG = {
    "keegan": {
        "reference_path": "./audios/keegan.wav",
        "reference_prompt": f"{SPEAKER_TAG} Oh and CNN, thank you so much for the wall to wall ebola coverage. For two whole weeks, we were one step away from the walking dead!",
        "description": "Keegan-Michael Key (energetic, comedic)"
    },
    "stephen": {
        "reference_path": "./audios/stephen_colbert.wav",
        "reference_prompt": f"{SPEAKER_TAG} Oh sarcasm works great..sarcasm is absolutely the best thing for a president to do in the middle of a pandemic. You're doing amaaazing, 'Mr. President'.",
        "description": "Stephen Colbert (dry, witty)"
    },
    "ricky": {
        "reference_path": "./audios/ricky_gervaise.wav",
        "reference_prompt": f"{SPEAKER_TAG} All the best actors have jumped to Netflix and HBO, you know, and the actors who just do hollywood movies now do fantasy adventure nonsense..they wear masks and capes and reaaally tight costumes. Their job isn't acting anymore! It's going to the gym twice a day and taking steroids really.",
        "description": "Ricky Gervais (brutally honest)"
    },
    "my_voice": {
        "reference_path": None,  # Will be set dynamically from recorded audio
        "reference_prompt": None,  # Will be set from transcription
        "description": "Your own voice (cloned from recording)"
    }
}

# Shared configuration
LAUGH_TRACK_PATH = "./audios/sitcom_laugh_track.wav"


def load_mode_config(mode: str) -> tuple:
    """Load configuration for the specified emotion mode.
    
    Args:
        mode: The emotion mode (angry, sarcastic, or roast)
        
    Returns:
        Tuple of (TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT)
    """
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(MODE_CONFIG.keys())}")
    
    module_name = MODE_CONFIG[mode]["module"]
    
    try:
        # Dynamic import of the appropriate prompts module
        prompts_module = __import__(module_name)
        return (
            prompts_module.TRANSLATOR_SYSTEM_PROMPT,
            prompts_module.TTS_SYSTEM_PROMPT,
            prompts_module.LANGUAGE_TEMPLATE
        )
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' is missing required attributes: {e}")

def load_voice_config(voice: str, recorded_audio: bytes = None, transcription: str = None) -> VoiceReference:
    """Load configuration for the specified voice.
    
    Args:
        voice: The voice name (keegan, stephen, ricky, or my_voice)
        recorded_audio: The recorded audio bytes (required for my_voice)
        transcription: The transcribed text (required for my_voice)
        
    Returns:
        VoiceReference object
    """
    if voice not in VOICE_CONFIG:
        raise ValueError(f"Unknown voice: {voice}. Valid voices: {list(VOICE_CONFIG.keys())}")
    
    config = VOICE_CONFIG[voice]
    
    # Special handling for voice cloning
    if voice == "my_voice":
        if recorded_audio is None or transcription is None:
            raise ValueError("Voice cloning requires recorded_audio and transcription")
        # Use the recorded audio and transcription as reference
        reference_prompt = f"{SPEAKER_TAG} {transcription}"
        return VoiceReference(recorded_audio, reference_prompt)
    
    # Pre-recorded voice from file
    return VoiceReference(config["reference_path"], config["reference_prompt"])


def main():
    parser = argparse.ArgumentParser(description="Higgs Audio Emotion Demo with Voice Recording")
    parser.add_argument(
        "--mode",
        choices=list(MODE_CONFIG.keys()),
        default="angry",
        help="Choose emotional mode: " + ", ".join([f"{k} ({v['description']})" for k, v in MODE_CONFIG.items()])
    )
    parser.add_argument(
        "--voice",
        choices=list(VOICE_CONFIG.keys()),
        default="keegan",
        help="Choose voice: " + ", ".join([f"{k} ({v['description']})" for k, v in VOICE_CONFIG.items()])
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Set maximum recording duration in seconds (optional)"
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Choose output language: english or mandarin"
    )
    args = parser.parse_args()

    # Load mode configuration (for emotion translation)
    try:
        TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE = load_mode_config(args.mode)
        if args.language is not "english":
            language_instruction = LANGUAGE_TEMPLATE.format(language=args.language)
        else:
            language_instruction = ""
            
        llm_system_prompt = TRANSLATOR_SYSTEM_PROMPT.format(language_instruction=language_instruction, language_instruction_repeated=language_instruction)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"❌ Error loading mode configuration: {e}")
        return

    client = get_client()

    print(f"🎭 Mode: {args.mode.capitalize()} ({MODE_CONFIG[args.mode]['description']})")
    print(f"🎤 Voice: {args.voice.capitalize()} ({VOICE_CONFIG[args.voice]['description']})")
    print(f"🌐 Language: {args.language.capitalize()}")
    if args.duration:
        print(f"🎧 Fixed recording duration: {args.duration} seconds")
    print("🎵 Listening for speech...")
    print()

    # Step 1: Record
    audio_bytes = record_microphone_segment(duration=args.duration)

    # Step 2: Transcribe
    captured_speech = transcribe_audio(client, audio_bytes)
    
    # Load voice configuration (for TTS voice)
    # For voice cloning, we need the recorded audio and transcription
    try:
        voice_reference = load_voice_config(
            args.voice, 
            recorded_audio=audio_bytes if args.voice == "my_voice" else None,
            transcription=captured_speech if args.voice == "my_voice" else None
        )
    except ValueError as e:
        print(f"❌ Error loading voice configuration: {e}")
        return
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Verify laugh track exists
    if not os.path.exists(LAUGH_TRACK_PATH):
        print(f"⚠️  Warning: Laugh track file not found: {LAUGH_TRACK_PATH}")

    # Step 3: Translate
    emotional_text = translate_emotion(client, captured_speech, llm_system_prompt)
    print(f"Rephrased text: {emotional_text}\nGenerating speech...")

    # Step 4: TTS
    stream_iter = tts_generate_streaming(client, emotional_text, TTS_SYSTEM_PROMPT, voice_reference)
    play_streaming_audio(stream_iter)

    # Step 5: Add laugh track
    play_wav_file(LAUGH_TRACK_PATH, volume=0.25)

    print("✅ Done.")


if __name__ == "__main__":
    main()

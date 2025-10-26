from __future__ import annotations

import argparse
import base64
import io
import os
import re
import wave
from typing import Iterable, Iterator, Optional

import numpy as np
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from a .env file if present.
load_dotenv()

# Speaker tag constant for TTS prompts
SPEAKER_TAG = "[SPEAKER1]"


def b64(path: str) -> str:
    """Return the base64 encoding of the given file on disk."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def get_client() -> OpenAI:
    """Initialise and return an OpenAI client configured for BosonAI."""
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://hackathon.boson.ai/v1")


def translate_emotion(client: OpenAI, user_prompt: str, translator_prompt: str) -> str:
    """Translate a calm user prompt into an emotional version (angry or sarcastic)."""
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=[
            {"role": "system", "content": translator_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    # Remove the `<think>...</think>` sections returned by Qwen models.
    # out = re.sub(r"<think>...</think>", "", content, flags=re.DOTALL).strip()
    print(content)
    return content

def tts_generate_streaming(client: OpenAI, text: str, tts_prompt: str, voice_prompt: str, ref_path: str) -> Iterator:
    """Create a streaming TTS completion from the Higgs audio model."""

    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": tts_prompt},
            {"role": "user", "content": voice_prompt},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64(ref_path), "format": "wav"},
                    }
                ],
            },
            {"role": "user", "content": f"{SPEAKER_TAG} {text}"},
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=True,  # enable streaming for real‑time audio playback
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )


def play_streaming_audio(chunks: Iterable) -> None:
    """Decode and play streaming audio chunks returned by the API.

    The API sends audio as a sequence of base64 encoded PCM buffers.  This
    function iterates over the chunks, extracts the audio payload when
    available and writes it directly to a PyAudio output stream.  The first
    audio payload triggers the creation of the stream and defines the
    playback parameters.  If no audio payload is encountered the function
    returns without playing anything.

    Parameters
    ----------
    chunks: iterable
        The generator returned from ``tts_generate_streaming``.  Each element
        should be a ``ChatCompletionChunk`` with a ``choices`` list.  Each
        choice has a ``delta`` attribute; if the ``delta`` includes an
        ``audio`` attribute then the ``data`` attribute of ``audio`` contains
        the base64 encoded PCM payload to play.
    """
    p = pyaudio.PyAudio()
    audio_stream = None  # Will be initialised on first audio payload
    try:
        for chunk in chunks:
            # Each chunk contains one choice; obtain the delta for streaming
            delta = chunk.choices[0].delta
            # Some chunks may not contain audio data; skip those.  The OpenAI
            # client represents the audio payload as either a dictionary or
            # an object with a ``data`` attribute.  Handle both cases.
            audio_field = getattr(delta, "audio", None)
            if not audio_field:
                continue
            # Determine the base64 payload.  When using dataclasses this
            # attribute will be a plain dict with ``{"id": ..., "data": ...}``.
            if isinstance(audio_field, dict):
                b64_data = audio_field.get("data")
            else:
                b64_data = getattr(audio_field, "data", None)
            # If no payload is present skip this chunk
            if not b64_data:
                continue
            pcm_bytes = base64.b64decode(b64_data)
            if audio_stream is None:
                # The Higgs Audio model outputs 16‑bit mono PCM at 24 kHz.  If
                # these parameters ever change the API should communicate new
                # values via metadata in the first chunk.  For now we hardcode
                # typical values to ensure the stream is configured correctly.
                try:
                    # Try to get default output device
                    default_output = p.get_default_output_device_info()['index']
                except (OSError, IOError):
                    # If no default, try to find any output device
                    default_output = None
                    for i in range(p.get_device_count()):
                        info = p.get_device_info_by_index(i)
                        if info.get('maxOutputChannels') > 0:
                            default_output = i
                            break
                
                audio_stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=24_000,
                    output=True,
                    output_device_index=default_output,
                )
            # Write PCM data to the output device.  PyAudio handles buffer
            # sizes internally so we simply write the full chunk.
            audio_stream.write(pcm_bytes)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        p.terminate()


def play_wav_file(path: str, volume: float = 1.0) -> None:
    """Play a WAV file at a given volume."""
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return

    with wave.open(path, "rb") as wf:
        p = pyaudio.PyAudio()
        
        # Try to get default output device
        try:
            default_output = p.get_default_output_device_info()['index']
        except (OSError, IOError):
            # If no default, try to find any output device
            default_output = None
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get('maxOutputChannels') > 0:
                    default_output = i
                    break
        
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=default_output,
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
            prompts_module.TTS_SYSTEM_PROMPT
        )
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Module '{module_name}' is missing required attributes: {e}")


def load_voice_config(voice: str) -> tuple:
    """Load configuration for the specified voice.
    
    Args:
        voice: The voice name (keegan, stephen, or ricky)
        
    Returns:
        Tuple of (VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH)
    """
    if voice not in VOICE_CONFIG:
        raise ValueError(f"Unknown voice: {voice}. Valid voices: {list(VOICE_CONFIG.keys())}")
    
    config = VOICE_CONFIG[voice]
    return (config["reference_prompt"], config["reference_path"])


def main() -> None:
    """Entry point for running a translation and streaming TTS demo."""
    parser = argparse.ArgumentParser(description="Higgs Audio Emotion Demo")
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
        "--prompt",
        type=str,
        default="I joined a hackathon.",
        help="The user prompt to translate"
    )
    args = parser.parse_args()

    # Load mode configuration (for emotion translation)
    try:
        TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT = load_mode_config(args.mode)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"❌ Error loading mode configuration: {e}")
        return

    # Load voice configuration (for TTS voice)
    try:
        VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH = load_voice_config(args.voice)
    except ValueError as e:
        print(f"❌ Error loading voice configuration: {e}")
        return

    # Verify audio files exist
    if not os.path.exists(VOICE_REFERENCE_PATH):
        print(f"❌ Error: Voice reference file not found: {VOICE_REFERENCE_PATH}")
        return
    if not os.path.exists(LAUGH_TRACK_PATH):
        print(f"⚠️  Warning: Laugh track file not found: {LAUGH_TRACK_PATH}")

    client = get_client()
    
    print(f"🎭 Mode: {args.mode.capitalize()} ({MODE_CONFIG[args.mode]['description']})")
    print(f"🎤 Voice: {args.voice.capitalize()} ({VOICE_CONFIG[args.voice]['description']})")
    print(f"💬 User prompt: {args.prompt}")
    print()
    
    # Step 1: Translate emotion
    translated_text = translate_emotion(client, args.prompt, TRANSLATOR_SYSTEM_PROMPT)
    print(f"Rephrased text: {translated_text}\nGenerating speech...")
    
    # Step 2: Request a streaming TTS response
    stream_iter = tts_generate_streaming(client, translated_text,
                                         TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH)
    
    # Step 3: Play the audio as it arrives
    play_streaming_audio(stream_iter)
    
    # Step 4: Add laugh track
    play_wav_file(LAUGH_TRACK_PATH, volume=0.25)
    
    print("✅ Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
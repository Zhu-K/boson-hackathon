from __future__ import annotations

import base64
import io
import os
import re
import wave
from typing import Iterable, Iterator, Optional

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PATH, VOICE_REFERENCE_PROMPT


# Load environment variables from a .env file if present.
load_dotenv()


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


def translate_anger(client: OpenAI, user_prompt: str) -> str:
    """Translate a calm user prompt into an angry, comedic version."""
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=[
            {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    # Remove the `<think>...</think>` sections returned by Qwen models.
    # out = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    print(content)
    return content

def tts_generate_streaming(client: OpenAI, text: str) -> Iterator:
    """Create a streaming TTS completion from the Higgs audio model."""

    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": TTS_SYSTEM_PROMPT},
            {"role": "user", "content": VOICE_REFERENCE_PROMPT},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64(VOICE_REFERENCE_PATH), "format": "wav"},
                    }
                ],
            },
            {"role": "user", "content": f"[SPEAKER1] {text}"},
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
                # The Higgs Audio model outputs 16‑bit mono PCM at 24 kHz.  If
                # these parameters ever change the API should communicate new
                # values via metadata in the first chunk.  For now we hardcode
                # typical values to ensure the stream is configured correctly.
                audio_stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=24_000,
                    output=True,
                )
            # Write PCM data to the output device.  PyAudio handles buffer
            # sizes internally so we simply write the full chunk.
            audio_stream.write(pcm_bytes)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        p.terminate()


def main() -> None:
    """Entry point for running a translation and streaming TTS demo."""
    client = get_client()
    user_prompt = "I joined a hackathon."
    translated_text = translate_anger(client, user_prompt)
    # Request a streaming TTS response
    stream_iter = tts_generate_streaming(client, translated_text)
    # Play the audio as it arrives
    play_streaming_audio(stream_iter)


if __name__ == "__main__":  # pragma: no cover
    main()
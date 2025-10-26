from __future__ import annotations

import audioop
import base64
import io
import os
import re
import wave
from typing import Iterable, Iterator, Optional

import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PATH, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PROMPT2, VOICE_REFERENCE_PATH2
import numpy as np
from collections import deque


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
    """Translate the user's prompt via a chat completion.

    The BosonAI sandbox demonstration includes a translation model which
    expresses polite English user prompts as if spoken with anger.  This
    function invokes the translation system and strips the `<think>` tags
    inserted by Qwen models to reveal the final translated output.

    Parameters
    ----------
    client: OpenAI
        An authenticated OpenAI client.
    user_prompt: str
        The input text to translate.

    Returns
    -------
    str
        The translated string with any `<think>` segments removed.
    """
    response = client.chat.completions.create(
        # model="Qwen3-32B-thinking-Hackathon",
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
    # return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:

    # Wrap the raw PCM in a WAV container to include sample rate metadata.
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16‑bit samples
            wf.setframerate(16_000)  # must match record_microphone_segment()
            wf.writeframes(audio_bytes)
        wav_data = buffer.getvalue()

    # Base64‑encode the WAV for transmission.
    audio_b64 = base64.b64encode(wav_data).decode("utf-8")

    response = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe this audio for me."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=256,
        temperature=0.0,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def tts_generate_streaming(client: OpenAI, text: str, original_text: str) -> Iterator:
    """Create a streaming TTS completion from the Higgs audio model.

    This function sets ``stream=True`` on the API call to enable real‑time
    generation.  Instead of returning a single response object, the OpenAI
    library yields a generator of incremental ``ChatCompletionChunk`` objects.

    Parameters
    ----------
    client: OpenAI
        An authenticated OpenAI client.
    text: str
        The text which will be synthesised into speech.  A reference audio
        sample at ``reference_path`` will be used to clone the speaker's
        characteristics.

    Returns
    -------
    Iterator
        An iterator yielding streaming chunks from the API.  Each chunk
        contains optional audio data accessible via ``chunk.choices[0].delta.audio``.
    """
    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": TTS_SYSTEM_PROMPT},
            {"role": "user", "content": VOICE_REFERENCE_PROMPT2},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64(VOICE_REFERENCE_PATH2), "format": "wav"},
                    }
                ],
            },
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
            {"role": "user",
                "content": f"[SPEAKER2] {original_text}\n\n[SPEAKER1] {text}"},
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=True,  # enable streaming for real‑time audio playback
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )


def record_microphone_segment(
    threshold: float = 500.0,
    rate: int = 16_000,
    chunk_size: int = 1024,
    pre_seconds: float = 1.0,
    post_seconds: float = 1.0,
    silence_tolerance: float = 2.0,
    input_device_index: Optional[int] = None,
) -> bytes:
    """Record audio from the default microphone and return a cropped segment.

    This helper listens on the system's default input device and watches the
    incoming amplitude to detect when the user starts and stops speaking.
    A rolling buffer of the last ``pre_seconds`` of audio is maintained so
    that leading context before the detection point is included.  Recording
    continues until the level drops below ``threshold`` for more than
    ``silence_tolerance`` seconds, after which an additional ``post_seconds``
    of audio is captured to include trailing silence.

    Parameters
    ----------
    threshold : float
        RMS amplitude threshold above which audio is considered speech.
    rate : int
        Sample rate in Hz for recording (16 kHz by default).  The returned
        segment will use the same rate.
    chunk_size : int
        Number of samples to read per frame.  Smaller values improve
        responsiveness at the expense of overhead.
    pre_seconds : float
        Seconds of audio to retain prior to the start of speech.
    post_seconds : float
        Seconds of audio to capture after the end of speech.
    silence_tolerance : float
        Maximum duration of consecutive silence (below ``threshold``) within
        speech before terminating the recording.

    Returns
    -------
    bytes
        Raw 16‑bit PCM audio including 1s of context before and after the
        detected speech segment.
    """
    p = pyaudio.PyAudio()
    # Open an input stream on the default microphone
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index,
    )
    # Compute the number of frames that correspond to the pre/post buffers and silence tolerance
    pre_max_chunks = int(pre_seconds * rate / chunk_size)
    post_max_chunks = int(post_seconds * rate / chunk_size)
    silence_max_chunks = int(silence_tolerance * rate / chunk_size)
    pre_buffer: deque[bytes] = deque(maxlen=pre_max_chunks)
    frames: list[bytes] = []
    recording = False
    silent_chunks = 0
    try:
        while True:
            data = stream.read(chunk_size)
            # Maintain a rolling buffer for pre‑speech audio
            pre_buffer.append(data)
            # Compute RMS amplitude
            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))
            if recording:
                frames.append(data)
                # Check for silence
                if rms < threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0
                # If we've seen too much silence, capture trailing audio and stop
                if silent_chunks >= silence_max_chunks:
                    # Capture post_seconds of additional audio
                    for _ in range(post_max_chunks):
                        trailing = stream.read(chunk_size)
                        frames.append(trailing)
                    break
            else:
                # Wait for speech start
                if rms >= threshold:
                    recording = True
                    # Include the pre‑speech buffer in the output
                    frames.extend(list(pre_buffer))
                    frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    return b"".join(frames)


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
            audio_stream.write(pcm_bytes)
    finally:
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
        p.terminate()


def play_wav_file(path: str, volume: float = 1.0) -> None:
    """Play a WAV file located at `path` using PyAudio, with adjustable volume."""
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
            data_to_write = (
                audioop.mul(data, wf.getsampwidth(),
                            volume) if volume != 1.0 else data
            )
            stream.write(data_to_write)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()


def get_input_device_index() -> Optional[int]:
    p = pyaudio.PyAudio()
    default_device_index = p.get_default_input_device_info()['index']

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            if i == default_device_index:
                print(f"*** {i}: {info['name']} - (default)")
            else:
                print(f"{i}: {info['name']}")

    p.terminate()
    # get user input to select device index
    selected_index = input(
        f"Enter the index of the input audio device to use (Or press enter to use DEFAULT): ")
    if selected_index == "":
        return default_device_index
    return int(selected_index)


def main() -> None:
    """Entry point for running a translation and streaming TTS demo."""

    client = get_client()
    input_device_index = get_input_device_index()

    while True:

        # Capture an utterance from the microphone.
        print("Listening for speech...")
        audio_bytes = record_microphone_segment(
            threshold=1500.0, input_device_index=input_device_index)
        print("Captured audio segment. Translating...")

        # 1. transcribe the captured audio to text
        captured_speech = transcribe_audio(client, audio_bytes)

        if captured_speech.strip().lower in ["exit", "quit", "end program"]:
            break

        # 2. translate the text to an angry version
        angry_text = translate_anger(client, captured_speech)
        print(f"Translated text: {angry_text}\nGenerating speech...")

        # 3. Request a streaming TTS response
        stream_iter = tts_generate_streaming(
            client, angry_text, captured_speech)

        print("Streaming generated speech...")

        # 4. Play the audio as it arrives
        play_streaming_audio(stream_iter)

        # laugh track :D
        play_wav_file('./audios/sitcom_laugh_track.wav', volume=0.2)

        print("Playback completed.")


if __name__ == "__main__":
    main()

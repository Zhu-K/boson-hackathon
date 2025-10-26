import base64
import io
import os
import wave
import re
from typing import Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SPEAKER_TAG = "[SPEAKER1]"


def get_client() -> OpenAI:
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://hackathon.boson.ai/v1")


class VoiceReference:
    def __init__(self, audio: str | bytes, text: str) -> None:
        if isinstance(audio, str):
            self.voice_audio = _b64(audio)
        else:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16_000)
                    wf.writeframes(audio)
                wav_data = buffer.getvalue()
            self.voice_audio = base64.b64encode(wav_data).decode("utf-8")
        self.text = text


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def translate_emotion(client: OpenAI, user_prompt: str, translator_prompt: str) -> str:
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
    output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return output


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:
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
            {"role": "system", "content": "Transcribe this audio for me. Include only the spoken text, nothing else."},
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
    return response.choices[0].message.content


def tts_generate_streaming(client: OpenAI, text: str, tts_prompt: str, voice_ref: VoiceReference) -> Iterator:
    messages = [
        {"role": "system", "content": tts_prompt},
        {"role": "user", "content": voice_ref.text},
        {
            "role": "assistant",
            "content": [
                {"type": "input_audio", "input_audio": {"data": voice_ref.voice_audio, "format": "wav"}},
            ],
        },
        {"role": "user", "content": f"{SPEAKER_TAG} {text}"},
    ]
    return client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=messages,
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=True,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )


MODE_CONFIG = {
    "angry": {"module": "prompts", "description": "Angry"},
    "sarcastic": {"module": "prompts_sarcastic", "description": "Sarcastic"},
    "roast": {"module": "prompts_roast", "description": "Roast"},
    "conversation": {"module": "prompts_conversation", "description": "Conversational Comedy"},
}


def load_mode_config(mode: str) -> tuple:
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(MODE_CONFIG.keys())}")
    try:
        if mode.lower() == "angry":
            from app.prompts.prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE
        elif mode.lower() == "sarcastic":
            from app.prompts.prompts_sarcastic import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE
        elif mode.lower() == "roast":
            from app.prompts.prompts_roast import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE
        elif mode.lower() == "conversation":
            from app.prompts.prompts_conversation import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE
        return (TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE)
    except ImportError as e:
        raise ImportError(f"Failed to import prompts for mode '{mode}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Prompt module missing attributes: {e}")


VOICE_CONFIG = {
    "keegan": {
        "reference_path": "./audios/keegan.wav",
        "reference_prompt": f"{SPEAKER_TAG} Oh and CNN, thank you so much for the wall to wall ebola coverage. For two whole weeks, we were one step away from the walking dead!",
        "description": "Keegan-Michael Key (energetic, comedic)",
    },
    "stephen": {
        "reference_path": "./audios/stephen_colbert.wav",
        "reference_prompt": f"{SPEAKER_TAG} Oh sarcasm works great..sarcasm is absolutely the best thing for a president to do in the middle of a pandemic. You're doing amaaazing, 'Mr. President'.",
        "description": "Stephen Colbert (dry, witty)",
    },
    "ricky": {
        "reference_path": "./audios/ricky_gervaise.wav",
        "reference_prompt": f"{SPEAKER_TAG} All the best actors have jumped to Netflix and HBO, you know, and the actors who just do hollywood movies now do fantasy adventure nonsense..they wear masks and capes and reaaally tight costumes. Their job isn't acting anymore! It's going to the gym twice a day and taking steroids really.",
        "description": "Ricky Gervais (brutally honest)",
    },
    "obama": {
        "reference_path": "./audios/obama.wav",
        "reference_prompt": "{SPEAKER_TAG} Anyway as always I want to close on a more serious note. You know I often joke about tensions between me and the press, but honestly what they say doesn't bother me. I understand we've got an adversarial system. I'm a mellow sort of guy.",
        "description": "Barack Obama (calm, measured)",
    },
    "my_voice": {
        "reference_path": None,
        "reference_prompt": None,
        "description": "Your own voice (cloned from recording)",
    },
}

LAUGH_TRACK_PATH = "./audios/sitcom_laugh_track.wav"


def load_voice_config(voice: str, recorded_audio: bytes = None, transcription: str = None) -> VoiceReference:
    if voice not in VOICE_CONFIG:
        raise ValueError(f"Unknown voice: {voice}. Valid voices: {list(VOICE_CONFIG.keys())}")
    config = VOICE_CONFIG[voice]
    if voice == "my_voice":
        if recorded_audio is None or transcription is None:
            raise ValueError("Voice cloning requires recorded_audio and transcription")
        reference_prompt = f"{SPEAKER_TAG} {transcription}"
        return VoiceReference(recorded_audio, reference_prompt)
    return VoiceReference(config["reference_path"], config["reference_prompt"])


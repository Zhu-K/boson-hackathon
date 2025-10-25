from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PATH, VOICE_REFERENCE_PROMPT
import base64
import os
import re
import io
import wave
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")


def get_client():
    BOSON_API_KEY = os.getenv("BOSON_API_KEY")
    client = OpenAI(api_key=BOSON_API_KEY,
                    base_url="https://hackathon.boson.ai/v1")
    return client


def translate_anger(client, user_prompt):
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",  # replace it
        messages=[
            {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=4096,
        temperature=0.7
    )

    content = response.choices[0].message.content
    # output = re.sub(r"<think>.*?</think>", "",
    #                 content, flags=re.DOTALL).strip()
    print(content)
    return content


def tts_generate(client, text):
    response = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": TTS_SYSTEM_PROMPT},
            {"role": "user", "content": VOICE_REFERENCE_PROMPT},
            {
                "role": "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": b64(VOICE_REFERENCE_PATH), "format": "wav"}
                }],
            },
            {"role": "user", "content": f"[SPEAKER1] {text}"},
        ],
        modalities=["text", "audio"],
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        stream=False,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        extra_body={"top_k": 50},
    )

    audio_b64 = response.choices[0].message.audio.data
    open("output.wav", "wb").write(base64.b64decode(audio_b64))
    return audio_b64
    


def play_audio(audio_b64):
    audio_bytes = base64.b64decode(audio_b64)

    # open in-memory WAV
    wf = wave.open(io.BytesIO(audio_bytes), "rb")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )

    # play
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    client = get_client()
    user_prompt = "I joined a hackathon."
    translated_text = translate_anger(client, user_prompt)
    audio_b64 = tts_generate(client, translated_text)
    play_audio(audio_b64)

if __name__ == "__main__":
    main()
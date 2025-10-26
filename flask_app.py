from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import io
import os
import wave
import time
from typing import Iterator
import numpy as np
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI
import threading

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'higgs-audio-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global client
client = None


def inject_language_instruction(prompt: str, language: str) -> str:
    """Inject language instruction after the first sentence of the prompt.
    
    Args:
        prompt: The original system prompt
        language: Target language (english or mandarin)
    
    Returns:
        Modified prompt with language instruction
    """
    if language.lower() == "mandarin":
        # Find the first sentence (ends with period, newline, or em dash)
        lines = prompt.strip().split('\n')
        if lines:
            first_line = lines[0]
            language_instruction = "You must translate and output all your output in Mandarin. Never use any other language."
            # Insert after first line
            modified_lines = [first_line, language_instruction] + lines[1:]
            return '\n'.join(modified_lines)
    return prompt


def get_client() -> OpenAI:
    """Initialize OpenAI client with Boson API."""
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://hackathon.boson.ai/v1")


def b64(path: str) -> str:
    """Encode file to base64."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


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


def record_microphone_segment(duration: float = 5.0, rate: int = 16_000, chunk_size: int = 1024) -> bytes:
    """Record audio from the microphone for a fixed duration."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        input_device_index=0,
        frames_per_buffer=chunk_size,
    )

    frames: list[bytes] = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            data = stream.read(chunk_size)
            frames.append(data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return b"".join(frames)


def load_wav_file_as_pcm(path: str, target_rate: int = 24_000, volume: float = 0.25) -> tuple[bytes, int]:
    """Load a WAV file and convert it to PCM format with volume adjustment.
    Returns (pcm_bytes, sample_rate)."""
    if not os.path.exists(path):
        return b"", target_rate
    
    with wave.open(path, "rb") as wf:
        # Get the actual sample rate from the file
        sample_rate = wf.getframerate()
        
        # Read all frames
        frames = wf.readframes(wf.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
        
        # Apply volume scaling
        scaled = np.clip(audio_array * volume, -32768, 32767).astype(np.int16)
        
        return scaled.tobytes(), sample_rate


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@socketio.on('start_recording')
def handle_recording(data):
    """Handle audio recording and processing."""
    try:
        mode = data.get('mode', 'angry')
        voice = data.get('voice', 'keegan')
        duration = float(data.get('duration', 5.0))
        language = data.get('language', 'english')
        
        # Load appropriate prompts for emotion mode
        if mode == "angry":
            from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT
        elif mode == "sarcastic":
            from prompts_sarcastic import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT
        else:  # roast mode
            from prompts_roast import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT
        
        # Inject language instruction if Mandarin is selected
        TRANSLATOR_SYSTEM_PROMPT = inject_language_instruction(TRANSLATOR_SYSTEM_PROMPT, language)
        
        # Step 1: Record
        emit('status', {'step': 'recording', 'message': f'Recording for {duration} seconds...'})
        audio_bytes = record_microphone_segment(duration=duration)
        emit('status', {'step': 'recording_complete', 'message': 'Recording complete!'})
        
        # Step 2: Transcribe
        emit('status', {'step': 'transcribing', 'message': 'Transcribing audio...'})
        captured_speech = transcribe_audio(client, audio_bytes)
        emit('transcription', {'text': captured_speech})
        
        # Get voice configuration
        voice_config = VOICE_CONFIG.get(voice, VOICE_CONFIG["keegan"])
        
        # Create voice reference
        if voice == "my_voice":
            # Voice cloning from recorded audio
            reference_prompt = f"{SPEAKER_TAG} {captured_speech}"
            voice_reference = VoiceReference(audio_bytes, reference_prompt)
        else:
            # Pre-recorded voice from file
            voice_reference = VoiceReference(voice_config["reference_path"], voice_config["reference_prompt"])
        
        # Laugh track path
        laugh_track = "./audios/sitcom_laugh_track.wav"

        # Initialize client
        global client
        if client is None:
            client = get_client()

        # Step 3: Translate
        emit('status', {'step': 'translating', 'message': f'Translating to {mode} mode...'})
        emotional_text = translate_emotion(client, captured_speech, TRANSLATOR_SYSTEM_PROMPT)
        emit('translation', {'text': emotional_text})
        emit('status', {'step': 'translation_complete', 'message': 'Translation complete!'})

        # Step 4: TTS - Stream audio chunks in real-time
        emit('status', {'step': 'generating_audio', 'message': 'Generating emotional speech...'})
        stream_iter = tts_generate_streaming(
            client, emotional_text,
            TTS_SYSTEM_PROMPT, voice_reference
        )
        
        # Stream audio chunks to client
        for chunk in stream_iter:
            delta = chunk.choices[0].delta
            audio_field = getattr(delta, "audio", None)
            if not audio_field:
                continue
            b64_data = audio_field.get("data") if isinstance(audio_field, dict) else getattr(audio_field, "data", None)
            if not b64_data:
                continue
            
            # Send audio chunk to client
            emit('audio_chunk', {'data': b64_data})
        
        emit('audio_complete', {})
        
        # Step 5: Send laugh track
        emit('status', {'step': 'laugh_track', 'message': 'Adding laugh track...'})
        laugh_pcm, laugh_rate = load_wav_file_as_pcm(laugh_track, volume=0.25)
        if laugh_pcm:
            # Send laugh track as base64 with sample rate
            laugh_b64 = base64.b64encode(laugh_pcm).decode('utf-8')
            emit('laugh_track', {'data': laugh_b64, 'sample_rate': laugh_rate})
        
        emit('status', {'step': 'complete', 'message': 'All done!'})

    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)

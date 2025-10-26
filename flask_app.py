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

# Import roast conversation functions
from roast_conversation import (
    save_audio_to_history,
    add_to_conversation_history,
    translate_emotion_with_history,
    clear_conversation_history,
    save_combined_conversation_audio,
    increment_recording_counter
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'higgs-audio-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global client
client = None


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


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history for roast mode."""
    clear_conversation_history()
    return jsonify({'status': 'success', 'message': 'Conversation history cleared'})


@app.route('/history_status', methods=['GET'])
def history_status():
    """Get conversation history status."""
    from roast_conversation import get_conversation_history_length
    history_length = get_conversation_history_length()
    return jsonify({
        'has_history': history_length > 0,
        'history_length': history_length
    })


@socketio.on('start_recording')
def handle_recording(data):
    """Handle audio recording and processing."""
    try:
        mode = data.get('mode', 'angry')
        duration = float(data.get('duration', 5.0))
        
        # Increment recording counter for roast mode
        if mode == "roast":
            increment_recording_counter()
        
        # Load appropriate prompts
        if mode == "angry":
            from prompts import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
            laugh_track = "./audios/sitcom_laugh_track.wav"
        elif mode == "sarcastic":
            from prompts_sarcastic import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
            laugh_track = "./audios/sitcom_laugh_track.wav"
        elif mode == "roast":
            from prompts_heckle import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
            laugh_track = "./audios/sitcom_laugh_track.wav"
        else:
            from prompts_sarcastic import TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
            laugh_track = "./audios/sitcom_laugh_track.wav"

        # Initialize client
        global client
        if client is None:
            client = get_client()

        # Step 1: Record
        emit('status', {'step': 'recording', 'message': f'Recording for {duration} seconds...'})
        audio_bytes = record_microphone_segment(duration=duration)
        emit('status', {'step': 'recording_complete', 'message': 'Recording complete!'})

        # Step 2: Transcribe
        emit('status', {'step': 'transcribing', 'message': 'Transcribing your speech...'})
        captured_speech = transcribe_audio(client, audio_bytes)
        emit('transcription', {'text': captured_speech})
        emit('status', {'step': 'transcription_complete', 'message': 'Transcription complete!'})

        # Step 3: Translate
        emit('status', {'step': 'translating', 'message': f'Translating to {mode} mode...'})
        
        # Use conversation-aware translation for roast mode
        if mode == "roast":
            emotional_text = translate_emotion_with_history(client, captured_speech, TRANSLATOR_SYSTEM_PROMPT)
        else:
            emotional_text = translate_emotion(client, captured_speech, TRANSLATOR_SYSTEM_PROMPT)
            
        emit('translation', {'text': emotional_text})
        emit('status', {'step': 'translation_complete', 'message': 'Translation complete!'})

        # Step 4: TTS - Stream audio chunks in real-time
        emit('status', {'step': 'generating_audio', 'message': 'Generating emotional speech...'})
        
        # Use TTS generation (same for all modes, no conversation history needed for TTS)
        stream_iter = tts_generate_streaming(
            client, emotional_text,
            TTS_SYSTEM_PROMPT, VOICE_REFERENCE_PROMPT, VOICE_REFERENCE_PATH
        )
        
        # Collect audio chunks for saving (roast mode only)
        assistant_audio_chunks = []
        
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
            
            # Collect chunks for roast mode history
            if mode == "roast":
                assistant_audio_chunks.append(base64.b64decode(b64_data))
        
        emit('audio_complete', {})
        
        # Save conversation to history for roast mode
        if mode == "roast":
            try:
                # First add user input to history
                user_audio_path = save_audio_to_history(audio_bytes, captured_speech, is_user=True)
                add_to_conversation_history(captured_speech, user_audio_path, is_user=True)
                
                # Then add assistant response to history
                if assistant_audio_chunks:
                    assistant_pcm = b"".join(assistant_audio_chunks)
                    assistant_audio_path = save_audio_to_history(assistant_pcm, emotional_text, is_user=False)
                    add_to_conversation_history(emotional_text, assistant_audio_path, is_user=False)
            except Exception as e:
                print(f"ERROR: Failed to save conversation to history: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 5: Send laugh track
        emit('status', {'step': 'laugh_track', 'message': 'Adding laugh track...'})
        laugh_pcm, laugh_rate = load_wav_file_as_pcm(laugh_track, volume=0.25)
        if laugh_pcm:
            # Send laugh track as base64 with sample rate
            laugh_b64 = base64.b64encode(laugh_pcm).decode('utf-8')
            emit('laugh_track', {'data': laugh_b64, 'sample_rate': laugh_rate})
        
        # Step 6: Save combined audio for roast mode
        if mode == "roast":
            try:
                emit('status', {'step': 'saving_combined', 'message': 'Saving combined audio...'})
                assistant_pcm = b"".join(assistant_audio_chunks) if assistant_audio_chunks else b""
                combined_path = save_combined_conversation_audio(
                    audio_bytes, assistant_pcm, laugh_pcm
                )
                print(f"Combined audio saved to: {combined_path}")
            except Exception as e:
                print(f"ERROR: Failed to save combined audio: {e}")
                import traceback
                traceback.print_exc()
        
        emit('status', {'step': 'complete', 'message': 'All done!'})

    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
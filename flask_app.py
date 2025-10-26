from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import os
import wave
from dotenv import load_dotenv
from openai import OpenAI

# Import roast conversation functions
from app.domain.conversation import (
    add_to_conversation_history,
    translate_emotion_with_history,
    clear_conversation_history,
    increment_recording_counter,
    get_conversation_history_length,
)

# Import functions from streaming script
from app.services.openai_service import (
    get_client,
    translate_emotion,
    transcribe_audio,
    tts_generate_streaming as streaming_tts_generate,
    load_mode_config,
    load_voice_config,
    VOICE_CONFIG,
    LAUGH_TRACK_PATH,
)

from app.services.audio import (
    record_microphone_segment,
    load_wav_file_as_pcm,
    list_audio_devices as list_audio_devices_service,
)

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.config['SECRET_KEY'] = 'higgs-audio-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global client
client = None


# Removed unused local helpers (b64 and TTS wrapper)

        
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
    history_length = get_conversation_history_length()
    return jsonify({
        'has_history': history_length > 0,
        'history_length': history_length
    })


@app.route('/devices', methods=['GET'])
def list_audio_devices():
    """Return available input audio devices and default index via service."""
    try:
        return jsonify(list_audio_devices_service())
    except Exception as e:
        return jsonify({'devices': [], 'default_index': None, 'error': str(e)})


@socketio.on('start_recording')
def handle_recording(data):
    """Handle audio recording and processing."""
    try:
        mode = data.get('mode', 'angry')
        language = data.get('language', 'english')  # NEW: Language support
        voice = data.get('voice', 'keegan')  # NEW: Voice selection support
        laugh_track_enabled = data.get('laughTrack', True)  # NEW: Laugh track toggle
        
        # Voice activity detection parameters (can be customized)
        threshold = float(data.get('threshold', 500.0))
        silence_tolerance = float(data.get('silenceTolerance', 2.0))
        max_duration = float(data.get('maxDuration', 30.0))
        
        # Load mode configuration (for emotion translation) - NEW: Using streaming script logic
        try:
            TRANSLATOR_SYSTEM_PROMPT, TTS_SYSTEM_PROMPT, LANGUAGE_TEMPLATE = load_mode_config(mode)
            if language != "english":
                language_instruction = LANGUAGE_TEMPLATE.format(language=language)
            else:
                language_instruction = ""
                
            llm_system_prompt = TRANSLATOR_SYSTEM_PROMPT.format(
                language_instruction=language_instruction, 
                language_instruction_repeated=language_instruction
            )
        except (ValueError, ImportError, AttributeError) as e:
            emit('error', {'message': f"Error loading mode configuration: {e}"})
            return

        # Initialize client
        global client
        if client is None:
            client = get_client()

        # Handle conversation mode (multi-turn) history
        is_conversation_mode = mode == "conversation"
        if is_conversation_mode:
            increment_recording_counter()
            history_length = get_conversation_history_length()
            print(f"💬 Found {history_length} previous messages in conversation history")

        # Step 1: Record with voice activity detection
        input_device_index = data.get('inputDeviceIndex', 0)
        audio_bytes = record_microphone_segment(
            threshold=threshold,
            silence_tolerance=silence_tolerance,
            max_duration=max_duration,
            emit_callback=emit,
            input_device_index=int(input_device_index) if isinstance(input_device_index, (int, str)) and str(input_device_index).isdigit() else 0
        )
        emit('status', {'step': 'recording_complete', 'message': 'Recording complete!'})

        # Step 2: Transcribe
        emit('status', {'step': 'transcribing', 'message': 'Transcribing your speech...'})
        captured_speech = transcribe_audio(client, audio_bytes)
        emit('transcription', {'text': captured_speech})
        emit('status', {'step': 'transcription_complete', 'message': 'Transcription complete!'})

        # Load voice configuration - NEW: From streaming script
        try:
            voice_reference = load_voice_config(
                voice, 
                recorded_audio=audio_bytes if voice == "my_voice" else None,
                transcription=captured_speech if voice == "my_voice" else None
            )
        except ValueError as e:
            emit('error', {'message': f"Error loading voice configuration: {e}"})
            return
        except FileNotFoundError as e:
            emit('error', {'message': f"Error: {e}"})
            return

        # Verify laugh track exists - NEW: From streaming script
        if not os.path.exists(LAUGH_TRACK_PATH):
            print(f"⚠️  Warning: Laugh track file not found: {LAUGH_TRACK_PATH}")

        # Step 3: Translate
        emit('status', {'step': 'translating', 'message': f'Translating to {mode} mode...'})
        
        # Use conversation-aware translation for conversation mode
        if is_conversation_mode:
            emotional_text = translate_emotion_with_history(client, captured_speech, llm_system_prompt)
        else:
            emotional_text = translate_emotion(client, captured_speech, llm_system_prompt)
            
        emit('translation', {'text': emotional_text})
        emit('status', {'step': 'translation_complete', 'message': 'Translation complete!'})

        # Step 4: Generate and stream audio in real time
        emit('status', {'step': 'generating_audio', 'message': '🎶 Cooking up the audio performance…'})
        
        # NEW: Enhanced TTS logic from streaming script
        
        if is_conversation_mode:
            print("Using conversational comedy mode (multi-turn)")
            # Handle voice cloning for my_voice - NEW: From streaming script
            ref_path = VOICE_CONFIG[voice]["reference_path"]
            if ref_path is None:  # Handle my_voice case
                temp_ref_path = "temp_voice_reference.wav"
                with wave.open(temp_ref_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16_000)
                    wf.writeframes(audio_bytes)
                ref_path = temp_ref_path
            
            # Use streaming TTS with prompt saving
            stream_iter = streaming_tts_generate(
                client, emotional_text, TTS_SYSTEM_PROMPT, 
                voice_reference
            )
            
            # Clean up temporary file if created
            temp_ref_created = ref_path == "temp_voice_reference.wav"
        else:
            # Use regular TTS
            stream_iter = streaming_tts_generate(
                client, emotional_text, TTS_SYSTEM_PROMPT, voice_reference
            )
            temp_ref_created = False
        
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
            
            # No server-side audio saving required
        
        # Clean up temporary voice reference file
        if temp_ref_created and os.path.exists("temp_voice_reference.wav"):
            os.remove("temp_voice_reference.wav")
        
        emit('audio_complete', {})
        
        # Save conversation to text-only history (no audio files)
        if is_conversation_mode:
            try:
                add_to_conversation_history(captured_speech, is_user=True)
                add_to_conversation_history(emotional_text, is_user=False)
            except Exception as e:
                print(f"ERROR: Failed to update conversation history: {e}")
        
        # Step 5: Send laugh track (only if enabled)
        if laugh_track_enabled:
            emit('status', {'step': 'laugh_track', 'message': 'Adding laugh track...'})
            laugh_pcm, laugh_rate = load_wav_file_as_pcm(LAUGH_TRACK_PATH, volume=0.25)  # NEW: Use LAUGH_TRACK_PATH
            if laugh_pcm:
                # Send laugh track as base64 with sample rate
                laugh_b64 = base64.b64encode(laugh_pcm).decode('utf-8')
                emit('laugh_track', {'data': laugh_b64, 'sample_rate': laugh_rate})
        
        # No server-side combined audio saving
        
        emit('status', {'step': 'complete', 'message': 'All done!'})

    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)

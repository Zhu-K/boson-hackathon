"""
Roast mode conversation history management.
Handles conversation history, audio saving, and history-aware TTS generation.
"""

import base64
import io
import os
import wave
import time
import json
import numpy as np
from typing import Iterator, List, Dict
from collections import deque
from openai import OpenAI

# Global conversation history and counters
conversation_history = deque(maxlen=4)  # Store last 2 exchanges (4 messages: user, assistant, user, assistant)
recording_counter = 0  # Counter for recording sessions


def b64(path: str) -> str:
    """Encode file to base64."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24_000) -> bytes:
    """Convert raw PCM audio to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def save_audio_to_history(pcm_data: bytes, text: str, is_user: bool = False, sample_rate: int = None) -> str:
    """Save audio to conversation history and return the file path."""
    # Create history directory if it doesn't exist
    history_dir = "conversation_history"
    os.makedirs(history_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    role = "user" if is_user else "speaker1"
    filename = f"{timestamp}_{role}.wav"
    filepath = os.path.join(history_dir, filename)
    
    # Convert PCM to WAV and save
    # Use 16kHz for user audio (microphone), 24kHz for assistant audio (API output)
    if sample_rate is None:
        sample_rate = 16_000 if is_user else 24_000
    
    wav_data = pcm_to_wav(pcm_data, sample_rate)
    with open(filepath, 'wb') as f:
        f.write(wav_data)
    
    return filepath


def save_prompt_context(messages: List[Dict], text: str, model_type: str = "tts") -> str:
    """Save the prompt context to a file for inspection."""
    global recording_counter
    
    # Create separate directories for different model types
    if model_type == "qwen":
        prompts_dir = "qwen_prompts"
    elif model_type == "tts":
        prompts_dir = "tts_prompts"
    else:
        prompts_dir = "roast_prompts"  # fallback
    
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Generate filename with recording counter
    filename = f"prepared_prompt_{recording_counter}.json"
    filepath = os.path.join(prompts_dir, filename)
    
    # Prepare data for saving (remove binary audio data for readability)
    prompt_data = {
        "recording_number": recording_counter,
        "model_type": model_type,
        "current_text": text,
        "conversation_history_length": len(conversation_history),
        "messages": []
    }
    
    for i, msg in enumerate(messages):
        message_copy = {
            "index": i,
            "role": msg["role"],
            "content": msg["content"]
        }
        
        # If content contains audio data, replace with placeholder
        if isinstance(msg["content"], list):
            content_items = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "input_audio":
                    content_items.append({
                        "type": "input_audio",
                        "input_audio": {
                            "format": item["input_audio"]["format"],
                            "data": f"<AUDIO_DATA_{len(item['input_audio']['data'][:50])}...>"
                        }
                    })
                else:
                    content_items.append(item)
            message_copy["content"] = content_items
        
        prompt_data["messages"].append(message_copy)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def add_to_conversation_history(text: str, audio_path: str = None, is_user: bool = False):
    """Add a message to the conversation history."""
    global conversation_history
    
    role = "user" if is_user else "assistant"
    message = {
        "role": role,
        "content": text,
        "audio_path": audio_path
    }
    
    conversation_history.append(message)


def build_qwen_messages_with_history(user_text: str, translator_prompt: str) -> List[Dict]:
    """Build messages array for Qwen model including conversation history."""
    messages = [
        {"role": "system", "content": translator_prompt}
    ]
    
    # Add conversation history (text only for context)
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg['content']})
        else:  # assistant
            messages.append({"role": "assistant", "content": msg['content']})
    
    # Add current user input
    messages.append({"role": "user", "content": user_text})
    
    return messages


def build_tts_messages(text: str, tts_prompt: str, voice_prompt: str, ref_path: str) -> List[Dict]:
    """Build messages array for TTS model (no conversation history, just voice conditioning)."""
    return [
        {"role": "system", "content": tts_prompt},
        {"role": "user", "content": voice_prompt},
        {
            "role": "assistant",
            "content": [
                {"type": "input_audio", "input_audio": {"data": b64(ref_path), "format": "wav"}},
            ],
        },
        {"role": "user", "content": f"[SPEAKER1] {text}"}
    ]


def tts_generate_streaming(client: OpenAI, text: str, tts_prompt: str, voice_prompt: str, ref_path: str) -> Iterator:
    """Generate TTS for roast mode (no conversation history, just voice conditioning)."""
    messages = build_tts_messages(text, tts_prompt, voice_prompt, ref_path)
    
    # Save the prompt context for inspection
    save_prompt_context(messages, text, "tts")
    
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


def translate_emotion_with_history(client: OpenAI, user_text: str, translator_prompt: str) -> str:
    """Translate user input to emotional response using conversation history."""
    messages = build_qwen_messages_with_history(user_text, translator_prompt)
    
    # Save the prompt context for inspection
    save_prompt_context(messages, user_text, "qwen")
    
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    return response.choices[0].message.content


def clear_conversation_history():
    """Clear the conversation history."""
    global conversation_history, recording_counter
    conversation_history.clear()
    recording_counter = 0  # Reset counter when history is cleared


def get_conversation_history_length() -> int:
    """Get the current length of conversation history."""
    return len(conversation_history)


def get_conversation_history() -> List[Dict]:
    """Get a copy of the current conversation history."""
    return list(conversation_history)


def combine_audio_files(user_pcm: bytes, assistant_pcm: bytes, laugh_pcm: bytes = None, 
                       user_rate: int = 16000, assistant_rate: int = 24000, laugh_rate: int = 24000) -> str:
    """Combine user, assistant, and laugh track audio into a single WAV file."""
    
    # Create combined audio directory
    combined_dir = "combined_audio"
    os.makedirs(combined_dir, exist_ok=True)
    
    # Target sample rate (use highest quality)
    target_rate = 24000
    
    # Convert all audio to same sample rate and format
    def resample_audio(pcm_data: bytes, from_rate: int, to_rate: int) -> np.ndarray:
        if not pcm_data:
            return np.array([], dtype=np.float32)
            
        # Convert PCM bytes to int16 array
        int16_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Convert to float32 (-1.0 to 1.0)
        float_array = int16_array.astype(np.float32) / 32768.0
        
        # Simple resampling (for production, use scipy.signal.resample)
        if from_rate != to_rate:
            # Calculate new length
            new_length = int(len(float_array) * to_rate / from_rate)
            # Simple linear interpolation resampling
            indices = np.linspace(0, len(float_array) - 1, new_length)
            float_array = np.interp(indices, np.arange(len(float_array)), float_array)
        
        return float_array
    
    # Resample all audio to target rate
    user_audio = resample_audio(user_pcm, user_rate, target_rate)
    assistant_audio = resample_audio(assistant_pcm, assistant_rate, target_rate)
    laugh_audio = resample_audio(laugh_pcm or b'', laugh_rate, target_rate)
    
    # Add small silence gaps between segments (0.5 seconds)
    silence_samples = int(target_rate * 0.5)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    # Combine audio: user -> silence -> assistant -> silence -> laugh
    combined_audio = np.concatenate([
        user_audio,
        silence,
        assistant_audio,
        silence if len(laugh_audio) > 0 else np.array([]),
        laugh_audio
    ])
    
    # Convert back to int16
    combined_int16 = (combined_audio * 32767).astype(np.int16)
    
    # Save as WAV file
    global recording_counter
    filename = f"roast_conversation_{recording_counter}.wav"
    filepath = os.path.join(combined_dir, filename)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(target_rate)
        wav_file.writeframes(combined_int16.tobytes())
    
    return filepath


def increment_recording_counter():
    """Increment the recording counter for new recording sessions."""
    global recording_counter
    recording_counter += 1


def get_recording_counter() -> int:
    """Get the current recording counter."""
    return recording_counter


def save_combined_conversation_audio(user_pcm: bytes, assistant_pcm: bytes, laugh_pcm: bytes = None) -> str:
    """Save a combined audio file of the entire conversation exchange."""
    return combine_audio_files(user_pcm, assistant_pcm, laugh_pcm)

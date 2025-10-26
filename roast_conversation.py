"""
Roast mode conversation management.
Handles conversation history, audio processing, and model prompt generation.
"""

import base64
import io
import json
import os
import time
import wave
from collections import deque
from typing import Dict, Iterator, List
import re 

import numpy as np
from openai import OpenAI

# Global state
conversation_history = deque(maxlen=4)  # Last 2 exchanges (user + assistant pairs)
recording_counter = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _b64_encode_file(path: str) -> str:
    """Encode file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24_000) -> bytes:
    """Convert raw PCM audio to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def _resample_audio(pcm_data: bytes, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample PCM audio to target sample rate."""
    if not pcm_data:
        return np.array([], dtype=np.float32)
    
    # Convert PCM to float32
    int16_array = np.frombuffer(pcm_data, dtype=np.int16)
    float_array = int16_array.astype(np.float32) / 32768.0
    
    # Simple linear interpolation resampling
    if from_rate != to_rate:
        new_length = int(len(float_array) * to_rate / from_rate)
        indices = np.linspace(0, len(float_array) - 1, new_length)
        float_array = np.interp(indices, np.arange(len(float_array)), float_array)
    
    return float_array


# =============================================================================
# CONVERSATION HISTORY MANAGEMENT
# =============================================================================

def add_to_conversation_history(text: str, audio_path: str = None, is_user: bool = False):
    """Add a message to the conversation history."""
    conversation_history.append({
        "role": "user" if is_user else "assistant",
        "content": text,
        "audio_path": audio_path
    })


def clear_conversation_history():
    """Clear conversation history and reset counter."""
    global conversation_history, recording_counter
    conversation_history.clear()
    recording_counter = 0


def get_conversation_history_length() -> int:
    """Get current conversation history length."""
    return len(conversation_history)


def increment_recording_counter():
    """Increment recording counter for new sessions."""
    global recording_counter
    recording_counter += 1


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def save_audio_to_history(pcm_data: bytes, text: str, is_user: bool = False) -> str:
    """Save audio to conversation history directory."""
    os.makedirs("conversation_history", exist_ok=True)
    
    # Generate timestamped filename
    timestamp = int(time.time() * 1000)
    role = "user" if is_user else "speaker1"
    filepath = f"conversation_history/{timestamp}_{role}.wav"
    
    # Convert and save audio
    sample_rate = 16_000 if is_user else 24_000
    wav_data = _pcm_to_wav(pcm_data, sample_rate)
    
    with open(filepath, 'wb') as f:
        f.write(wav_data)
    
    return filepath


def save_combined_conversation_audio(user_pcm: bytes, assistant_pcm: bytes, laugh_pcm: bytes = None) -> str:
    """Combine and save complete conversation audio."""
    os.makedirs("combined_audio", exist_ok=True)
    
    # Resample all audio to 24kHz
    user_audio = _resample_audio(user_pcm, 16000, 24000)
    assistant_audio = _resample_audio(assistant_pcm, 24000, 24000)
    laugh_audio = _resample_audio(laugh_pcm or b'', 24000, 24000)
    
    # Add 0.5s silence between segments
    silence = np.zeros(int(24000 * 0.5), dtype=np.float32)
    
    # Combine: user -> silence -> assistant -> silence -> laugh
    combined_audio = np.concatenate([
        user_audio,
        silence,
        assistant_audio,
        silence if len(laugh_audio) > 0 else np.array([]),
        laugh_audio
    ])
    
    # Convert to int16 and save
    combined_int16 = (combined_audio * 32767).astype(np.int16)
    filepath = f"combined_audio/roast_conversation_{recording_counter}.wav"
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(combined_int16.tobytes())
    
    return filepath


# =============================================================================
# PROMPT GENERATION AND SAVING
# =============================================================================

def _save_prompt_context(messages: List[Dict], text: str, model_type: str):
    """Save prompt context to appropriate directory for inspection."""
    # Create model-specific directory
    prompts_dir = f"{model_type}_prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Prepare data for saving
    prompt_data = {
        "recording_number": recording_counter,
        "model_type": model_type,
        "current_text": text,
        "conversation_history_length": len(conversation_history),
        "messages": []
    }
    
    # Process messages (replace audio data with placeholders)
    for i, msg in enumerate(messages):
        message_copy = {
            "index": i,
            "role": msg["role"],
            "content": msg["content"]
        }
        
        # Replace binary audio data with readable placeholder
        if isinstance(msg["content"], list):
            content_items = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "input_audio":
                    content_items.append({
                        "type": "input_audio",
                        "input_audio": {
                            "format": item["input_audio"]["format"],
                            "data": f"<AUDIO_DATA_{len(str(item['input_audio']['data'])[:50])}...>"
                        }
                    })
                else:
                    content_items.append(item)
            message_copy["content"] = content_items
        
        prompt_data["messages"].append(message_copy)
    
    # Save to JSON
    filepath = f"{prompts_dir}/prepared_prompt_{recording_counter}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)


def build_qwen_messages(user_text: str, system_prompt: str) -> List[Dict]:
    """Build message array for Qwen model with conversation history."""
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (text only)
    for msg in conversation_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current user input
    messages.append({"role": "user", "content": user_text})
    return messages




# =============================================================================
# MODEL INTERACTION
# =============================================================================

def translate_emotion_with_history(client: OpenAI, user_text: str, system_prompt: str) -> str:
    """Generate contextual roast response using Qwen model."""
    messages = build_qwen_messages(user_text, system_prompt)
    _save_prompt_context(messages, user_text, "qwen")
    
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return output


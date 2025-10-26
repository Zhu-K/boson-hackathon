"""
Roast mode conversation management.
Handles conversation history, audio processing, and model prompt generation.
"""

import json
import os
from collections import deque
from typing import Dict, List
import re
from openai import OpenAI

# Global state
conversation_history = deque(maxlen=4)  # Last 2 exchanges (user + assistant pairs)
recording_counter = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Removed unused _b64_encode_file helper


# Removed WAV/PCM utilities; no longer saving audio files server-side.


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

# Removed save_audio_to_history; UI handles saving locally.


# Removed save_combined_conversation_audio; not needed for UI.


# =============================================================================
# PROMPT GENERATION AND SAVING
# =============================================================================

# Removed _save_prompt_context (prompt logging) — not required for runtime


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
    
    response = client.chat.completions.create(
        model="Qwen3-32B-non-thinking-Hackathon",
        messages=messages,
        max_tokens=4096,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    output = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return output


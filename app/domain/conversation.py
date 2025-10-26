from collections import deque
from typing import Dict, List
import re
from openai import OpenAI

# Keep a small rolling history: last 2 exchanges (user+assistant pairs)
conversation_history = deque(maxlen=4)
recording_counter = 0


def add_to_conversation_history(text: str, is_user: bool = False) -> None:
    conversation_history.append({
        "role": "user" if is_user else "assistant",
        "content": text,
    })


def clear_conversation_history() -> None:
    global conversation_history, recording_counter
    conversation_history.clear()
    recording_counter = 0


def get_conversation_history_length() -> int:
    return len(conversation_history)


def increment_recording_counter() -> None:
    global recording_counter
    recording_counter += 1


def build_qwen_messages(user_text: str, system_prompt: str) -> List[Dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def translate_emotion_with_history(client: OpenAI, user_text: str, system_prompt: str) -> str:
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


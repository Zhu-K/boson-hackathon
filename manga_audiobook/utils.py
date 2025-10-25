"""
Utility functions for manga audiobook generation.

This module contains helper functions for audio processing, file operations,
and prompt management.
"""

import base64
import io
import os
import wave
from pathlib import Path
from typing import Iterator, List, Dict
from dialogue_parser import DialogueElement


def b64_from_bytes(data: bytes) -> str:
    """Return the base64 encoding of raw bytes."""
    return base64.b64encode(data).decode("utf-8")


def collect_audio_from_stream(stream: Iterator) -> bytes:
    """Collect all audio chunks from a streaming response into bytes."""
    audio_chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        audio_field = getattr(delta, "audio", None)
        if not audio_field:
            continue
        
        if isinstance(audio_field, dict):
            b64_data = audio_field.get("data")
        else:
            b64_data = getattr(audio_field, "data", None)
        
        if b64_data:
            pcm_bytes = base64.b64decode(b64_data)
            audio_chunks.append(pcm_bytes)
    
    return b"".join(audio_chunks)


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Convert raw PCM audio to WAV format.

    Args:
        pcm_data: Raw PCM audio (24kHz, 16-bit, mono)

    Returns:
        WAV formatted audio bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(24000)  # 24kHz
        wav_file.writeframes(pcm_data)

    return wav_buffer.getvalue()


def concatenate_wav_files(wav_data_list: List[bytes]) -> bytes:
    """Concatenate multiple WAV files into one.

    Args:
        wav_data_list: List of WAV file data as bytes

    Returns:
        Single concatenated WAV file as bytes
    """
    if not wav_data_list:
        return b""

    if len(wav_data_list) == 1:
        return wav_data_list[0]

    # Collect all PCM frames from all WAV files
    all_frames = []
    sample_rate = None
    sample_width = None
    channels = None

    for wav_data in wav_data_list:
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            # Get parameters from first file
            if sample_rate is None:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()

            # Read all frames
            frames = wav_file.readframes(wav_file.getnframes())
            all_frames.append(frames)

    # Create concatenated WAV file
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, 'wb') as output_wav:
        output_wav.setnchannels(channels)
        output_wav.setsampwidth(sample_width)
        output_wav.setframerate(sample_rate)
        output_wav.writeframes(b"".join(all_frames))

    return output_buffer.getvalue()


def save_prompt_to_file(messages: List[Dict], element: DialogueElement, 
                       prompt_counter: int, prompts_dir: Path) -> None:
    """Save the prepared prompt to a text file for inspection."""
    filename = f"{prompt_counter:03d}_{element.speaker}_{element.text[:30].replace(' ', '_')}.txt"
    filepath = prompts_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"PROMPT #{prompt_counter}\n")
        f.write(f"Speaker: {element.speaker}\n")
        f.write(f"Text: {element.text}\n")
        f.write(f"Type: {element.dialogue_type.value}\n")
        if element.tone_modifier:
            f.write(f"Tone: {element.tone_modifier}\n")
        f.write("="*80 + "\n\n")
        
        for i, msg in enumerate(messages, 1):
            f.write(f"[Message {i}] Role: {msg['role']}\n")
            f.write("-" * 80 + "\n")
            
            content = msg.get('content')
            if isinstance(content, str):
                f.write(f"{content}\n")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'input_audio':
                            audio_info = item.get('input_audio', {})
                            audio_format = audio_info.get('format', 'unknown')
                            audio_data_len = len(audio_info.get('data', ''))
                            f.write(f"[AUDIO: {audio_format} format, {audio_data_len} chars base64]\n")
                        else:
                            f.write(f"[Content: {item}]\n")
            else:
                f.write(f"[Content: {content}]\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"  Saved prompt to: {filepath.name}")


def save_wav_file(wav_data: bytes, element: DialogueElement, 
                 prompt_counter: int, wav_dir: Path) -> None:
    """Save the converted WAV file for preview."""
    filename = f"{prompt_counter:03d}_{element.speaker}_{element.text[:30].replace(' ', '_')}.wav"
    filepath = wav_dir / filename
    
    with open(filepath, 'wb') as f:
        f.write(wav_data)
    
    print(f"  Saved WAV file to: {filepath.name}")


def create_output_directories(base_dir: Path) -> tuple[Path, Path]:
    """Create output directories for prompts and WAV files.
    
    Returns:
        Tuple of (prompts_dir, wav_dir)
    """
    prompts_dir = base_dir / "prepared_prompts"
    wav_dir = base_dir / "generated_wavs"
    
    prompts_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    
    return prompts_dir, wav_dir

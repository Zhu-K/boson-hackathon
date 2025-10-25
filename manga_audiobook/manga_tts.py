"""
Text-to-speech generation for manga audiobook.

This module handles TTS generation using the BosonAI API with sliding window
conditioning - using each speaker's previous outputs to condition new generations.
"""

import argparse
import base64
import io
import os
import sys
import time
import wave
from typing import Optional, List, Dict
from collections import deque
from pathlib import Path
from openai import OpenAI
import pyaudio

from manga_prompts import (
    MANGA_TTS_SYSTEM_PROMPT, 
    SPEAKER_PERSONALITIES,
    SPEAKER0_VOICE_REFERENCE_PROMPT,
    SPEAKER1_VOICE_REFERENCE_PROMPT,
    SOUND_EFFECTS
)
from dialogue_parser import DialogueElement, DialogueType, DialogueParser
from utils import (
    b64_from_bytes,
    collect_audio_from_stream,
    pcm_to_wav,
    concatenate_wav_files,
    save_prompt_to_file,
    save_wav_file,
    create_output_directories
)


def b64(path: str) -> str:
    """Return the base64 encoding of the given file on disk."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def get_client() -> OpenAI:
    """Initialize and return an OpenAI client configured for BosonAI."""
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://hackathon.boson.ai/v1")


class MangaTTSGenerator:
    """Handles TTS generation for manga dialogue with sliding window conditioning.
    
    Uses previous outputs from the same speaker to condition new generations,
    maintaining voice consistency while preventing cross-speaker contamination.
    """
    
    def __init__(self, client: OpenAI, window_size: int = 2, save_prompts: bool = False, direct_mode: bool = False, batch_size: int = 0):
        self.client = client
        self.parser = DialogueParser()
        self.window_size = window_size
        self.save_prompts = save_prompts
        self.direct_mode = direct_mode
        self.batch_size = batch_size  # Number of chunks to split dialogue into (0 = no batching)
        # Store last N (audio, transcript) tuples per speaker for conditioning
        self.speaker_history: Dict[str, deque] = {}
        # Track if we've generated first output for each speaker
        self.speaker_initialized: Dict[str, bool] = {}
        # Counter for saved prompts
        self.prompt_counter = 0

        # Create directories if saving is enabled
        if self.save_prompts:
            self.prompts_dir, self.wav_dir = create_output_directories(Path(__file__).parent)
        
    def get_voice_reference_for_speaker(self, speaker: str) -> tuple[Optional[str], Optional[str]]:
        """Get voice reference prompt and audio path for a speaker."""
        if speaker == "SPEAKER0":
            return SPEAKER0_VOICE_REFERENCE_PROMPT, SPEAKER_PERSONALITIES["SPEAKER0"]["reference_path"]
        elif speaker == "SPEAKER1":
            return SPEAKER1_VOICE_REFERENCE_PROMPT, SPEAKER_PERSONALITIES["SPEAKER1"]["reference_path"]
        else:
            return None, None
    
    def _add_to_history(self, speaker: str, audio_data: bytes, transcript: str):
        """Add generated audio and transcript to speaker's history (sliding window)."""
        if speaker not in self.speaker_history:
            self.speaker_history[speaker] = deque(maxlen=self.window_size)
        self.speaker_history[speaker].append((audio_data, transcript))
        self.speaker_initialized[speaker] = True
    
    def _save_prompt_and_wav(self, messages: List[Dict], element: DialogueElement, wav_data: bytes = None):
        """Save prompt and WAV file if enabled."""
        if not self.save_prompts:
            return
        
        self.prompt_counter += 1
        save_prompt_to_file(messages, element, self.prompt_counter, self.prompts_dir)
        
        if wav_data:
            save_wav_file(wav_data, element, self.prompt_counter, self.wav_dir)
    
    def generate_speech(self, element: DialogueElement) -> Optional[bytes]:
        """Generate TTS for a single dialogue element with sliding window conditioning.
        
        Uses initial reference ONLY for the first generation per speaker.
        All subsequent generations condition on previous outputs with transcripts.
        
        Returns:
            Raw PCM audio bytes (24kHz, 16-bit, mono) or None for sound effects
        """
        if element.dialogue_type == DialogueType.SOUND_EFFECT:
            return None
        
        # Get voice reference
        voice_prompt, voice_path = self.get_voice_reference_for_speaker(element.speaker)
        if voice_prompt is None or voice_path is None:
            print(f"Warning: No voice reference for '{element.speaker}'")
            return None
        
        # Format the text
        formatted_text = self.parser.format_for_tts(element)

        # Check if this is the first generation for this speaker
        is_first_generation = not self.speaker_initialized.get(element.speaker, False)

        # Build system prompt with tone modifier if present
        system_prompt = MANGA_TTS_SYSTEM_PROMPT
        if element.tone_modifier:
            system_prompt += f"\n\nYou must use this tone: {element.tone_modifier}"

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Always add initial reference audio first
        messages.append({
            "role": "user", 
            "content": f"[{element.speaker}] {voice_prompt}"
        })
        if os.path.exists(voice_path):
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": b64(voice_path), "format": "wav"},
                }],
            })
        
        # Add previous outputs from same speaker (sliding window) with transcripts
        # This provides additional conditioning on top of the original reference
        if element.speaker in self.speaker_history and len(self.speaker_history[element.speaker]) > 0:
            for i, (prev_audio_wav, prev_transcript) in enumerate(self.speaker_history[element.speaker]):
                # Add transcript as user message
                messages.append({
                    "role": "user", 
                    "content": f"[{element.speaker}] {prev_transcript}"
                })
                # Add corresponding audio as assistant response (already in WAV format)
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_from_bytes(prev_audio_wav),
                            "format": "wav"
                        },
                    }],
                })
        
        # Add the actual dialogue to generate
        messages.append({"role": "user", "content": formatted_text})
        
        # Generate audio
        stream = self.client.chat.completions.create(
            model="higgs-audio-generation-Hackathon",
            messages=messages,
            modalities=["text", "audio"],
            max_completion_tokens=4096,
            temperature=1.0,
            top_p=0.9,
            stream=True,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": 40},
        )
        
        # Collect audio (raw PCM)
        audio_pcm = collect_audio_from_stream(stream)
        
        # Convert to WAV format and add to history for future conditioning
        if audio_pcm:
            audio_wav = pcm_to_wav(audio_pcm)
            self._add_to_history(element.speaker, audio_wav, element.text)
            
            # Save prompt and WAV file for inspection if enabled
            self._save_prompt_and_wav(messages, element, audio_wav)
        else:
            # Save prompt even if no audio generated
            self._save_prompt_and_wav(messages, element)
        
        # Return raw PCM for playback
        return audio_pcm
    
    def generate_direct_audiobook(self, dialogue_file: str) -> bytes:
        """Generate audio for all dialogue elements in one direct API call.

        Combines all elements into a single prompt with reference audios.

        Note: In direct mode, tone modifiers are included inline in the text
        since we can't have a per-element system prompt when combining all dialogue.
        """
        elements = self.parser.parse_dialogue_file(dialogue_file)

        # Build combined prompt
        messages = [
            {"role": "system", "content": MANGA_TTS_SYSTEM_PROMPT},
        ]
        
        # Add voice references for both speakers
        for speaker in ["SPEAKER0", "SPEAKER1"]:
            voice_prompt, voice_path = self.get_voice_reference_for_speaker(speaker)
            if voice_prompt and voice_path and os.path.exists(voice_path):
                messages.append({
                    "role": "user", 
                    "content": f"[{speaker}] {voice_prompt}"
                })
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": b64(voice_path), "format": "wav"},
                    }],
                })
        
        # Combine all dialogue elements
        # In direct mode, include tone in the text since we can't have per-element system prompts
        combined_text_parts = []
        for element in elements:
            if element.dialogue_type == DialogueType.SOUND_EFFECT:
                combined_text_parts.append(f"[SOUND_EFFECT: {element.text}]")
            else:
                formatted_text = self.parser.format_for_tts(element, include_tone=True)
                combined_text_parts.append(formatted_text)
        
        combined_text = "\n".join(combined_text_parts)
        messages.append({"role": "user", "content": combined_text})
        
        print(f"Generating audio for ALL {len(elements)} elements in direct mode...")
        print(f"Combined text preview: {combined_text[:200]}...")
        
        # Save combined prompt
        if self.save_prompts:
            dummy_element = DialogueElement(
                speaker="ALL",
                text="Combined_Direct_Mode",
                dialogue_type=DialogueType.SPEECH
            )
            self._save_prompt_and_wav(messages, dummy_element)
        
        # Generate audio
        stream = self.client.chat.completions.create(
            model="higgs-audio-generation-Hackathon",
            messages=messages,
            modalities=["text", "audio"],
            max_completion_tokens=4096,
            temperature=1.0,
            top_p=0.9,
            stream=True,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": 40},
        )
        
        # Collect and return audio
        audio_pcm = collect_audio_from_stream(stream)
        
        if audio_pcm and self.save_prompts:
            audio_wav = pcm_to_wav(audio_pcm)
            save_wav_file(audio_wav, dummy_element, self.prompt_counter, self.wav_dir)
        
        return audio_pcm

    def generate_batched_audiobook(self, dialogue_file: str, num_chunks: int) -> bytes:
        """Generate audio in batches, using each batch's output as context for the next.

        Args:
            dialogue_file: Path to dialogue file
            num_chunks: Number of chunks to split dialogue into

        Returns:
            Complete audiobook as WAV file bytes
        """
        elements = self.parser.parse_dialogue_file(dialogue_file)

        # Split elements into chunks
        chunk_size = len(elements) // num_chunks
        if chunk_size == 0:
            chunk_size = 1
        chunks = [elements[i:i + chunk_size] for i in range(0, len(elements), chunk_size)]

        print(f"\nBatched mode: Processing {len(elements)} elements in {len(chunks)} chunks")
        print(f"Chunk size: ~{chunk_size} elements per chunk\n")

        accumulated_wavs = []
        previous_chunk_audio = None
        previous_chunk_transcript = None

        for chunk_idx, chunk in enumerate(chunks, 1):
            print(f"\n{'='*60}")
            print(f"Processing Chunk {chunk_idx}/{len(chunks)} ({len(chunk)} elements)")
            print(f"{'='*60}")

            # Build messages for this chunk
            messages = [
                {"role": "system", "content": MANGA_TTS_SYSTEM_PROMPT},
            ]

            # Add voice references for both speakers
            for speaker in ["SPEAKER0", "SPEAKER1"]:
                voice_prompt, voice_path = self.get_voice_reference_for_speaker(speaker)
                if voice_prompt and voice_path and os.path.exists(voice_path):
                    messages.append({
                        "role": "user",
                        "content": f"[{speaker}] {voice_prompt}"
                    })
                    messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "input_audio",
                            "input_audio": {"data": b64(voice_path), "format": "wav"},
                        }],
                    })

            # Add previous chunk output as context (if exists)
            if previous_chunk_audio and previous_chunk_transcript:
                print(f"  Using previous chunk as context")
                messages.append({
                    "role": "user",
                    "content": f"Previous chunk transcript:\n{previous_chunk_transcript}"
                })
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_from_bytes(previous_chunk_audio),
                            "format": "wav"
                        },
                    }],
                })

            # Combine all elements in this chunk
            chunk_text_parts = []
            chunk_transcript_parts = []
            for element in chunk:
                if element.dialogue_type == DialogueType.SOUND_EFFECT:
                    chunk_text_parts.append(f"[SOUND_EFFECT: {element.text}]")
                    chunk_transcript_parts.append(f"[SOUND_EFFECT: {element.text}]")
                else:
                    formatted_text = self.parser.format_for_tts(element, include_tone=True)
                    chunk_text_parts.append(formatted_text)
                    # For transcript, just include speaker and text
                    chunk_transcript_parts.append(f"[{element.speaker}] {element.text}")

            combined_text = "\n".join(chunk_text_parts)
            messages.append({"role": "user", "content": combined_text})

            print(f"  Generating audio for chunk...")

            # Save prompt for this chunk
            if self.save_prompts:
                dummy_element = DialogueElement(
                    speaker="BATCHED",
                    text=f"Chunk_{chunk_idx}_of_{len(chunks)}",
                    dialogue_type=DialogueType.SPEECH
                )
                self.prompt_counter += 1
                save_prompt_to_file(messages, dummy_element, self.prompt_counter, self.prompts_dir)

            # Generate audio for this chunk
            stream = self.client.chat.completions.create(
                model="higgs-audio-generation-Hackathon",
                messages=messages,
                modalities=["text", "audio"],
                max_completion_tokens=4096,
                temperature=1.0,
                top_p=0.9,
                stream=True,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                extra_body={"top_k": 40},
            )

            # Collect audio
            audio_pcm = collect_audio_from_stream(stream)

            if audio_pcm:
                audio_wav = pcm_to_wav(audio_pcm)
                accumulated_wavs.append(audio_wav)

                # Save this chunk as previous for next iteration
                previous_chunk_audio = audio_wav
                previous_chunk_transcript = "\n".join(chunk_transcript_parts)

                print(f"  ✓ Generated {len(audio_pcm)} bytes PCM")

                # Save chunk WAV
                if self.save_prompts:
                    save_wav_file(audio_wav, dummy_element, self.prompt_counter, self.wav_dir)

                # Wait before next chunk
                if chunk_idx < len(chunks):
                    print(f"  Waiting 2 seconds before next chunk...")
                    time.sleep(2)
            else:
                print(f"  ✗ No audio generated for this chunk")

        # Concatenate all chunks
        print(f"\n{'='*60}")
        print(f"Concatenating {len(accumulated_wavs)} chunks...")
        final_wav = concatenate_wav_files(accumulated_wavs)
        print(f"Final audiobook size: {len(final_wav)} bytes")

        # Save final concatenated file
        if self.save_prompts and final_wav:
            combined_element = DialogueElement(
                speaker="ALL_SPEAKERS",
                text="Combined_Batched_Mode",
                dialogue_type=DialogueType.SPEECH
            )
            self.prompt_counter += 1
            save_wav_file(final_wav, combined_element, self.prompt_counter, self.wav_dir)

        return final_wav

    def generate_audiobook(self, dialogue_file: str) -> bytes:
        """Generate audio for all dialogue elements with sliding window conditioning.

        Returns:
            Complete audiobook as WAV file bytes (concatenated from all elements)
        """
        # Check for batched mode first
        if self.batch_size > 0:
            return self.generate_batched_audiobook(dialogue_file, self.batch_size)

        if self.direct_mode:
            # Generate everything in one shot
            audio_pcm = self.generate_direct_audiobook(dialogue_file)
            if audio_pcm:
                audio_wav = pcm_to_wav(audio_pcm)

                # Save the final WAV file
                if self.save_prompts:
                    elements = self.parser.parse_dialogue_file(dialogue_file)
                    combined_element = DialogueElement(
                        speaker="ALL_SPEAKERS",
                        text="Combined_Direct_Mode",
                        dialogue_type=DialogueType.SPEECH
                    )
                    self.prompt_counter += 1
                    save_wav_file(audio_wav, combined_element, self.prompt_counter, self.wav_dir)

                return audio_wav
            return b""

        # Sliding window mode - accumulate all WAV files
        elements = self.parser.parse_dialogue_file(dialogue_file)
        accumulated_wavs = []

        for i, element in enumerate(elements, 1):
            print(f"[{i}/{len(elements)}] Processing: {element.speaker} - \"{element.text[:40]}...\"")

            if element.dialogue_type == DialogueType.SOUND_EFFECT:
                print(f"  Sound effect: {element.text}")
                # Load and add sound effect WAV
                if element.text in SOUND_EFFECTS:
                    effect_info = SOUND_EFFECTS[element.text]
                    file_path = str(effect_info["file_path"])
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            sfx_wav = f.read()
                        accumulated_wavs.append(sfx_wav)
                        print(f"  Added sound effect: {len(sfx_wav)} bytes")
                    else:
                        print(f"  Warning: Sound effect file not found: {file_path}")
                else:
                    print(f"  Warning: Unknown sound effect: {element.text}")
            else:
                try:
                    # Show conditioning info
                    is_first = not self.speaker_initialized.get(element.speaker, False)
                    if is_first:
                        print(f"  Using initial voice reference only")
                    elif element.speaker in self.speaker_history:
                        hist_len = len(self.speaker_history[element.speaker])
                        print(f"  Using initial reference + {hist_len} previous output(s)")
                    else:
                        print(f"  Using initial voice reference only")

                    audio_pcm = self.generate_speech(element)
                    if audio_pcm:
                        # Convert to WAV and add to accumulator
                        audio_wav = pcm_to_wav(audio_pcm)
                        accumulated_wavs.append(audio_wav)
                        print(f"  Generated audio: {len(audio_pcm)} bytes PCM")
                    else:
                        print(f"  Warning: No audio generated")

                    # Wait 2 seconds before next API call to prevent rate limiting
                    if i < len(elements):  # Don't wait after the last element
                        print(f"  Waiting 2 seconds before next API call...")
                        time.sleep(2)
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

        # Concatenate all WAV files into one
        print(f"\nConcatenating {len(accumulated_wavs)} audio segments...")
        final_wav = concatenate_wav_files(accumulated_wavs)
        print(f"Final audiobook size: {len(final_wav)} bytes")

        # Save the final concatenated WAV file
        if self.save_prompts and final_wav:
            combined_element = DialogueElement(
                speaker="ALL_SPEAKERS",
                text="Combined_Sliding_Window_Mode",
                dialogue_type=DialogueType.SPEECH
            )
            self.prompt_counter += 1
            save_wav_file(final_wav, combined_element, self.prompt_counter, self.wav_dir)

        return final_wav


def play_audio(audio_data: bytes):
    """Play raw PCM audio (24kHz, 16-bit, mono)."""
    if not audio_data:
        return
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24_000,
        output=True,
    )
    
    try:
        stream.write(audio_data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def play_sound_effect(effect_name: str):
    """Play a sound effect by name."""
    if effect_name not in SOUND_EFFECTS:
        print(f"Unknown sound effect: {effect_name}")
        return
    
    effect_info = SOUND_EFFECTS[effect_name]
    file_path = str(effect_info["file_path"])
    volume = effect_info.get("volume", 1.0)
    
    if not os.path.exists(file_path):
        print(f"Sound effect file not found: {file_path}")
        return
    
    with wave.open(file_path, "rb") as wf:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
        
        try:
            chunk = 1024
            data = wf.readframes(chunk)
            while data:
                if volume != 1.0:
                    import audioop
                    data = audioop.mul(data, wf.getsampwidth(), volume)
                stream.write(data)
                data = wf.readframes(chunk)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def test_tts_generation():
    """Test TTS generation with sample dialogue to demonstrate sliding window conditioning."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test TTS generation")
    parser.add_argument("--direct", action="store_true", help="Use direct mode (all elements in one API call)")
    parser.add_argument("--batch", type=int, default=0, metavar="N", help="Use batched mode (N chunks)")
    args = parser.parse_args()

    # Validate arguments
    if args.direct and args.batch > 0:
        print("❌ Error: Cannot use --direct and --batch together. Choose one mode.")
        return

    client = get_client()
    tts_generator = MangaTTSGenerator(client, window_size=2, save_prompts=True, direct_mode=args.direct, batch_size=args.batch)
    
    # Create test dialogues - 2 per speaker
    test_dialogues = [
        DialogueElement(
            speaker="SPEAKER0",
            text="Hi anya. What do you want to do today?",
            dialogue_type=DialogueType.SPEECH,
            tone_modifier="happy",
        ),
        DialogueElement(
            speaker="SPEAKER1",
            text="Papa! Can we go to the park?",
            dialogue_type=DialogueType.SPEECH,
            tone_modifier="excited",
        ),
        DialogueElement(
            speaker="SPEAKER0",
            text="Sure we can do that.",
            dialogue_type=DialogueType.SPEECH,
            tone_modifier="disinterested",
        ),
        DialogueElement(
            speaker="SPEAKER1",
            text="Yay! Thank you so much!",
            dialogue_type=DialogueType.SPEECH,
            tone_modifier="excited",
        ),
    ]
    
    print("\n" + "="*60)
    if args.batch > 0:
        mode_name = f"Batched Mode ({args.batch} chunks)"
    elif args.direct:
        mode_name = "Direct Mode"
    else:
        mode_name = "Sliding Window Conditioning"
    print(f"Testing TTS with {mode_name}")
    print("="*60)

    if args.batch > 0:
        print(f"\nBatched mode will:")
        print(f"  - Split {len(test_dialogues)} elements into {args.batch} chunks")
        print(f"  - Each chunk uses previous chunk's output as context")
        print(f"  - Include both speaker references")
        print(f"  - Generate audio for each chunk separately")
    elif args.direct:
        print("\nDirect mode will:")
        print("  - Combine all dialogue into one API call")
        print("  - Include both speaker references")
        print("  - Generate continuous audio stream")
    else:
        print("\nSliding window mode will show:")
        print("  - SPEAKER0 line 1: Uses initial voice reference only")
        print("  - SPEAKER0 line 2: Uses initial reference + line 1 conditioning")
        print("  - SPEAKER1 line 1: Uses initial voice reference only")
        print("  - SPEAKER1 line 2: Uses initial reference + line 1 conditioning")
    
    print("\n📝 Prompts will be saved to: manga_audiobook/prepared_prompts/")
    print("🎵 WAV files will be saved to: manga_audiobook/generated_wavs/")
    print("="*60 + "\n")
    
    # Create temporary dialogue file for testing
    import tempfile
    temp_dialogue_content = []
    for element in test_dialogues:
        if element.tone_modifier:
            temp_dialogue_content.append(f'[{element.speaker}] (tone: {element.tone_modifier}) "{element.text}"')
        else:
            temp_dialogue_content.append(f'[{element.speaker}] "{element.text}"')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(temp_dialogue_content))
        temp_file = f.name
    
    try:
        # Generate audiobook using the selected mode
        final_audio_wav = tts_generator.generate_audiobook(temp_file)

        if final_audio_wav:
            print(f"\n✓ Successfully generated audiobook!")
            print(f"  Total size: {len(final_audio_wav)} bytes (WAV format)")
            print("\n  Playing complete audiobook...")

            # Play the WAV file
            with wave.open(io.BytesIO(final_audio_wav), 'rb') as wf:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )

                try:
                    chunk = 1024
                    data = wf.readframes(chunk)
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk)
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()

            print("  ✓ Playback complete")
        else:
            print("  ✗ No audio generated")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp file
        os.unlink(temp_file)

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_tts_generation()

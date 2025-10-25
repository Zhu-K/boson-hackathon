#!/usr/bin/env python3
"""
Manga Audiobook Generator

Converts manga dialogue into immersive audio with character voices using
sliding window conditioning for voice consistency.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from manga_tts import MangaTTSGenerator, get_client
from manga_prompts import SPEAKER_PERSONALITIES


def load_environment():
    """Load environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, using system environment variables")


def print_header():
    """Print information about the manga audiobook."""
    print("\n" + "="*60)
    print("MANGA AUDIOBOOK GENERATOR")
    print("="*60)
    print("\nCharacters:")
    for speaker_id, info in SPEAKER_PERSONALITIES.items():
        print(f"  {speaker_id} ({info['name']}): {info['description']}")
    print("\nFeatures:")
    print("  - Sliding window conditioning (uses last 2 outputs per speaker)")
    print("  - Prevents cross-speaker tone contamination")
    print("  - Maintains voice consistency within each character")
    print("="*60 + "\n")


def check_audio_files():
    """Check if required audio files exist."""
    audio_dir = Path(__file__).parent.parent / "audios"
    audio_dir.mkdir(exist_ok=True)
    
    missing_files = []
    
    for speaker_id, info in SPEAKER_PERSONALITIES.items():
        ref_path = Path(info["reference_path"])
        if not ref_path.exists():
            missing_files.append(f"{speaker_id}: {ref_path}")
    
    if missing_files:
        print("⚠️  Warning: Missing reference audio files:")
        for file in missing_files:
            print(f"  - {file}")
        print()


def play_audiobook(tts_generator: MangaTTSGenerator, dialogue_file: str):
    """Generate and play the manga audiobook."""
    import io
    import wave
    import pyaudio

    print("Starting audiobook generation...\n")
    print("="*60)

    # Generate the complete audiobook (returns WAV bytes)
    final_audio_wav = tts_generator.generate_audiobook(dialogue_file)

    if not final_audio_wav:
        print("\n⚠️  No audio generated\n")
        return

    print(f"\n✅ Successfully generated audiobook!")
    print(f"   Total size: {len(final_audio_wav)} bytes (WAV format)")
    print("\n🎵 Playing complete audiobook...\n")

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

    print("\n✓ Playback complete\n")
    print("-" * 60)


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate manga audiobook")
    parser.add_argument("--direct", action="store_true",
                        help="Use direct mode (all elements in one API call)")
    parser.add_argument("--batch", type=int, default=0, metavar="N",
                        help="Use batched mode (split dialogue into N chunks, each chunk uses previous as context)")
    parser.add_argument("--save-prompts", action="store_true",
                        help="Save prompts and WAV files for inspection")
    args = parser.parse_args()

    # Validate arguments
    if args.direct and args.batch > 0:
        print("❌ Error: Cannot use --direct and --batch together. Choose one mode.")
        return 1

    load_environment()
    print_header()
    check_audio_files()

    # Check for dialogue file
    dialogue_file = Path(__file__).parent / "manga_panel.txt"
    if not dialogue_file.exists():
        print(f"❌ Error: Dialogue file not found: {dialogue_file}")
        return 1

    try:
        # Initialize
        print("Initializing TTS generator...")
        client = get_client()
        tts_generator = MangaTTSGenerator(
            client,
            window_size=2,
            save_prompts=args.save_prompts,
            direct_mode=args.direct,
            batch_size=args.batch
        )
        
        # Parse and show preview
        print("Parsing dialogue...")
        from dialogue_parser import DialogueParser
        parser = DialogueParser()
        elements = parser.parse_dialogue_file(str(dialogue_file))
        print(f"Found {len(elements)} dialogue elements\n")
        
        # Show preview
        print("Dialogue Preview:")
        for i, element in enumerate(elements[:5], 1):
            print(f"  {i}. {element.speaker}: {element.text[:50]}...")
        if len(elements) > 5:
            print(f"  ... and {len(elements) - 5} more elements\n")
        
        print()
        
        # Generate and play
        play_audiobook(tts_generator, str(dialogue_file))
        
        print("\n" + "="*60)
        print("✅ Audiobook complete!")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

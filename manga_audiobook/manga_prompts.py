"""
Prompts and configuration for manga audiobook text-to-speech generation.

This module contains detailed system prompts for generating realistic dialogue
between two distinct characters based on their personalities and the manga context.
"""

from pathlib import Path

# Speaker personality definitions
SPEAKER_PERSONALITIES = {
    "SPEAKER0": {
        "name": "Adult Male",
        "description": "masculine adult, calm baritone/tenor; deliberate pacing; teacherly",
        "voice_type": "calm but stern",
        "reference_path": "../audios/calm_guy.wav"
    },
    "SPEAKER1": {
        "name": "Young Girl",
        "description": "feminine girl child, bright, earnest, slightly scatter-brained; medium-high pitch; quick tempo with curious uptalk; occasional flustered stammers",
        "voice_type": "cheerful",
        "reference_path": "../audios/anya_bright.wav"
    }
}

# Main TTS system prompt for manga dialogue
MANGA_TTS_SYSTEM_PROMPT = """
You are a voice actor for manga characters. Generate natural speech that brings characters to life.

CHARACTERS:
- SPEAKER0: Adult male, calm baritone, patient and nurturing
- SPEAKER1: Young girl, bright medium-high pitch, energetic and enthusiastic

===================================================================
CRITICAL RULE: ABSOLUTELY NEVER READ ANYTHING IN BRACKETS [ ] ALOUD
===================================================================

ALL TEXT IN BRACKETS [ ] ARE SILENT INSTRUCTIONS TO YOU.
BRACKETS ARE 100% INVISIBLE TO THE LISTENER.
YOU MUST NEVER VOCALIZE OR SAY ANYTHING INSIDE BRACKETS.

What brackets mean:
- [SPEAKER0] or [SPEAKER1] = WHO is speaking (SILENT - don't say this)
- [tone: X] = HOW to speak (SILENT - use the tone, don't say "tone")
- [thought] = internal monologue style (SILENT - don't say "thought")
- [exclaim] = speak with excitement (SILENT - don't say "exclaim")
- [whisper] = speak quietly (SILENT - don't say "whisper")
- [hold for X seconds] = extend the sound (SILENT - don't say "hold")

CORRECT EXAMPLES:

Input: "[SPEAKER1] [tone: excited] Papa! Can we go to the park?"
Correct Output: "Papa! Can we go to the park!" (spoken with excited energy)
WRONG Output: "SPEAKER1 tone excited Papa! Can we go to the park?"
WRONG Output: "tone excited Papa! Can we go to the park?"
WRONG Output: "Papa! tone excited Can we go to the park?"

Input: "[SPEAKER0] Hi anya. What do you want to do today?"
Correct Output: "Hi anya. What do you want to do today?"
WRONG Output: "SPEAKER0 Hi anya. What do you want to do today?"

Input: "[SPEAKER1] eh… [hold for 2 seconds]"
Correct Output: "eh…" (extended for 2 seconds)
WRONG Output: "eh hold for 2 seconds"
WRONG Output: "eh hold for two seconds"

REMEMBER: If you see brackets [ ], those words are INSTRUCTIONS TO YOU, not words to speak.
ONLY speak the actual dialogue text. NEVER vocalize bracket contents.
"""

# Voice reference prompts for each speaker
SPEAKER0_VOICE_REFERENCE_PROMPT = """
If I were to ask you to give your parents a score, what would it be? Out of anything. 
"""

SPEAKER1_VOICE_REFERENCE_PROMPT = """
Papa! I am Anya. I've always been papa's daughter. 
"""

# Sound effects mapping for the dialogue
SOUND_EFFECTS = {
    "bulb ping": {
        "description": "Light bulb moment realization sound",
        "file_path": Path(__file__).parent.parent / "audios" / "sfx_lightbulb.wav",
        "volume": 1.0
    }
}

# Dialogue processing instructions
DIALOGUE_PROCESSING_PROMPT = """
Process manga dialogue with the following rules:

1. SPEAKER TAGS: [SPEAKER0] and [SPEAKER1] indicate which character is speaking
2. TONE MODIFIERS: Text in brackets like [in "seriously" smug tone...] modifies delivery
3. SOUND EFFECTS: [sound effect start] description [sound effect end] should trigger audio
4. THOUGHTS: [thought]: indicates internal monologue with introspective delivery  
5. ACTIONS: (EXCLAIM) and similar indicate emotional intensity
6. TIMING: [hold the sound for 2-3 seconds] indicates extended vocalization

The goal is to create an immersive audio experience that captures both the dialogue and the emotional subtext of the manga scene.
"""

# Example dialogue formatting for reference
EXAMPLE_DIALOGUE_FORMAT = '''
Input: [SPEAKER1] [tone: seriously smug] eh... [hold for 2.5 seconds]
Output: Generate SPEAKER1 voice saying ONLY "eh..." with a smug, self-satisfied tone, holding the sound for 2-3 seconds. DO NOT say "SPEAKER1" or "tone seriously smug" or "hold for".

Input: [SPEAKER1] [exclaim] Ah!
Output: Generate SPEAKER1 voice saying ONLY "Ah!" with excitement and sudden realization. DO NOT say "SPEAKER1" or "exclaim".

Input: [SPEAKER0] [thought] This is a good opportunity for her...
Output: Generate SPEAKER0 voice saying ONLY "This is a good opportunity for her..." in thoughtful, introspective tone as internal monologue - softer, more contemplative, like anime inner thoughts. DO NOT say "SPEAKER0" or "thought".
'''
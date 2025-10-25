"""
Dialogue parser for manga audiobook generation.

This module handles parsing of manga dialogue with speaker tags, tone modifiers,
sound effects, and other special formatting.

STANDARDIZED FORMAT:
- Speaker tags: [SPEAKER0] or [SPEAKER1]
- Tone modifiers: (tone: description)
- Thoughts: (thought)
- Volume: (exclaim), (whisper), etc.
- Duration: (hold: seconds)
- Sound effects: [sound_effect_start] name [sound_effect_end]
- Dialogue text: Always in "quotes"
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class DialogueType(Enum):
    SPEECH = "speech"
    THOUGHT = "thought" 
    SOUND_EFFECT = "sound_effect"
    PAUSE = "pause"

@dataclass
class DialogueElement:
    """Represents a single element of dialogue or audio."""
    speaker: Optional[str]  # SPEAKER0, SPEAKER1, or None for sound effects
    text: str
    dialogue_type: DialogueType
    tone_modifier: Optional[str] = None
    duration: Optional[float] = None  # For pauses or extended sounds
    volume_modifier: Optional[str] = None  # exclaim, whisper, etc.

class DialogueParser:
    """Parses manga dialogue text into structured audio elements."""
    
    def __init__(self):
        # Regex patterns for standardized format
        self.speaker_pattern = r'\[SPEAKER(\d+)\]'
        self.sound_effect_pattern = r'\[sound_effect_start\]\s*([^\[]+?)\s*\[sound_effect_end\]'
        self.tone_pattern = r'\(tone:\s*([^)]+)\)'
        self.thought_pattern = r'\(thought\)'
        self.duration_pattern = r'\(hold:\s*([\d.]+)\)'
        self.exclaim_pattern = r'\(exclaim\)'
        self.whisper_pattern = r'\(whisper\)'
        self.quoted_text_pattern = r'"([^"]+)"'
        
    def parse_line(self, line: str) -> List[DialogueElement]:
        """Parse a single line of dialogue into structured elements.
        
        Format:
        [SPEAKER0] (modifiers) "dialogue text"
        [sound_effect_start] effect name [sound_effect_end]
        """
        elements = []
        
        # Remove leading/trailing whitespace
        line = line.strip()
        if not line:
            return elements
        
        # Check for sound effects first (they don't have speakers)
        sound_effect_match = re.search(self.sound_effect_pattern, line)
        if sound_effect_match:
            effect_name = sound_effect_match.group(1).strip()
            elements.append(DialogueElement(
                speaker=None,
                text=effect_name,
                dialogue_type=DialogueType.SOUND_EFFECT
            ))
            return elements
        
        # Extract speaker tag
        speaker_match = re.search(self.speaker_pattern, line)
        if not speaker_match:
            return elements  # No speaker, skip this line
        
        current_speaker = f"SPEAKER{speaker_match.group(1)}"
        
        # Extract modifiers
        is_thought = bool(re.search(self.thought_pattern, line))
        is_exclaim = bool(re.search(self.exclaim_pattern, line))
        is_whisper = bool(re.search(self.whisper_pattern, line))
        
        tone_match = re.search(self.tone_pattern, line)
        tone_modifier = tone_match.group(1).strip() if tone_match else None
        
        duration_match = re.search(self.duration_pattern, line)
        duration = float(duration_match.group(1)) if duration_match else None
        
        # Extract quoted text
        quoted_matches = re.findall(self.quoted_text_pattern, line)
        if not quoted_matches:
            return elements  # No dialogue text found
        
        # Determine dialogue type and volume modifier
        dialogue_type = DialogueType.THOUGHT if is_thought else DialogueType.SPEECH
        volume_modifier = None
        if is_exclaim:
            volume_modifier = "exclaim"
        elif is_whisper:
            volume_modifier = "whisper"
        
        # Create dialogue element for each quoted text
        for text in quoted_matches:
            elements.append(DialogueElement(
                speaker=current_speaker,
                text=text,
                dialogue_type=dialogue_type,
                tone_modifier=tone_modifier,
                duration=duration,
                volume_modifier=volume_modifier
            ))
        
        return elements
    
    def parse_dialogue_file(self, file_path: str) -> List[DialogueElement]:
        """Parse an entire dialogue file into structured elements."""
        all_elements = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                elements = self.parse_line(line)
                all_elements.extend(elements)
        
        return all_elements
    
    def format_for_tts(self, element: DialogueElement, include_tone: bool = False) -> str:
        """Format a dialogue element for TTS generation.

        Args:
            element: The dialogue element to format
            include_tone: If True, includes tone modifier in the formatted text.
                         Use this for direct/combined mode where per-element system
                         prompts aren't possible. Default False (tone goes in system prompt).

        Note: By default, tone modifier is NOT included in the formatted text.
        It should be added to the system prompt instead for better results.
        """
        if element.dialogue_type == DialogueType.SOUND_EFFECT:
            return f"[SOUND_EFFECT: {element.text}]"

        # Start with speaker tag
        formatted = f"[{element.speaker}] "

        # Add dialogue type if thought
        if element.dialogue_type == DialogueType.THOUGHT:
            formatted += "[thought] "

        # Add tone modifier if present and requested (for direct mode)
        if include_tone and element.tone_modifier:
            formatted += f"[tone: {element.tone_modifier}] "

        # Add volume modifier if present
        if element.volume_modifier:
            formatted += f"[{element.volume_modifier}] "

        # Add the text
        formatted += element.text

        # Add duration if specified
        if element.duration:
            formatted += f" [hold for {element.duration} seconds]"

        return formatted
    
    def format_element_details(self, element: DialogueElement) -> str:
        """Format a dialogue element with all its attributes clearly labeled."""
        parts = []
        
        # Speaker
        if element.speaker:
            parts.append(f"Speaker: {element.speaker}")
        else:
            parts.append("Speaker: [SOUND_EFFECT]")
        
        # Type
        parts.append(f"Type: {element.dialogue_type.value.upper()}")
        
        # Text
        parts.append(f"Text: \"{element.text}\"")
        
        # Optional attributes
        if element.tone_modifier:
            parts.append(f"Tone: {element.tone_modifier}")
        if element.volume_modifier:
            parts.append(f"Volume: {element.volume_modifier}")
        if element.duration:
            parts.append(f"Duration: {element.duration}s")
        
        return " | ".join(parts)

def test_parser():
    """Test the dialogue parser with sample input using the standardized format."""
    parser = DialogueParser()
    
    print("\n" + "=" * 80)
    print("TESTING DIALOGUE PARSER WITH STANDARDIZED FORMAT")
    print("=" * 80)
    print("\nFormat Rules:")
    print("- Speaker: [SPEAKER0] or [SPEAKER1]")
    print("- Tone: (tone: description)")
    print("- Thought: (thought)")
    print("- Volume: (exclaim) or (whisper)")
    print("- Duration: (hold: seconds)")
    print("- Sound Effect: [sound_effect_start] name [sound_effect_end]")
    print("- Text: Always in \"quotes\"")
    print("=" * 80)
    print()
    
    test_lines = [
        '[SPEAKER1] (tone: seriously smug) "eh…" (hold: 2.5)',
        '[sound_effect_start] bulb ping [sound_effect_end]',
        '[SPEAKER1] "Ah!"',
        '[SPEAKER0] (thought) "This is a good opportunity for her…"',
        '[SPEAKER0] (exclaim) "That\'s close!!"',
        '[SPEAKER0] (tone: doubtful uneasy) "Oh, well…"'
    ]
    
    for i, line in enumerate(test_lines, 1):
        print(f"[{i}] Input: {line}")
        elements = parser.parse_line(line)
        if elements:
            for element in elements:
                print(f"    Parsed: {parser.format_element_details(element)}")
                print(f"    For TTS: {parser.format_for_tts(element)}")
        else:
            print(f"    Error: No elements parsed")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    test_parser()

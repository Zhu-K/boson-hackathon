#!/usr/bin/env python3
"""
Test script to verify the dialogue parser works with the manga panel content.
"""

from dialogue_parser import DialogueParser
from pathlib import Path

def test_manga_panel():
    """Test parsing the actual manga panel dialogue."""
    parser = DialogueParser()
    
    # Test with the actual manga panel file
    manga_file = Path(__file__).parent / "manga_panel.txt"
    
    print("\n" + "=" * 70)
    print("TESTING DIALOGUE PARSER WITH MANGA_PANEL.TXT")
    print("=" * 70)
    
    if not manga_file.exists():
        print(f"Error: {manga_file} not found")
        return
    
    elements = parser.parse_dialogue_file(str(manga_file))
    
    print(f"\n✓ Successfully parsed {len(elements)} dialogue elements\n")
    print("=" * 70)
    print("PARSED ELEMENTS:")
    print("=" * 70)
    
    for i, element in enumerate(elements, 1):
        print(f"\n{i:2d}. {parser.format_element_details(element)}")
    
    print("\n" + "=" * 70)
    print("FORMATTED FOR TTS API:")
    print("=" * 70 + "\n")
    
    for i, element in enumerate(elements, 1):
        formatted = parser.format_for_tts(element)
        print(f"{i:2d}. {formatted}")
    
    print("\n" + "=" * 70)
    print(f"✓ All {len(elements)} elements ready for TTS generation")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    test_manga_panel()

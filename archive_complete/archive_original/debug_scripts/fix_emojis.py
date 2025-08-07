#!/usr/bin/env python3
"""
Quick fix to remove emojis from server script
"""

import re

def fix_emojis():
    script_path = r"c:\Users\Benchoff\Documents\GitHub\OrthoRoute\addon_package\plugins\orthoroute_standalone_server.py"
    
    # Read file
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements
    replacements = {
        '🚀': '[START]',
        '📁': '[DIR]',
        '📊': '[INFO]',
        '🔧': '[LOAD]',
        '✅': '[OK]',
        '❌': '[ERROR]', 
        '⚠': '[WARN]',
        '💥': '[CRASH]',
        '⏰': '[TIME]',
        '⌨️': '[KEY]'
    }
    
    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining unicode emojis using regex
    content = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF]', '[EMOJI]', content)
    
    # Write back
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Emojis replaced successfully")

if __name__ == "__main__":
    fix_emojis()

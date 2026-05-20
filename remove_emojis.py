import os
import glob
import re

def remove_emojis():
    files = glob.glob("0*.py") + glob.glob("10*.py") + ["run_pipeline.py"]
    
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace common emojis used by subagents
        emojis_to_remove = ["📂", "🏆", "⏳", "📊", "💾", "✅", "⚠️", "❌", "✔", "🔍"]
        
        changed = False
        for emoji in emojis_to_remove:
            if emoji in content:
                content = content.replace(emoji, "")
                changed = True
        
        # Also let's try to remove any high surrogate or emojis using a regex just in case
        # Basic emoji range
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        if emoji_pattern.search(content):
            content = emoji_pattern.sub(r'', content)
            changed = True
            
        if changed:
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Removed emojis from {file}")

if __name__ == "__main__":
    remove_emojis()

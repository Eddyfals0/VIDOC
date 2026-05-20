import os
import glob
import re

def fix_classes():
    files = glob.glob("0*.py") + glob.glob("10*.py")
    
    dynamic_classes_code = """
import os
from pathlib import Path

# Cargar clases dinámicamente de lo que se haya logrado descargar (ej. 14 de 16)
_dataset_path = Path("dataset")
if _dataset_path.exists():
    CLASSES = sorted([d.name for d in _dataset_path.iterdir() if d.is_dir()])
else:
    CLASSES = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ]
"""

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Match the hardcoded CLASSES list
        pattern = r"CLASSES\s*=\s*\[\s*(?:\"[^\"]+\",?\s*)+\]"
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, dynamic_classes_code.strip(), content)
            
            with open(file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Fixed {file}")
        else:
            print(f"No match found in {file}")

if __name__ == "__main__":
    fix_classes()

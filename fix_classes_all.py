import os
import glob
import re

def fix_classes_all():
    files = glob.glob("0*.py") + glob.glob("10*.py")
    
    dynamic_classes_code = """
import os
from pathlib import Path

# Cargar clases dinámicamente de lo que se haya logrado descargar (ej. 14 de 16)
_dataset_path = Path("dataset")
if _dataset_path.exists():
    _clases_encontradas = sorted([d.name for d in _dataset_path.iterdir() if d.is_dir()])
    if _clases_encontradas:
        CLASES = _clases_encontradas
        CLASSES = _clases_encontradas
    else:
        CLASES = ["letter", "form", "email", "handwritten", "advertisement", "scientific_report", "scientific_publication", "specification", "file_folder", "news_article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
        CLASSES = CLASES
else:
    CLASES = ["letter", "form", "email", "handwritten", "advertisement", "scientific_report", "scientific_publication", "specification", "file_folder", "news_article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]
    CLASSES = CLASES
"""

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Match CLASES = [...] or CLASSES = [...]
        pattern1 = r"CLASSES\s*=\s*\[[\s\S]*?\]"
        pattern2 = r"CLASES\s*=\s*\[[\s\S]*?\]"
        
        changed = False
        if re.search(pattern1, content) and "Cargar clases dinámicamente" not in content:
            content = re.sub(pattern1, dynamic_classes_code.strip(), content)
            changed = True
        
        if re.search(pattern2, content) and "Cargar clases dinámicamente" not in content:
            content = re.sub(pattern2, dynamic_classes_code.strip(), content)
            changed = True
            
        if changed:
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed {file}")
        else:
            print(f"No match or already fixed in {file}")

if __name__ == "__main__":
    fix_classes_all()

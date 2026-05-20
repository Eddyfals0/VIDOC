"""
Completa la descarga de las clases faltantes del dataset.
Evita caracteres Unicode (como el check) que causan problemas en cmd/PowerShell de Windows.
"""
import csv
import re
import io
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()

def main():
    output_dir = Path("dataset")
    
    # Contar existentes
    existing = {}
    for c in CLASSES:
        sname = safe_name(c)
        d = output_dir / sname
        if d.exists():
            existing[c] = len(list(d.glob("*.jpg")))
        else:
            d.mkdir(parents=True, exist_ok=True)
            existing[c] = 0
    
    target = 1000
    faltantes = {c: target - existing[c] for c in CLASSES if existing[c] < target}
    
    if not faltantes:
        print("Dataset completo! Todas las clases tienen 1000 imagenes.")
        return
    
    print("Clases faltantes:")
    for c, n in faltantes.items():
        print(f"  {c}: {existing[c]}/{target} (faltan {n})")
    
    labels_path = output_dir / "labels.csv"
    csv_file = open(labels_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["path", "class_name", "label"])
    
    counts = dict(existing)
    total_needed = sum(faltantes.values())
    
    print(f"\nConectando a HuggingFace...")
    # Probamos de nuevo con el split 'test'
    dataset = load_dataset("chainyo/rvl-cdip", split="test", streaming=True)
    
    pbar = tqdm(total=total_needed, desc="Completando", unit="img")
    
    try:
        for row in dataset:
            label = row.get("label", -1)
            if label < 0 or label >= len(CLASSES):
                continue
            
            class_name = CLASSES[label]
            
            if class_name not in faltantes or counts[class_name] >= target:
                if all(counts.get(c, 0) >= target for c in faltantes):
                    break
                continue
            
            img = row.get("image")
            if img is None:
                continue
            
            try:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                
                counts[class_name] += 1
                sname = safe_name(class_name)
                file_name = f"{sname}_{counts[class_name]:05d}.jpg"
                file_path = output_dir / sname / file_name
                
                img.save(file_path, "JPEG", quality=95)
                
                writer.writerow({
                    "path": str(file_path.as_posix()),
                    "class_name": class_name,
                    "label": label,
                })
                pbar.update(1)
            except Exception as e:
                # Ignorar imagenes corruptas o no identificables silenciosamente
                counts[class_name] -= 1
                continue
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print(f"\nError de conexion o descarga: {e}")
    finally:
        csv_file.close()
        pbar.close()
    
    print("\nResumen final:")
    total = 0
    for c in CLASSES:
        n = counts.get(c, 0)
        total += n
        status = "[OK]" if n >= target else "[FALTA]"
        print(f"  {status} {c}: {n}/{target}")
    print(f"\nTotal: {total} imagenes")

if __name__ == "__main__":
    main()

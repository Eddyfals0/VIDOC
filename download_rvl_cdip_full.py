import argparse
import csv
import os
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATASET = "sabaridsnfuji/rvl-cdip-filtered"

CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def safe_name(value):
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()

def download_dataset(output_dir, samples_per_class):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {c: 0 for c in CLASSES}
    
    # Pre-crear directorios
    for c in CLASSES:
        (output_dir / safe_name(c)).mkdir(parents=True, exist_ok=True)
        
    records = []
    labels_path = output_dir / "labels_full.csv"
    fieldnames = ["path", "class_name", "label"]
    
    print(f"Iniciando descarga en streaming desde {DATASET}")
    print(f"Objetivo: {samples_per_class} imágenes por clase.")
    print("\n" + "="*60)
    print("ATENCIÓN: La librería 'datasets' necesita descargar el primer")
    print("bloque de datos (aprox 1 GB) antes de extraer la primera imagen.")
    print("ESTO PUEDE TARDAR VARIOS MINUTOS Y PARECER CONGELADO AL 0%.")
    print("Por favor, ten paciencia. El progreso comenzará una vez baje el bloque.")
    print("="*60 + "\n", flush=True)
    
    # Cargar en modo streaming para no saturar memoria/disco innecesariamente
    dataset = load_dataset(DATASET, split="train", streaming=True)
    
    # Preparar el CSV
    csv_file = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    pbar = tqdm(total=samples_per_class * len(CLASSES), desc="Descargando imágenes")
    
    try:
        for row in dataset:
            # Obtener datos
            class_name = row.get("class_name", "")
            if class_name not in CLASSES:
                # Tratar de obtener de 'label' si no hay class_name
                label = row.get("label", -1)
                if 0 <= label < len(CLASSES):
                    class_name = CLASSES[label]
                else:
                    continue
                    
            if counts[class_name] >= samples_per_class:
                # Comprobar si ya terminamos todas las clases
                if all(c_val >= samples_per_class for c_val in counts.values()):
                    print("\n¡Objetivo alcanzado para todas las clases!")
                    break
                continue
                
            # Incrementar contador
            counts[class_name] += 1
            
            # Extraer y guardar imagen
            img_b64 = row.get("image_base64")
            if img_b64 is None:
                counts[class_name] -= 1
                continue
                
            try:
                img_data = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_data))
            except Exception as e:
                counts[class_name] -= 1
                continue
                
            # Normalizar modo a RGB si es necesario (y no es ya B/W)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
                
            file_name = f"{safe_name(class_name)}_{counts[class_name]:05d}.jpg"
            class_dir = output_dir / safe_name(class_name)
            file_path = class_dir / file_name
            
            img.save(file_path, "JPEG")
            
            # Guardar en CSV
            record = {
                "path": str(file_path.as_posix()),
                "class_name": class_name,
                "label": row.get("label", CLASSES.index(class_name))
            }
            records.append(record)
            writer.writerow(record)
            
            pbar.update(1)
            
    except KeyboardInterrupt:
        print("\nDescarga interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError durante la descarga: {e}")
    finally:
        csv_file.close()
        pbar.close()
        
    print("\nResumen final:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")
    
    print(f"\nMetadatos guardados en: {labels_path}")

def main():
    parser = argparse.ArgumentParser(description="Descarga masiva de RVL-CDIP (Streaming)")
    parser.add_argument("--output", default="dataset_completo", help="Carpeta de salida.")
    parser.add_argument("--samples-per-class", type=int, default=5000, help="Imágenes por clase (5000 = ~6 GB total)")
    args = parser.parse_args()

    download_dataset(args.output, args.samples_per_class)

if __name__ == "__main__":
    main()

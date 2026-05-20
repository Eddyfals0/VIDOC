import argparse
import csv
import os
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm

CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def safe_name(value):
    import re
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()

def extract_from_parquet(parquet_files, output_dir, samples_per_class):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {c: 0 for c in CLASSES}
    
    for c in CLASSES:
        (output_dir / safe_name(c)).mkdir(parents=True, exist_ok=True)
        
    records = []
    labels_path = output_dir / "labels_local.csv"
    fieldnames = ["path", "class_name", "label"]
    
    csv_file = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    print(f"Extrayendo imágenes desde archivos locales. Objetivo: {samples_per_class} por clase.")
    
    for parquet_file in parquet_files:
        print(f"\nProcesando {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            class_name = row.get("class_name", "")
            if class_name not in CLASSES:
                label = row.get("label", -1)
                if 0 <= label < len(CLASSES):
                    class_name = CLASSES[label]
                else:
                    continue
                    
            if counts[class_name] >= samples_per_class:
                if all(c_val >= samples_per_class for c_val in counts.values()):
                    print("\n¡Objetivo alcanzado para todas las clases!")
                    return
                continue
                
            img_b64 = row.get("image_base64")
            if not img_b64:
                continue
                
            try:
                img_data = base64.b64decode(img_b64)
                img = Image.open(BytesIO(img_data))
                
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                    
                counts[class_name] += 1
                
                file_name = f"{safe_name(class_name)}_{counts[class_name]:05d}.jpg"
                class_dir = output_dir / safe_name(class_name)
                file_path = class_dir / file_name
                
                img.save(file_path, "JPEG")
                
                record = {
                    "path": str(file_path.as_posix()),
                    "class_name": class_name,
                    "label": row.get("label", CLASSES.index(class_name))
                }
                records.append(record)
                writer.writerow(record)
                
            except Exception as e:
                continue
                
    csv_file.close()
    
    print("\nResumen final:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Extraer dataset desde archivos parquet locales")
    parser.add_argument("parquet_files", nargs='+', help="Ruta a los archivos .parquet descargados")
    parser.add_argument("--output", default="dataset_completo", help="Carpeta de salida.")
    parser.add_argument("--samples-per-class", type=int, default=5000, help="Imágenes por clase")
    args = parser.parse_args()

    extract_from_parquet(args.parquet_files, args.output, args.samples_per_class)

if __name__ == "__main__":
    main()

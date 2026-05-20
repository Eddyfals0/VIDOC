import os
import urllib.request
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import re

URL_TEST_PARQUET = "https://huggingface.co/datasets/sabaridsnfuji/rvl-cdip-filtered/resolve/main/test.parquet?download=true"
PARQUET_FILE = "test.parquet"

CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific report", "scientific publication", "specification",
    "file folder", "news article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    print(f"Descargando {output_path} (aprox. 1.3 GB) desde Hugging Face...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print("¡Descarga completada!\n")

def extract_images(parquet_file, output_dir="dataset_completo", samples_per_class=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {c: 0 for c in CLASSES}
    for c in CLASSES:
        (output_dir / safe_name(c)).mkdir(parents=True, exist_ok=True)
        
    labels_path = output_dir / "labels_local.csv"
    fieldnames = ["path", "class_name", "label"]
    
    csv_file = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    
    print(f"Extrayendo imágenes desde {parquet_file}. Objetivo: {samples_per_class} por clase.")
    
    df = pd.read_parquet(parquet_file)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
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
                break
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
            writer.writerow(record)
            
        except Exception as e:
            continue
            
    csv_file.close()
    print("\nResumen final de imágenes extraídas:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")

if __name__ == "__main__":
    if not os.path.exists(PARQUET_FILE):
        download_file(URL_TEST_PARQUET, PARQUET_FILE)
    else:
        print(f"El archivo {PARQUET_FILE} ya existe, omitiendo descarga.\n")
        
    extract_images(PARQUET_FILE, samples_per_class=500)

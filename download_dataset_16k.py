"""
Descarga 1000 imágenes por clase del dataset RVL-CDIP (16 clases = 16,000 imágenes).
Fuente: chainyo/rvl-cdip (formato Parquet nativo en HuggingFace).

Uso:
    python download_dataset_16k.py
    python download_dataset_16k.py --output dataset --samples-per-class 1000
"""

import argparse
import csv
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Las 16 clases de RVL-CDIP (orden por label index 0-15)
CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]


def safe_name(value):
    """Convierte nombre de clase a formato seguro para directorios."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def download_dataset(output_dir, samples_per_class, split="test"):
    """
    Descarga imágenes del dataset RVL-CDIP usando streaming.
    
    Args:
        output_dir: Carpeta de salida
        samples_per_class: Número de imágenes por clase
        split: Split del dataset ('train', 'test', 'validation')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Crear directorios por clase
    for c in CLASSES:
        (output_dir / safe_name(c)).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    total_target = samples_per_class * len(CLASSES)

    # Preparar CSV de metadatos
    labels_path = output_dir / "labels.csv"
    fieldnames = ["path", "class_name", "label"]
    csv_file = open(labels_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    print("=" * 60)
    print(f"  DESCARGA DE DATASET RVL-CDIP")
    print(f"  Fuente: chainyo/rvl-cdip (split={split})")
    print(f"  Objetivo: {samples_per_class} imágenes × {len(CLASSES)} clases = {total_target}")
    print(f"  Salida: {output_dir.resolve()}")
    print("=" * 60)
    print()

    # Cargar dataset en modo streaming
    print("Conectando a HuggingFace (esto puede tardar unos segundos)...")
    dataset = load_dataset("chainyo/rvl-cdip", split=split, streaming=True)
    print("¡Conectado! Iniciando descarga...\n")

    pbar = tqdm(total=total_target, desc="Descargando", unit="img")

    try:
        for row in dataset:
            label = row.get("label", -1)
            if label < 0 or label >= len(CLASSES):
                continue

            class_name = CLASSES[label]

            if counts[class_name] >= samples_per_class:
                if all(c >= samples_per_class for c in counts.values()):
                    print("\n¡Objetivo alcanzado para todas las clases!")
                    break
                continue

            # Obtener imagen PIL directamente
            img = row.get("image")
            if img is None:
                continue

            try:
                # Convertir a RGB si es necesario
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                counts[class_name] += 1
                sname = safe_name(class_name)
                file_name = f"{sname}_{counts[class_name]:05d}.jpg"
                class_dir = output_dir / sname
                file_path = class_dir / file_name

                img.save(file_path, "JPEG", quality=95)

                record = {
                    "path": str(file_path.as_posix()),
                    "class_name": class_name,
                    "label": label,
                }
                writer.writerow(record)
                pbar.update(1)

            except Exception:
                counts[class_name] -= 1
                continue

    except KeyboardInterrupt:
        print("\n\nDescarga interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError durante la descarga: {e}")
    finally:
        csv_file.close()
        pbar.close()

    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN DE DESCARGA")
    print("=" * 60)
    total = 0
    for class_name in CLASSES:
        count = counts[class_name]
        total += count
        status = "✓" if count >= samples_per_class else "✗"
        print(f"  {status} {class_name:25s}: {count:5d} / {samples_per_class}")
    print(f"\n  Total: {total} imágenes")
    print(f"  CSV:   {labels_path.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Descarga 1000 imágenes por clase de RVL-CDIP (16 clases)"
    )
    parser.add_argument("--output", default="dataset", help="Carpeta de salida (default: dataset)")
    parser.add_argument("--samples-per-class", type=int, default=1000,
                        help="Imágenes por clase (default: 1000)")
    parser.add_argument("--split", default="test",
                        help="Split del dataset: train, test, validation (default: test)")
    args = parser.parse_args()

    download_dataset(args.output, args.samples_per_class, args.split)


if __name__ == "__main__":
    main()

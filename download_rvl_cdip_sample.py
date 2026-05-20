import argparse
import base64
import csv
import re
import time
from pathlib import Path

import requests


DATASET = "sabaridsnfuji/rvl-cdip-filtered"
API_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_CLASSES = ("letter", "form", "email", "resume")


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def image_extension(row):
    fmt = str(row.get("format") or "").lower()
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt in {"png", "tif", "tiff", "bmp", "gif"}:
        return "tif" if fmt == "tiff" else fmt
    path_ext = Path(str(row.get("image_path") or "")).suffix.lower().lstrip(".")
    return path_ext or "tif"


def fetch_rows(split, offset, length):
    params = {
        "dataset": DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    }
    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("rows", [])


def save_sample(row, output_dir, index_by_class):
    data = row["row"]
    class_name = str(data["class_name"])
    class_dir = output_dir / safe_name(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)

    index_by_class[class_name] += 1
    ext = image_extension(data)
    file_name = f"{safe_name(class_name)}_{index_by_class[class_name]:03d}.{ext}"
    file_path = class_dir / file_name

    image_bytes = base64.b64decode(data["image_base64"])
    file_path.write_bytes(image_bytes)

    return {
        "path": str(file_path.as_posix()),
        "class_name": class_name,
        "label": data["label"],
        "source_image_path": data["image_path"],
        "width": data.get("width", ""),
        "height": data.get("height", ""),
        "format": data.get("format", ext),
        "file_size_bytes": data.get("file_size_bytes", len(image_bytes)),
        "split": data.get("split", ""),
    }


def write_labels(output_dir, records):
    labels_path = output_dir / "labels.csv"
    fieldnames = [
        "path",
        "class_name",
        "label",
        "source_image_path",
        "width",
        "height",
        "format",
        "file_size_bytes",
        "split",
    ]
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def download_sample(output_dir, samples_per_class, split, classes, chunk_size, max_rows, sleep):
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(classes)
    counts = {class_name: 0 for class_name in classes}
    index_by_class = {class_name: 0 for class_name in classes}
    records = []

    offset = 0
    while offset < max_rows and any(count < samples_per_class for count in counts.values()):
        rows = fetch_rows(split=split, offset=offset, length=chunk_size)
        if not rows:
            break

        for wrapped_row in rows:
            data = wrapped_row["row"]
            class_name = str(data.get("class_name", ""))
            if class_name not in wanted or counts[class_name] >= samples_per_class:
                continue

            record = save_sample(wrapped_row, output_dir, index_by_class)
            records.append(record)
            counts[class_name] += 1
            print(f"Guardado {record['path']} ({counts[class_name]}/{samples_per_class})")

            if all(count >= samples_per_class for count in counts.values()):
                break

        offset += len(rows)
        if sleep:
            time.sleep(sleep)

    write_labels(output_dir, records)
    print("\nResumen:")
    for class_name, count in counts.items():
        print(f"- {class_name}: {count}")
    print(f"\nCSV: {output_dir / 'labels.csv'}")

    missing = [class_name for class_name, count in counts.items() if count < samples_per_class]
    if missing:
        raise SystemExit(f"No se completo la muestra para: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Descarga una muestra pequena de RVL-CDIP filtrado desde Hugging Face."
    )
    parser.add_argument("--output", default="dataset_simple", help="Carpeta de salida.")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--split", default="train")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    download_sample(
        output_dir=Path(args.output),
        samples_per_class=args.samples_per_class,
        split=args.split,
        classes=args.classes,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()

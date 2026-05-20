import os
import pandas as pd
from pathlib import Path

# Lista de clases oficial de chainyo/rvl-cdip (orden alfabético en HuggingFace)
HF_CLASSES = [
    "advertisement", "budget", "email", "file_folder", "form", 
    "handwritten", "invoice", "letter", "memo", "news_article", 
    "presentation", "questionnaire", "resume", "scientific_publication", 
    "scientific_report", "specification"
]

# Lista de clases que usó el script original
ORIGINAL_CLASSES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice",
    "presentation", "questionnaire", "resume", "memo"
]

def safe_name(c):
    return c.replace(" ", "_").lower()

def main():
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("Error: La carpeta 'dataset' no existe.")
        return

    print("=" * 60)
    print("  REORGANIZACIÓN Y CORRECCIÓN DEL DATASET")
    print("=" * 60)
    
    # 1. Crear un mapeo temporal para renombrar sin colisiones
    # Mapea el nombre de la carpeta errónea actual al nombre correcto real de su contenido
    mapping = {}
    for label_idx, orig_class in enumerate(ORIGINAL_CLASSES):
        real_class = HF_CLASSES[label_idx]
        mapping[safe_name(orig_class)] = safe_name(real_class)
        
    print("\nMapeo de corrección (Carpeta actual -> Contenido real):")
    for k, v in mapping.items():
        print(f"  {k:25s} -> {v:25s}")

    # Renombrar carpetas usando nombres temporales para evitar colisiones
    temp_mapping = {}
    print("\nRenombrando carpetas a nombres temporales...")
    for orig_folder, real_folder in mapping.items():
        src = dataset_dir / orig_folder
        if src.exists():
            temp_name = f"temp_{orig_folder}"
            src.rename(dataset_dir / temp_name)
            temp_mapping[temp_name] = real_folder
            print(f"  {orig_folder} -> {temp_name}")

    # Renombrar de temporales a nombres reales definitivos
    print("\nAplicando nombres reales definitivos...")
    for temp_name, real_folder in temp_mapping.items():
        src = dataset_dir / temp_name
        dest = dataset_dir / real_folder
        
        # Si ya existe por algún motivo, fusionamos o sobreescribimos
        if dest.exists() and dest != src:
            # Mover todos los archivos dentro
            for f in src.glob("*"):
                f.rename(dest / f.name)
            src.rmdir()
            print(f"  FUSIONADO: {temp_name} -> {real_folder}")
        else:
            src.rename(dest)
            print(f"  {temp_name} -> {real_folder}")

    # Crear las carpetas para las clases reales que ahora quedarán vacías
    # (scientific_report y specification son las que realmente nos faltan ahora)
    print("\nCreando carpetas vacías para las clases realmente faltantes...")
    for c in HF_CLASSES:
        folder = dataset_dir / safe_name(c)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"  Creada: {folder}")

    # 2. Corregir el archivo labels.csv
    labels_path = dataset_dir / "labels.csv"
    if labels_path.exists():
        print("\nCorrigiendo archivo labels.csv...")
        df = pd.read_csv(labels_path)
        
        # Corregir cada fila según su etiqueta real de HuggingFace
        corrected_records = []
        for _, row in df.iterrows():
            label = int(row["label"])
            # La etiqueta real de Hugging Face
            real_class = HF_CLASSES[label]
            real_folder = safe_name(real_class)
            
            # Reconstruir la ruta correcta del archivo
            old_path = Path(row["path"])
            file_name = old_path.name
            
            # Cambiar el nombre del archivo si es necesario para que coincida con su nueva clase
            # Ejemplo: letter_00001.jpg -> advertisement_00001.jpg
            new_file_name = f"{real_folder}_{file_name.split('_')[-1]}"
            new_path = dataset_dir / real_folder / new_file_name
            
            # Renombrar físicamente el archivo de imagen
            current_path = dataset_dir / real_folder / file_name
            if current_path.exists():
                current_path.rename(new_path)
                
            corrected_records.append({
                "path": str(new_path.as_posix()),
                "class_name": real_class,
                "label": label
            })
            
        new_df = pd.DataFrame(corrected_records)
        new_df.to_csv(labels_path, index=False)
        print("  labels.csv corregido y guardado exitosamente.")
    else:
        print("\nAdvertencia: No se encontró labels.csv para corregir.")

    print("\n" + "=" * 60)
    print("  ¡REORGANIZACIÓN COMPLETADA!")
    print("=" * 60)

if __name__ == "__main__":
    main()

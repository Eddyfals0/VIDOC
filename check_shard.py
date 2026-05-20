import urllib.request
import pandas as pd
from pathlib import Path

url = "https://huggingface.co/datasets/chainyo/rvl-cdip/resolve/main/data/test-00000-of-00015.parquet"
output = "test_shard.parquet"

print("Descargando shard de prueba (180 MB)...")
try:
    urllib.request.urlretrieve(url, output)
    print("¡Descarga completada!")
    df = pd.read_parquet(output)
    print("Columnas:", df.columns)
    print("Tamaño:", df.shape)
    print("Distribución de etiquetas (label):")
    print(df["label"].value_counts().sort_index())
except Exception as e:
    print("Error:", e)

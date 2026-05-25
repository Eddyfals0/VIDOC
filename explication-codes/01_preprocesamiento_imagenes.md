# Explicación Detallada: `01_preprocesamiento_imagenes.py`

Este módulo constituye la primera etapa del pipeline del proyecto final. Su propósito principal es **limpiar, homogeneizar y preparar** las imágenes de documentos escaneados para su consumo posterior en dos ramas distintas:
1.  **Rama de Deep Learning (CNN):** Requiere imágenes pequeñas de **128x128 píxeles** para optimizar el entrenamiento y reducir la carga computacional.
2.  **Rama de Clasificación Visual Clásica (LBP / GLCM):** Requiere una resolución mayor de **256x256 píxeles** para no perder texturas finas ni detalles estructurales del documento.

---

## 1. Importaciones y Configuración del Sistema

El script comienza importando las librerías necesarias para el procesamiento numérico y de imágenes:

```python
import os
import sys
import argparse
import glob
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar figuras sin mostrar en pantalla
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
```

### Elementos Clave:
*   **`cv2` (OpenCV):** La biblioteca principal utilizada para todas las tareas de lectura, redimensionamiento, conversión de color y ecualización adaptativa de las imágenes.
*   **`matplotlib.use('Agg')`:** Configura Matplotlib para ejecutarse sin un entorno gráfico interactivo (servidor X o ventana de Windows). Esto es fundamental para entornos de ejecución automatizados o ejecuciones en la consola de comandos, permitiendo renderizar gráficos directamente a archivos de disco.
*   **`project_config`:** Modulo local que unifica la lectura de clases directamente desde el dataset de entrada para mantener la consistencia en todo el pipeline:
    ```python
    from project_config import classes_from_dataset
    CLASES = classes_from_dataset("dataset")
    ```

---

## 2. Constantes y Parámetros del Pipeline

Se definen las resoluciones objetivo y el directorio donde se almacenarán los resultados visuales del análisis:

```python
# Resoluciones objetivo para el pipeline dual
RESOLUCIONES = {
    "128x128": (128, 128),  # Para Redes Neuronales Convolucionales (CNN)
    "256x256": (256, 256),  # Para descriptores de textura clásicos (LBP, GLCM)
}

DIR_RESULTADOS = "resultados"
```

---

## 3. Función Central: `preprocesar_imagen`

Esta función contiene el **núcleo matemático y algorítmico** del preprocesamiento de cada documento. Aplica un pipeline secuencial de 4 pasos fundamentales:

```python
def preprocesar_imagen(ruta_imagen: str, tamano: tuple) -> np.ndarray | None:
```

### Paso 1: Lectura Física de Imagen
```python
imagen = cv2.imread(ruta_imagen)
```
Lee la imagen desde disco. Si el archivo está corrupto o no se encuentra en la ruta especificada, la función retorna `None` de forma segura.

### Paso 2: Conversión a Escala de Grises
```python
if len(imagen.shape) == 3:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
else:
    gris = imagen
```
Los documentos escaneados a menudo contienen colores de fondo o logotipos que no aportan información relevante para distinguir la estructura del tipo de documento. Convertir a escala de grises reduce los canales de color de 3 (BGR) a 1 (Luminancia), acelerando el procesamiento y enfocando la extracción en el contraste textual y estructural.

### Paso 3: Redimensionamiento Inteligente
```python
if gris.shape[0] > tamano[1] and gris.shape[1] > tamano[0]:
    interpolacion = cv2.INTER_AREA
else:
    interpolacion = cv2.INTER_LINEAR
redimensionada = cv2.resize(gris, tamano, interpolation=interpolacion)
```
El cambio de escala de una imagen requiere interpolar valores de píxeles. La selección del algoritmo de interpolación es crítica para la calidad:
*   **`cv2.INTER_AREA` (Interpolación por relación de área):** Se selecciona cuando reducimos la escala de la imagen (downsampling). Evita el aliasing (efecto de dientes de sierra) y preserva los bordes del texto fino al promediar los píxeles vecinos del área original.
*   **`cv2.INTER_LINEAR` (Interpolación bilineal):** Se utiliza en caso de tener que ampliar una imagen pequeña (upsampling), ofreciendo una transición suave entre píxeles.

### Paso 4: Ecualización de Histograma Adaptativa (CLAHE)
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
ecualizada = clahe.apply(redimensionada)
```

En la digitalización de documentos, es muy común encontrar iluminación no uniforme (por ejemplo, sombras generadas al tomar fotos con celulares o escaneos con bordes oscuros).
*   **¿Por qué no usar ecualización de histograma global?** La ecualización global incrementa el contraste general utilizando la distribución de la imagen completa, lo que suele sobreexponer zonas claras del documento y perder el texto en regiones sombreadas.
*   **¿Cómo funciona CLAHE (Contrast Limited Adaptive Histogram Equalization)?**
    *   Divide la imagen en una cuadrícula de bloques rectangulares llamados sub-regiones o azulejos (configurado mediante `tileGridSize=(8, 8)`).
    *   Ecualiza el histograma de cada bloque de forma independiente.
    *   **Límite de Contraste (`clipLimit=2.0`):** Si hay mucho ruido en algún bloque, su histograma local presentará picos muy altos. El parámetro `clipLimit` recorta estos picos y redistribuye los píxeles uniformemente antes de aplicar la ecualización, controlando la amplificación innecesaria del ruido de fondo.
    *   Finalmente, utiliza una interpolación bilineal para suavizar las transiciones entre las fronteras de los bloques y eliminar artefactos de cuadrícula.

---

## 4. Gestión Estructurada de Datos

### Creación de Directorios Dinámicos (`crear_directorios_salida`)
El pipeline organiza los archivos preprocesados automáticamente en carpetas separadas por resolución y clase para simplificar la lectura en los siguientes scripts:

```python
def crear_directorios_salida(dir_salida: str) -> dict:
    rutas = {}
    for nombre_res, _ in RESOLUCIONES.items():
        rutas[nombre_res] = {}
        for clase in CLASES:
            ruta = os.path.join(dir_salida, nombre_res, clase)
            os.makedirs(ruta, exist_ok=True)
            rutas[nombre_res][clase] = ruta
    return rutas
```
Genera la siguiente estructura física:
```
dataset_preprocessed/
├── 128x128/
│   ├── advertisement/
│   ├── budget/
│   └── ...
└── 256x256/
    ├── advertisement/
    ├── budget/
    └── ...
```

### Escaneo y Clasificación de Orígenes (`obtener_imagenes_por_clase`)
Utiliza patrones de búsqueda glob para identificar todas las imágenes compatibles (`.jpg`, `.png`, etc.) agrupadas por su carpeta contenedora (que define la clase del documento), arrojando un reporte estadístico detallado por consola.

---

## 5. Orquestación del Pipeline: `ejecutar_preprocesamiento`

La función principal integra las etapas anteriores en un ciclo de procesamiento masivo:

```python
def ejecutar_preprocesamiento(dir_entrada: str, dir_salida: str) -> None:
```

### Características de Robustez Implementadas:
1.  **Validación de Origen:** Termina la ejecución de manera limpia si el dataset no existe o está vacío.
2.  **Mecanismo de Caching (Skip de Existentes):**
    ```python
    if os.path.exists(ruta_destino):
        total_procesadas += 1
        continue
    ```
    Si el script es interrumpido o se ejecuta de nuevo tras agregar nuevas imágenes, no vuelve a procesar los archivos que ya existen en el directorio de destino. Esto ahorra horas de computación en datasets grandes.
3.  **Monitoreo visual del progreso:** Se integra `tqdm` para mostrar barras de progreso en tiempo real por cada clase y resolución procesada.
4.  **Generación de Reportes Comparativos:** Al finalizar, crea una gráfica de visualización comparativa de muestras en `resultados/01_muestras_preprocesamiento.png` que permite inspeccionar la calidad del preprocesamiento visual de manera inmediata.

---

## 6. Interfaz de Línea de Comandos (CLI)

El script finaliza proporcionando un módulo ejecutable mediante argumentos de consola usando `argparse`:

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--input", "-i", type=str, default="dataset", ...)
    parser.add_argument("--output", "-o", type=str, default="dataset_preprocessed", ...)
    ...
```

Esto permite parametrizar las carpetas de origen y destino desde un script orquestador global o terminal externa de forma limpia:

```bash
python 01_preprocesamiento_imagenes.py --input dataset --output dataset_preprocessed
```

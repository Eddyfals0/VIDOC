#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
01_preprocesamiento_imagenes.py
==============================================================================
Proyecto Final - Recuperación de Información (8vo Semestre)
Clasificador de Documentos por OCR y Características Visuales

Descripción:
    Preprocesa imágenes de documentos para el pipeline de clasificación visual.
    - Lee imágenes originales del dataset organizado por clases.
    - Redimensiona a 128x128 (para CNN) y 256x256 (para descriptores visuales).
    - Convierte a escala de grises.
    - Aplica ecualización de histograma (CLAHE) para mejorar contraste.
    - Guarda las imágenes preprocesadas en directorios organizados por resolución.
    - Genera una visualización de muestras del preprocesamiento.

Uso:
    python 01_preprocesamiento_imagenes.py --input dataset --output dataset_preprocessed

Autor: Estudiante RI
Fecha: Mayo 2026
==============================================================================
"""

import os
import sys
import argparse
import glob
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar figuras sin mostrar
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


# =============================================================================
# Constantes del proyecto
# =============================================================================

# Las 16 clases del dataset de documentos
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

# Resoluciones objetivo
RESOLUCIONES = {
    "128x128": (128, 128),  # Para redes neuronales convolucionales (CNN)
    "256x256": (256, 256),  # Para descriptores visuales (LBP, GLCM, etc.)
}

# Directorio de resultados para visualizaciones
DIR_RESULTADOS = "resultados"


# =============================================================================
# Funciones de preprocesamiento
# =============================================================================

def preprocesar_imagen(ruta_imagen: str, tamano: tuple) -> np.ndarray | None:
    """
    Preprocesa una imagen de documento aplicando el pipeline completo.

    Pipeline:
        1. Lectura de la imagen original
        2. Conversión a escala de grises
        3. Redimensionamiento al tamaño objetivo
        4. Ecualización de histograma adaptativa (CLAHE)

    Args:
        ruta_imagen (str): Ruta absoluta o relativa a la imagen original.
        tamano (tuple): Tupla (ancho, alto) con el tamaño de salida deseado.

    Returns:
        np.ndarray | None: Imagen preprocesada como array NumPy, o None si falla.
    """
    try:
        # Paso 1: Leer la imagen desde disco
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"  [ADVERTENCIA] No se pudo leer la imagen: {ruta_imagen}")
            return None

        # Paso 2: Convertir a escala de grises (si no lo está ya)
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen

        # Paso 3: Redimensionar a la resolución objetivo
        # Se usa INTER_AREA para reducción (mejor calidad) e INTER_LINEAR para ampliación
        if gris.shape[0] > tamano[1] and gris.shape[1] > tamano[0]:
            interpolacion = cv2.INTER_AREA
        else:
            interpolacion = cv2.INTER_LINEAR
        redimensionada = cv2.resize(gris, tamano, interpolation=interpolacion)

        # Paso 4: Ecualización de histograma adaptativa (CLAHE)
        # CLAHE divide la imagen en bloques y ecualiza cada uno de forma local,
        # lo que produce un mejor contraste que la ecualización global, especialmente
        # en documentos con iluminación no uniforme.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ecualizada = clahe.apply(redimensionada)

        return ecualizada

    except Exception as e:
        print(f"  [ERROR] Fallo al procesar {ruta_imagen}: {e}")
        return None


def crear_directorios_salida(dir_salida: str) -> dict:
    """
    Crea la estructura de directorios de salida para todas las resoluciones y clases.

    Estructura creada:
        dir_salida/
        ├── 128x128/
        │   ├── letter/
        │   ├── form/
        │   └── ... (16 clases)
        └── 256x256/
            ├── letter/
            ├── form/
            └── ... (16 clases)

    Args:
        dir_salida (str): Directorio raíz de salida.

    Returns:
        dict: Diccionario con rutas creadas {resolución: {clase: ruta}}.
    """
    rutas = {}
    for nombre_res, _ in RESOLUCIONES.items():
        rutas[nombre_res] = {}
        for clase in CLASES:
            ruta = os.path.join(dir_salida, nombre_res, clase)
            os.makedirs(ruta, exist_ok=True)
            rutas[nombre_res][clase] = ruta
    return rutas


def obtener_imagenes_por_clase(dir_entrada: str) -> dict:
    """
    Escanea el directorio de entrada y recopila las rutas de imágenes por clase.

    Args:
        dir_entrada (str): Directorio raíz del dataset (ej: 'dataset/').

    Returns:
        dict: Diccionario {nombre_clase: [lista_de_rutas_de_imágenes]}.
    """
    imagenes_por_clase = {}
    total_imagenes = 0

    for clase in CLASES:
        dir_clase = os.path.join(dir_entrada, clase)

        if not os.path.isdir(dir_clase):
            print(f"  [ADVERTENCIA] Directorio de clase no encontrado: {dir_clase}")
            imagenes_por_clase[clase] = []
            continue

        # Buscar archivos de imagen con extensiones comunes
        extensiones = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
        rutas = []
        for ext in extensiones:
            rutas.extend(glob.glob(os.path.join(dir_clase, ext)))

        # Ordenar para reproducibilidad
        rutas.sort()
        imagenes_por_clase[clase] = rutas
        total_imagenes += len(rutas)

    print(f"\n{'='*60}")
    print(f"  Resumen del dataset de entrada")
    print(f"{'='*60}")
    for clase, rutas in imagenes_por_clase.items():
        print(f"  {clase:30s} -> {len(rutas):5d} imágenes")
    print(f"{'='*60}")
    print(f"  {'TOTAL':30s} -> {total_imagenes:5d} imágenes")
    print(f"{'='*60}\n")

    return imagenes_por_clase


# =============================================================================
# Visualización de muestras
# =============================================================================

def generar_visualizacion_muestras(
    dir_entrada: str,
    dir_salida: str,
    ruta_visualizacion: str,
    n_muestras: int = 4
) -> None:
    """
    Genera una grilla de visualización comparando imágenes originales vs preprocesadas.

    Muestra n_muestras imágenes aleatorias con sus versiones:
    original, 128x128, y 256x256 tras el preprocesamiento.

    Args:
        dir_entrada (str): Directorio del dataset original.
        dir_salida (str): Directorio del dataset preprocesado.
        ruta_visualizacion (str): Ruta donde guardar la imagen de visualización.
        n_muestras (int): Número de muestras a mostrar (default: 4).
    """
    print("\n[INFO] Generando visualización de muestras de preprocesamiento...")

    # Recolectar muestras aleatorias de distintas clases
    muestras = []
    clases_disponibles = list(CLASES)
    random.shuffle(clases_disponibles)

    for clase in clases_disponibles:
        dir_clase_original = os.path.join(dir_entrada, clase)
        if not os.path.isdir(dir_clase_original):
            continue

        archivos = glob.glob(os.path.join(dir_clase_original, "*.jpg"))
        if not archivos:
            continue

        archivo = random.choice(archivos)
        nombre_archivo = os.path.basename(archivo)

        # Buscar las versiones preprocesadas correspondientes
        ruta_128 = os.path.join(dir_salida, "128x128", clase, nombre_archivo)
        ruta_256 = os.path.join(dir_salida, "256x256", clase, nombre_archivo)

        if os.path.exists(ruta_128) and os.path.exists(ruta_256):
            muestras.append({
                "clase": clase,
                "original": archivo,
                "128x128": ruta_128,
                "256x256": ruta_256,
            })

        if len(muestras) >= n_muestras:
            break

    if not muestras:
        print("  [ADVERTENCIA] No se encontraron muestras para visualizar.")
        return

    # Crear la figura con la grilla de comparación
    n = len(muestras)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(
        "Preprocesamiento de Imágenes de Documentos\n"
        "Original → 128×128 (CNN) → 256×256 (Descriptores)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Ajustar para el caso de una sola muestra
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, muestra in enumerate(muestras):
        # Imagen original
        img_orig = cv2.imread(muestra["original"], cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(img_orig, cmap="gray")
        axes[i, 0].set_title(
            f'{muestra["clase"]}\nOriginal ({img_orig.shape[1]}×{img_orig.shape[0]})',
            fontsize=9
        )
        axes[i, 0].axis("off")

        # Imagen 128x128
        img_128 = cv2.imread(muestra["128x128"], cv2.IMREAD_GRAYSCALE)
        axes[i, 1].imshow(img_128, cmap="gray")
        axes[i, 1].set_title("128×128 (CNN)", fontsize=9)
        axes[i, 1].axis("off")

        # Imagen 256x256
        img_256 = cv2.imread(muestra["256x256"], cv2.IMREAD_GRAYSCALE)
        axes[i, 2].imshow(img_256, cmap="gray")
        axes[i, 2].set_title("256×256 (Descriptores)", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(ruta_visualizacion, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Visualización guardada en: {ruta_visualizacion}")


# =============================================================================
# Función principal de preprocesamiento
# =============================================================================

def ejecutar_preprocesamiento(dir_entrada: str, dir_salida: str) -> None:
    """
    Ejecuta el pipeline completo de preprocesamiento sobre el dataset.

    Args:
        dir_entrada (str): Directorio raíz del dataset original.
        dir_salida (str): Directorio raíz donde se guardarán las imágenes preprocesadas.
    """
    # Validar que el directorio de entrada existe
    if not os.path.isdir(dir_entrada):
        print(f"\n[ERROR] El directorio de entrada no existe: {dir_entrada}")
        print("  Asegúrate de que el dataset esté organizado como:")
        print("    dataset/{nombre_clase}/{nombre_clase}_00001.jpg")
        sys.exit(1)

    # Crear estructura de directorios de salida
    print("\n[INFO] Creando estructura de directorios de salida...")
    rutas_salida = crear_directorios_salida(dir_salida)

    # Obtener listado de imágenes por clase
    imagenes_por_clase = obtener_imagenes_por_clase(dir_entrada)

    # Contadores de estadísticas
    total_procesadas = 0
    total_errores = 0

    # Procesar cada resolución
    for nombre_res, tamano in RESOLUCIONES.items():
        print(f"\n{'='*60}")
        print(f"  Procesando resolución: {nombre_res}")
        print(f"{'='*60}")

        for clase in CLASES:
            rutas_imgs = imagenes_por_clase.get(clase, [])
            if not rutas_imgs:
                continue

            dir_destino = rutas_salida[nombre_res][clase]

            # Barra de progreso por clase y resolución
            desc = f"  [{nombre_res}] {clase}"
            for ruta_img in tqdm(rutas_imgs, desc=desc, ncols=90):
                nombre_archivo = os.path.basename(ruta_img)
                ruta_destino = os.path.join(dir_destino, nombre_archivo)

                # Saltar si la imagen ya fue procesada (evitar reprocesamiento)
                if os.path.exists(ruta_destino):
                    total_procesadas += 1
                    continue

                # Aplicar pipeline de preprocesamiento
                img_procesada = preprocesar_imagen(ruta_img, tamano)

                if img_procesada is not None:
                    cv2.imwrite(ruta_destino, img_procesada)
                    total_procesadas += 1
                else:
                    total_errores += 1

    # Resumen final
    print(f"\n{'='*60}")
    print(f"  Preprocesamiento completado")
    print(f"{'='*60}")
    print(f"  Imágenes procesadas exitosamente : {total_procesadas}")
    print(f"  Errores encontrados              : {total_errores}")
    print(f"  Directorio de salida             : {dir_salida}")
    print(f"{'='*60}\n")

    # Generar visualización de muestras
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta_vis = os.path.join(DIR_RESULTADOS, "01_muestras_preprocesamiento.png")
    generar_visualizacion_muestras(dir_entrada, dir_salida, ruta_vis)


# =============================================================================
# Punto de entrada - CLI con argparse
# =============================================================================

def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesamiento de imágenes de documentos para clasificación visual. "
            "Redimensiona, convierte a escala de grises y aplica ecualización de histograma."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de uso:\n"
            "  python 01_preprocesamiento_imagenes.py --input dataset --output dataset_preprocessed\n\n"
            "Estructura esperada del dataset:\n"
            "  dataset/\n"
            "  ├── letter/\n"
            "  │   ├── letter_00001.jpg\n"
            "  │   └── ...\n"
            "  ├── form/\n"
            "  │   ├── form_00001.jpg\n"
            "  │   └── ...\n"
            "  └── ... (16 clases)"
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="dataset",
        help="Directorio raíz del dataset original (default: dataset)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset_preprocessed",
        help="Directorio raíz de salida para imágenes preprocesadas (default: dataset_preprocessed)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PREPROCESAMIENTO DE IMÁGENES DE DOCUMENTOS")
    print("  Proyecto Final - Recuperación de Información")
    print("=" * 60)
    print(f"  Entrada  : {os.path.abspath(args.input)}")
    print(f"  Salida   : {os.path.abspath(args.output)}")
    print(f"  Clases   : {len(CLASES)}")
    print(f"  Resoluciones: {', '.join(RESOLUCIONES.keys())}")
    print("=" * 60)

    ejecutar_preprocesamiento(args.input, args.output)

    print("[INFO] Script finalizado exitosamente.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==============================================================================
02_extraccion_features_visuales.py
==============================================================================
Proyecto Final - Recuperación de Información (8vo Semestre)
Clasificador de Documentos por OCR y Características Visuales

Descripción:
    Extrae características visuales de las imágenes preprocesadas (256×256)
    utilizando tres tipos de descriptores:

    1. LBP (Local Binary Patterns):
       - Captura texturas locales del documento.
       - radio=3, n_puntos=24, método='uniform'.
       - El histograma de valores LBP se usa como vector de características.

    2. GLCM (Gray-Level Co-occurrence Matrix):
       - Captura relaciones espaciales entre niveles de gris.
       - Propiedades: contraste, disimilaridad, homogeneidad, energía, correlación.
       - Distancias: [1, 2, 3], Ángulos: [0, π/4, π/2, 3π/4].

    3. Histogramas de Intensidad:
       - Distribución de 256 bins de los valores de gris.
       - Histograma normalizado.

    Guarda los vectores de características como arrays NumPy (.npy).

Uso:
    python 02_extraccion_features_visuales.py --input dataset_preprocessed/256x256 --output features

Autor: Estudiante RI
Fecha: Mayo 2026
==============================================================================
"""

import os
import sys
import argparse
import glob
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Descriptores de textura de scikit-image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Suprimir advertencias innecesarias de skimage
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Constantes del proyecto
# =============================================================================

# Las 16 clases del dataset de documentos
import os
from pathlib import Path

from project_config import classes_with_files

CLASES = classes_with_files("dataset_preprocessed/256x256")
CLASSES = CLASES

# Parámetros de LBP (Local Binary Patterns)
LBP_RADIUS = 3       # Radio del vecindario circular
LBP_N_POINTS = 24    # Número de puntos en el vecindario
LBP_METHOD = "uniform"  # Método uniforme (reduce bins y captura patrones fundamentales)
LBP_N_BINS = LBP_N_POINTS + 2  # Número de bins para histograma LBP uniforme

# Parámetros de GLCM (Gray-Level Co-occurrence Matrix)
GLCM_DISTANCES = [1, 2, 3]                           # Distancias de desplazamiento
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Ángulos: 0°, 45°, 90°, 135°
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

# Número de bins para histograma de intensidad
HIST_N_BINS = 256

# Directorio de resultados
DIR_RESULTADOS = "resultados"


# =============================================================================
# Funciones de extracción de características
# =============================================================================

def extraer_lbp(imagen: np.ndarray) -> np.ndarray:
    """
    Extrae el descriptor LBP (Local Binary Pattern) de una imagen en escala de grises.

    LBP compara cada píxel con sus vecinos en un radio circular. El patrón
    'uniform' reduce la dimensionalidad a (n_points + 2) bins, capturando
    los patrones de textura más relevantes.

    Args:
        imagen (np.ndarray): Imagen en escala de grises (2D array, dtype uint8).

    Returns:
        np.ndarray: Histograma normalizado de LBP con (n_points + 2) bins.
    """
    # Calcular el mapa de LBP para toda la imagen
    lbp_mapa = local_binary_pattern(
        imagen,
        P=LBP_N_POINTS,
        R=LBP_RADIUS,
        method=LBP_METHOD
    )

    # Construir histograma normalizado del mapa LBP
    # Para método 'uniform', los valores van de 0 a n_points + 1
    histograma, _ = np.histogram(
        lbp_mapa.ravel(),
        bins=LBP_N_BINS,
        range=(0, LBP_N_BINS),
        density=True  # Normalizar para que sea invariante al tamaño de la imagen
    )

    return histograma.astype(np.float32)


def extraer_glcm(imagen: np.ndarray) -> np.ndarray:
    """
    Extrae características de textura usando GLCM (Gray-Level Co-occurrence Matrix).

    La GLCM cuantifica cómo los pares de píxeles con ciertos valores de gris
    y relaciones espaciales ocurren en la imagen. Se extraen 5 propiedades
    estadísticas para cada combinación de distancia y ángulo.

    Dimensiones del vector resultante:
        5 propiedades × 3 distancias × 4 ángulos = 60 valores

    Args:
        imagen (np.ndarray): Imagen en escala de grises (2D array, dtype uint8).

    Returns:
        np.ndarray: Vector de características GLCM (60 dimensiones).
    """
    # Asegurar que la imagen sea uint8 para la GLCM
    if imagen.dtype != np.uint8:
        imagen = imagen.astype(np.uint8)

    # Calcular la matriz de co-ocurrencia de niveles de gris
    # Reducir a 64 niveles para hacer la GLCM más robusta y eficiente
    imagen_cuantizada = (imagen // 4).astype(np.uint8)  # 256 -> 64 niveles
    glcm = graycomatrix(
        imagen_cuantizada,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )

    # Extraer propiedades estadísticas de la GLCM
    features = []
    for prop in GLCM_PROPERTIES:
        valores = graycoprops(glcm, prop)  # Shape: (n_distances, n_angles)
        features.extend(valores.ravel().tolist())

    return np.array(features, dtype=np.float32)


def extraer_histograma_intensidad(imagen: np.ndarray) -> np.ndarray:
    """
    Extrae el histograma normalizado de intensidades de la imagen.

    Captura la distribución global de niveles de gris del documento,
    útil para distinguir tipos de documentos con diferentes densidades
    de tinta, fondos, etc.

    Args:
        imagen (np.ndarray): Imagen en escala de grises (2D array, dtype uint8).

    Returns:
        np.ndarray: Histograma normalizado de 256 bins.
    """
    # Calcular histograma de intensidad con 256 bins
    histograma, _ = np.histogram(
        imagen.ravel(),
        bins=HIST_N_BINS,
        range=(0, 256),
        density=True  # Normalizar
    )

    return histograma.astype(np.float32)


# =============================================================================
# Obtención de imágenes
# =============================================================================

def obtener_imagenes_por_clase(dir_entrada: str) -> dict:
    """
    Escanea el directorio de entrada y recopila rutas de imágenes preprocesadas
    organizadas por clase.

    Args:
        dir_entrada (str): Directorio raíz con imágenes preprocesadas (256x256).

    Returns:
        dict: Diccionario {nombre_clase: [lista_de_rutas]}.
    """
    imagenes_por_clase = {}
    total = 0

    for clase in CLASES:
        dir_clase = os.path.join(dir_entrada, clase)

        if not os.path.isdir(dir_clase):
            print(f"  [ADVERTENCIA] Directorio no encontrado: {dir_clase}")
            imagenes_por_clase[clase] = []
            continue

        # Buscar imágenes con extensiones comunes
        extensiones = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
        rutas = []
        for ext in extensiones:
            rutas.extend(glob.glob(os.path.join(dir_clase, ext)))

        rutas.sort()  # Ordenar para reproducibilidad
        imagenes_por_clase[clase] = rutas
        total += len(rutas)

    print(f"\n{'='*60}")
    print(f"  Imágenes preprocesadas encontradas")
    print(f"{'='*60}")
    for clase, rutas in imagenes_por_clase.items():
        print(f"  {clase:30s} -> {len(rutas):5d} imágenes")
    print(f"{'='*60}")
    print(f"  {'TOTAL':30s} -> {total:5d} imágenes")
    print(f"{'='*60}\n")

    return imagenes_por_clase


# =============================================================================
# Visualización: Histogramas LBP por categoría
# =============================================================================

def generar_visualizacion_lbp(
    features_lbp: np.ndarray,
    labels: np.ndarray,
    nombres_clases: list,
    ruta_salida: str
) -> None:
    """
    Genera una visualización con los histogramas LBP promedio por categoría.

    Cada subplot muestra el histograma LBP promediado sobre todas las imágenes
    de una clase, permitiendo comparar los patrones de textura entre documentos.

    Args:
        features_lbp (np.ndarray): Matriz de features LBP (n_muestras, n_bins).
        labels (np.ndarray): Array de etiquetas numéricas (n_muestras,).
        nombres_clases (list): Lista de nombres de clases.
        ruta_salida (str): Ruta donde guardar la visualización.
    """
    print("\n[INFO] Generando visualización de histogramas LBP por categoría...")

    n_clases = len(nombres_clases)
    n_cols = 4
    n_filas = (n_clases + n_cols - 1) // n_cols  # Redondeo hacia arriba

    fig, axes = plt.subplots(n_filas, n_cols, figsize=(16, 3.5 * n_filas))
    fig.suptitle(
        "Histogramas LBP Promedio por Categoría de Documento\n"
        f"(radio={LBP_RADIUS}, puntos={LBP_N_POINTS}, método='{LBP_METHOD}')",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Paleta de colores para las clases
    colores = plt.cm.tab20(np.linspace(0, 1, n_clases))

    axes_flat = axes.flatten()

    for idx, clase_nombre in enumerate(nombres_clases):
        ax = axes_flat[idx]

        # Filtrar features de esta clase
        mascara = labels == idx
        if not np.any(mascara):
            ax.set_title(f"{clase_nombre}\n(sin datos)", fontsize=9)
            ax.axis("off")
            continue

        # Calcular histograma LBP promedio para esta clase
        lbp_promedio = features_lbp[mascara].mean(axis=0)
        lbp_std = features_lbp[mascara].std(axis=0)

        # Graficar el histograma con barras y banda de desviación estándar
        bins = np.arange(LBP_N_BINS)
        ax.bar(bins, lbp_promedio, color=colores[idx], alpha=0.7, width=0.8)
        ax.fill_between(
            bins,
            np.maximum(lbp_promedio - lbp_std, 0),
            lbp_promedio + lbp_std,
            alpha=0.2, color=colores[idx]
        )

        n_muestras = np.sum(mascara)
        ax.set_title(f"{clase_nombre}\n(n={n_muestras})", fontsize=9)
        ax.set_xlabel("Bin LBP", fontsize=7)
        ax.set_ylabel("Frecuencia", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-0.5, LBP_N_BINS - 0.5)

    # Desactivar ejes sobrantes
    for idx in range(n_clases, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Visualización guardada en: {ruta_salida}")


# =============================================================================
# Pipeline principal de extracción
# =============================================================================

def ejecutar_extraccion(dir_entrada: str, dir_salida: str) -> None:
    """
    Ejecuta el pipeline completo de extracción de características visuales.

    Para cada imagen preprocesada (256×256, escala de grises):
    1. Extrae descriptor LBP (26 dimensiones)
    2. Extrae descriptor GLCM (60 dimensiones)
    3. Extrae histograma de intensidad (256 dimensiones)

    Guarda los resultados como arrays NumPy individuales y combinados.

    Args:
        dir_entrada (str): Directorio con imágenes preprocesadas (256x256).
        dir_salida (str): Directorio donde se guardarán los archivos .npy.
    """
    global CLASES, CLASSES

    # Validar directorio de entrada
    if not os.path.isdir(dir_entrada):
        print(f"\n[ERROR] Directorio de entrada no encontrado: {dir_entrada}")
        print("  Ejecuta primero: python 01_preprocesamiento_imagenes.py")
        sys.exit(1)

    CLASES = classes_with_files(dir_entrada)
    CLASSES = CLASES

    # Crear directorio de salida para features
    os.makedirs(dir_salida, exist_ok=True)

    # Obtener imágenes organizadas por clase
    imagenes_por_clase = obtener_imagenes_por_clase(dir_entrada)

    # Listas para acumular features y etiquetas
    todas_lbp = []
    todas_glcm = []
    todos_hist = []
    todas_etiquetas = []
    nombres_archivos = []  # Para trazabilidad

    # Contadores
    total_procesadas = 0
    total_errores = 0

    print("\n[INFO] Extrayendo características visuales...")
    print(f"  LBP: radio={LBP_RADIUS}, puntos={LBP_N_POINTS}, bins={LBP_N_BINS}")
    print(f"  GLCM: distancias={GLCM_DISTANCES}, ángulos=4, propiedades={len(GLCM_PROPERTIES)}")
    print(f"  Histograma: {HIST_N_BINS} bins")
    print(f"  Dimensiones esperadas: LBP={LBP_N_BINS}, GLCM={len(GLCM_PROPERTIES)*len(GLCM_DISTANCES)*len(GLCM_ANGLES)}, Hist={HIST_N_BINS}\n")

    for idx_clase, clase in enumerate(CLASES):
        rutas = imagenes_por_clase.get(clase, [])
        if not rutas:
            continue

        desc = f"  [{clase}]"
        for ruta in tqdm(rutas, desc=desc, ncols=90):
            try:
                # Leer imagen preprocesada en escala de grises
                imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                if imagen is None:
                    print(f"\n  [ADVERTENCIA] No se pudo leer: {ruta}")
                    total_errores += 1
                    continue

                # Extraer las tres familias de características
                feat_lbp = extraer_lbp(imagen)
                feat_glcm = extraer_glcm(imagen)
                feat_hist = extraer_histograma_intensidad(imagen)

                # Acumular resultados
                todas_lbp.append(feat_lbp)
                todas_glcm.append(feat_glcm)
                todos_hist.append(feat_hist)
                todas_etiquetas.append(idx_clase)
                nombres_archivos.append(str(Path(ruta).relative_to(dir_entrada)))

                total_procesadas += 1

            except Exception as e:
                print(f"\n  [ERROR] Fallo en {ruta}: {e}")
                total_errores += 1

    # Verificar que se procesaron imágenes
    if total_procesadas == 0:
        print("\n[ERROR] No se procesó ninguna imagen. Verifica el dataset.")
        sys.exit(1)

    # Convertir listas a arrays NumPy
    print("\n[INFO] Convirtiendo a arrays NumPy...")
    features_lbp = np.array(todas_lbp, dtype=np.float32)
    features_glcm = np.array(todas_glcm, dtype=np.float32)
    features_hist = np.array(todos_hist, dtype=np.float32)
    labels = np.array(todas_etiquetas, dtype=np.int32)
    clases_presentes = [CLASES[i] for i in sorted(set(labels.tolist()))]
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(labels.tolist())))}
    labels = np.array([remap[int(label)] for label in labels], dtype=np.int32)
    class_names = np.array(clases_presentes)

    # Combinar las tres familias de features en un solo vector
    # Dimensión total: LBP(26) + GLCM(60) + Hist(256) = 342
    features_combined = np.concatenate(
        [features_lbp, features_glcm, features_hist],
        axis=1
    )

    # Guardar todos los arrays a disco
    print("\n[INFO] Guardando features a disco...")
    archivos_salida = {
        "features_lbp.npy": features_lbp,
        "features_glcm.npy": features_glcm,
        "features_histograms.npy": features_hist,
        "features_visual_combined.npy": features_combined,
        "labels.npy": labels,
        "class_names.npy": class_names,
        "file_names_visual.npy": np.array(nombres_archivos),
    }

    for nombre_archivo, array in archivos_salida.items():
        ruta_completa = os.path.join(dir_salida, nombre_archivo)
        np.save(ruta_completa, array)
        print(f"  [OK] {nombre_archivo:35s} -> shape: {str(array.shape):20s} dtype: {array.dtype}")

    # Resumen final
    print(f"\n{'='*60}")
    print(f"  Extracción de características completada")
    print(f"{'='*60}")
    print(f"  Imágenes procesadas  : {total_procesadas}")
    print(f"  Errores              : {total_errores}")
    print(f"  Dimensiones LBP      : {features_lbp.shape[1]}")
    print(f"  Dimensiones GLCM     : {features_glcm.shape[1]}")
    print(f"  Dimensiones Histograma: {features_hist.shape[1]}")
    print(f"  Dimensiones Combinadas: {features_combined.shape[1]}")
    print(f"  Directorio de salida : {os.path.abspath(dir_salida)}")
    print(f"{'='*60}")

    # Generar visualización de histogramas LBP por categoría
    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    ruta_vis = os.path.join(DIR_RESULTADOS, "02_histogramas_lbp.png")
    generar_visualizacion_lbp(features_lbp, labels, list(class_names), ruta_vis)


# =============================================================================
# Punto de entrada - CLI con argparse
# =============================================================================

def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description=(
            "Extracción de características visuales (LBP, GLCM, histogramas) "
            "a partir de imágenes de documentos preprocesadas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplo de uso:\n"
            "  python 02_extraccion_features_visuales.py --input dataset_preprocessed/256x256 --output features\n\n"
            "Requisitos:\n"
            "  - Ejecutar primero 01_preprocesamiento_imagenes.py\n"
            "  - Las imágenes deben estar en escala de grises (256×256)\n\n"
            "Características extraídas:\n"
            "  - LBP: Local Binary Patterns (26 dimensiones)\n"
            "  - GLCM: Gray-Level Co-occurrence Matrix (60 dimensiones)\n"
            "  - Histograma de intensidad (256 dimensiones)\n"
            "  - Combinado: LBP + GLCM + Histograma (342 dimensiones)"
        )
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.join("dataset_preprocessed", "256x256"),
        help="Directorio con imágenes preprocesadas 256×256 (default: dataset_preprocessed/256x256)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="features",
        help="Directorio de salida para archivos .npy (default: features)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  EXTRACCIÓN DE CARACTERÍSTICAS VISUALES")
    print("  Proyecto Final - Recuperación de Información")
    print("=" * 60)
    print(f"  Entrada    : {os.path.abspath(args.input)}")
    print(f"  Salida     : {os.path.abspath(args.output)}")
    print(f"  Descriptores: LBP, GLCM, Histograma de Intensidad")
    print(f"  Clases     : {len(CLASES)}")
    print("=" * 60)

    ejecutar_extraccion(args.input, args.output)

    print("\n[INFO] Script finalizado exitosamente.\n")


if __name__ == "__main__":
    main()

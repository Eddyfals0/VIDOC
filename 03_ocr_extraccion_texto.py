#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
  03_ocr_extraccion_texto.py
  Proyecto Final - Recuperación de Información (8vo Semestre)
=============================================================================
  Descripción:
    Extrae texto de imágenes de documentos usando Tesseract OCR y lo
    preprocesa siguiendo los patrones de los laboratorios 2 y 9:
      1. Conversión a minúsculas
      2. Eliminación de caracteres especiales (regex)
      3. Eliminación de números
      4. Eliminación de espacios extra
      5. Eliminación de stopwords (inglés)
      6. Stemming con SnowballStemmer (inglés)

  Entradas:
    - dataset/{clase}/*.jpg  (imágenes de documentos)

  Salidas:
    - dataset_text_raw/{clase}/{archivo}.txt  (texto OCR crudo)
    - dataset_text/{clase}/{archivo}.txt      (texto preprocesado)
    - resultados/03_ocr_estadisticas.txt      (estadísticas de OCR)
    - resultados/03_ejemplos_ocr.png          (ejemplos visuales)

  Uso:
    python 03_ocr_extraccion_texto.py
    python 03_ocr_extraccion_texto.py --dataset_dir dataset --workers 4
=============================================================================
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para guardar figuras sin mostrarlas
import matplotlib.pyplot as plt
import pytesseract
from tqdm import tqdm

# --- Configuración de NLTK (importar después para evitar errores si no está instalado) ---
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# =============================================================================
#  CONFIGURACIÓN GLOBAL
# =============================================================================

# Ruta de Tesseract OCR en este sistema Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Clases del proyecto RVL-CDIP reducido a 14 categorias descargadas.
import os
from pathlib import Path

from project_config import classes_with_files

CLASES = classes_with_files("dataset")
CLASSES = CLASES

# Stopwords y stemmer (se inicializan una sola vez)
STOPWORDS_EN = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')


# =============================================================================
#  FUNCIONES DE PREPROCESAMIENTO DE IMAGEN
# =============================================================================

def preprocesar_imagen(ruta_imagen):
    """
    Preprocesa una imagen para mejorar la calidad del OCR:
      1. Carga la imagen en escala de grises
      2. Aplica umbralización OTSU para binarizar

    Args:
        ruta_imagen (str): Ruta absoluta a la imagen .jpg

    Returns:
        numpy.ndarray: Imagen binarizada lista para OCR, o None si falla
    """
    try:
        # Leer la imagen en escala de grises
        imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            return None

        # Aplicar umbralización de Otsu para binarizar la imagen
        _, imagen_bin = cv2.threshold(
            imagen, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return imagen_bin

    except Exception as e:
        print(f"  [ERROR] No se pudo preprocesar la imagen {ruta_imagen}: {e}")
        return None


# =============================================================================
#  FUNCIONES DE EXTRACCIÓN OCR
# =============================================================================

def extraer_texto_ocr(ruta_imagen):
    """
    Extrae texto de una imagen usando Tesseract OCR.

    Args:
        ruta_imagen (str): Ruta a la imagen

    Returns:
        str: Texto extraído (crudo), o cadena vacía si falla
    """
    try:
        # Preprocesar la imagen antes del OCR
        imagen_bin = preprocesar_imagen(ruta_imagen)
        if imagen_bin is None:
            return ""

        # Ejecutar Tesseract OCR con configuración optimizada para documentos
        # PSM 6 = Assume a single uniform block of text (bueno para documentos)
        config_tesseract = '--psm 6 --oem 3'
        texto = pytesseract.image_to_string(imagen_bin, lang='eng', config=config_tesseract)
        return texto.strip()

    except Exception as e:
        print(f"  [ERROR] OCR falló para {ruta_imagen}: {e}")
        return ""


# =============================================================================
#  FUNCIONES DE PREPROCESAMIENTO DE TEXTO
# =============================================================================

def preprocesar_texto(texto):
    """
    Preprocesa el texto extraído por OCR siguiendo los patrones de los
    laboratorios 2 y 9 del curso:
      1. Convertir a minúsculas
      2. Eliminar caracteres especiales (solo letras, números y espacios)
      3. Eliminar números
      4. Eliminar espacios extra
      5. Eliminar stopwords en inglés
      6. Aplicar stemming con SnowballStemmer

    Args:
        texto (str): Texto crudo del OCR

    Returns:
        str: Texto preprocesado listo para representación TF-IDF
    """
    if not texto or len(texto.strip()) == 0:
        return ""

    # Paso 1: Convertir a minúsculas
    texto = texto.lower()

    # Paso 2: Eliminar caracteres especiales (dejar solo letras, números y espacios)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)

    # Paso 3: Eliminar números
    texto = re.sub(r'\d+', '', texto)

    # Paso 4: Eliminar espacios extra (múltiples espacios → uno solo)
    texto = re.sub(r'\s+', ' ', texto).strip()

    # Paso 5: Tokenizar y eliminar stopwords
    tokens = texto.split()
    tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 1]

    # Paso 6: Aplicar stemming
    tokens = [STEMMER.stem(t) for t in tokens]

    return ' '.join(tokens)


# =============================================================================
#  FUNCIÓN DE PROCESAMIENTO POR ARCHIVO (para multiprocessing)
# =============================================================================

def procesar_archivo(args_tupla):
    """
    Procesa un solo archivo: extrae texto OCR, lo preprocesa y lo guarda.
    Diseñada para ser usada con ProcessPoolExecutor.

    Args:
        args_tupla: (ruta_imagen, ruta_raw, ruta_procesado)

    Returns:
        dict: Resultados del procesamiento con estadísticas
    """
    ruta_imagen, ruta_raw, ruta_procesado = args_tupla

    resultado = {
        'archivo': str(ruta_imagen),
        'exito': False,
        'len_raw': 0,
        'len_procesado': 0,
        'texto_raw': '',
        'texto_procesado': ''
    }

    try:
        # Extraer texto con OCR
        texto_raw = extraer_texto_ocr(ruta_imagen)

        # Preprocesar el texto
        texto_procesado = preprocesar_texto(texto_raw)

        # Crear directorios si no existen
        os.makedirs(os.path.dirname(ruta_raw), exist_ok=True)
        os.makedirs(os.path.dirname(ruta_procesado), exist_ok=True)

        # Guardar texto crudo
        with open(ruta_raw, 'w', encoding='utf-8') as f:
            f.write(texto_raw)

        # Guardar texto preprocesado
        with open(ruta_procesado, 'w', encoding='utf-8') as f:
            f.write(texto_procesado)

        resultado['exito'] = len(texto_raw) > 0
        resultado['len_raw'] = len(texto_raw)
        resultado['len_procesado'] = len(texto_procesado)
        resultado['texto_raw'] = texto_raw[:500]  # Guardar solo primeros 500 chars para ejemplos
        resultado['texto_procesado'] = texto_procesado[:500]

    except Exception as e:
        print(f"  [ERROR] Falló procesamiento de {ruta_imagen}: {e}")

    return resultado


# =============================================================================
#  FUNCIÓN PARA GENERAR VISUALIZACIÓN DE EJEMPLOS
# =============================================================================

def generar_ejemplos_ocr(dataset_dir, resultados_por_clase, ruta_salida):
    """
    Genera una imagen con 4 ejemplos de OCR: imagen original + texto extraído.

    Args:
        dataset_dir (str): Directorio del dataset
        resultados_por_clase (dict): Resultados organizados por clase
        ruta_salida (str): Ruta donde guardar la imagen
    """
    print("\n[INFO] Generando visualización de ejemplos OCR...")

    # Seleccionar 4 clases con buenos resultados para los ejemplos
    clases_ejemplo = []
    for clase in CLASES:
        if clase in resultados_por_clase:
            # Buscar un resultado exitoso con texto
            for r in resultados_por_clase[clase]:
                if r['exito'] and r['len_raw'] > 50:
                    clases_ejemplo.append((clase, r))
                    break
        if len(clases_ejemplo) >= 4:
            break

    # Si no hay suficientes, tomar los que haya
    if len(clases_ejemplo) == 0:
        print("  [ADVERTENCIA] No hay ejemplos exitosos para visualizar")
        return

    n_ejemplos = min(4, len(clases_ejemplo))

    fig, axes = plt.subplots(n_ejemplos, 2, figsize=(16, 5 * n_ejemplos))
    fig.suptitle('Ejemplos de Extracción OCR con Tesseract', fontsize=16, fontweight='bold')

    if n_ejemplos == 1:
        axes = [axes]  # Asegurar que sea iterable

    for idx, (clase, resultado) in enumerate(clases_ejemplo[:n_ejemplos]):
        # Cargar imagen original
        ruta_img = resultado['archivo']
        imagen = cv2.imread(ruta_img)

        if imagen is not None:
            # Convertir BGR a RGB para matplotlib
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            axes[idx][0].imshow(imagen_rgb, cmap='gray')
        else:
            axes[idx][0].text(0.5, 0.5, 'Imagen no disponible',
                              ha='center', va='center', fontsize=12)

        axes[idx][0].set_title(f'Clase: {clase}', fontsize=12, fontweight='bold')
        axes[idx][0].axis('off')

        # Mostrar texto extraído
        texto_mostrar = resultado['texto_raw'][:400]
        if len(resultado['texto_raw']) > 400:
            texto_mostrar += '\n...(truncado)'

        axes[idx][1].text(
            0.05, 0.95, texto_mostrar,
            transform=axes[idx][1].transAxes,
            fontsize=7, verticalalignment='top',
            fontfamily='monospace',
            wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
        axes[idx][1].set_title(
            f'Texto OCR ({resultado["len_raw"]} caracteres)',
            fontsize=12, fontweight='bold'
        )
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Ejemplos guardados en: {ruta_salida}")


# =============================================================================
#  FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal: orquesta la extracción OCR de todo el dataset.
    """
    # --- Argumentos de línea de comandos ---
    parser = argparse.ArgumentParser(
        description='Extracción de texto OCR de imágenes de documentos con Tesseract'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='dataset',
        help='Directorio raíz del dataset con las imágenes (default: dataset)'
    )
    parser.add_argument(
        '--output_raw', type=str, default='dataset_text_raw',
        help='Directorio de salida para texto OCR crudo (default: dataset_text_raw)'
    )
    parser.add_argument(
        '--output_text', type=str, default='dataset_text',
        help='Directorio de salida para texto preprocesado (default: dataset_text)'
    )
    parser.add_argument(
        '--resultados_dir', type=str, default='resultados',
        help='Directorio para estadísticas y visualizaciones (default: resultados)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Número de workers para procesamiento paralelo (default: 4)'
    )
    parser.add_argument(
        '--max_per_class', type=int, default=0,
        help='Máximo de imágenes por clase (0 = todas, default: 0)'
    )
    args = parser.parse_args()

    # --- Crear directorios de salida ---
    os.makedirs(args.output_raw, exist_ok=True)
    os.makedirs(args.output_text, exist_ok=True)
    os.makedirs(args.resultados_dir, exist_ok=True)

    print("=" * 70)
    print("  EXTRACCIÓN DE TEXTO OCR CON TESSERACT")
    print("=" * 70)
    print(f"  Dataset:         {args.dataset_dir}")
    print(f"  Salida crudo:    {args.output_raw}")
    print(f"  Salida procesado:{args.output_text}")
    print(f"  Workers:         {args.workers}")
    print(f"  Máx por clase:   {'todas' if args.max_per_class == 0 else args.max_per_class}")
    print("=" * 70)

    # --- Verificar que Tesseract está disponible ---
    try:
        version = pytesseract.get_tesseract_version()
        print(f"\n[OK] Tesseract OCR v{version} encontrado")
    except Exception as e:
        print(f"\n[ERROR FATAL] Tesseract no encontrado: {e}")
        print("  Asegúrate de que está instalado en:")
        print(f"  {pytesseract.pytesseract.tesseract_cmd}")
        sys.exit(1)

    # --- Recopilar todas las imágenes por clase ---
    print("\n[INFO] Escaneando imágenes del dataset...")
    tareas = []  # Lista de (ruta_imagen, ruta_raw, ruta_procesado)
    conteo_por_clase = {}

    for clase in CLASES:
        dir_clase = Path(args.dataset_dir) / clase
        if not dir_clase.exists():
            print(f"  [ADVERTENCIA] Directorio no encontrado: {dir_clase}")
            continue

        # Obtener todas las imágenes JPG de esta clase
        imagenes = sorted(dir_clase.glob('*.jpg'))

        # Limitar si se especificó --max_per_class
        if args.max_per_class > 0:
            imagenes = imagenes[:args.max_per_class]

        conteo_por_clase[clase] = len(imagenes)

        for img_path in imagenes:
            nombre_base = img_path.stem  # nombre sin extensión
            ruta_raw = Path(args.output_raw) / clase / f"{nombre_base}.txt"
            ruta_procesado = Path(args.output_text) / clase / f"{nombre_base}.txt"
            
            # Saltar si ya existe
            if ruta_procesado.exists() and ruta_procesado.stat().st_size > 0:
                continue
                
            tareas.append((str(img_path), str(ruta_raw), str(ruta_procesado)))

    total_imagenes = len(tareas)
    print(f"\n[INFO] Total de imágenes a procesar (saltando existentes): {total_imagenes}")
    for clase, conteo in conteo_por_clase.items():
        print(f"  - {clase:30s}: {conteo:5d} imágenes")

    if total_imagenes == 0:
        print("\n[ERROR] No se encontraron imágenes para procesar.")
        sys.exit(1)

    # --- Procesar imágenes con multiprocessing ---
    print(f"\n[INFO] Iniciando extracción OCR con {args.workers} workers...")
    inicio = time.time()

    resultados_todos = []
    resultados_por_clase = {clase: [] for clase in CLASES}

    # Usar ProcessPoolExecutor para procesamiento paralelo
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Enviar todas las tareas
        futuros = {executor.submit(procesar_archivo, tarea): tarea for tarea in tareas}

        # Recopilar resultados con barra de progreso
        with tqdm(total=total_imagenes, desc="Procesando OCR", unit="img") as pbar:
            for futuro in as_completed(futuros):
                resultado = futuro.result()
                resultados_todos.append(resultado)

                # Clasificar por clase
                ruta_img = resultado['archivo']
                for clase in CLASES:
                    if f"/{clase}/" in ruta_img.replace('\\', '/'):
                        resultados_por_clase[clase].append(resultado)
                        break

                pbar.update(1)

    tiempo_total = time.time() - inicio

    # --- Calcular estadísticas ---
    print("\n[INFO] Calculando estadísticas de OCR...")

    total_exitosos = sum(1 for r in resultados_todos if r['exito'])
    total_fallidos = total_imagenes - total_exitosos
    tasa_exito = (total_exitosos / total_imagenes) * 100 if total_imagenes > 0 else 0

    # Estadísticas por clase
    estadisticas_clase = {}
    for clase in CLASES:
        resultados_clase = resultados_por_clase[clase]
        if not resultados_clase:
            continue

        exitosos = [r for r in resultados_clase if r['exito']]
        n_total = len(resultados_clase)
        n_exitosos = len(exitosos)

        if exitosos:
            avg_len_raw = np.mean([r['len_raw'] for r in exitosos])
            avg_len_proc = np.mean([r['len_procesado'] for r in exitosos])
            max_len = max(r['len_raw'] for r in exitosos)
            min_len = min(r['len_raw'] for r in exitosos)
        else:
            avg_len_raw = avg_len_proc = max_len = min_len = 0

        estadisticas_clase[clase] = {
            'total': n_total,
            'exitosos': n_exitosos,
            'tasa_exito': (n_exitosos / n_total) * 100 if n_total > 0 else 0,
            'avg_len_raw': avg_len_raw,
            'avg_len_proc': avg_len_proc,
            'max_len': max_len,
            'min_len': min_len
        }

    # --- Guardar estadísticas en archivo ---
    ruta_stats = Path(args.resultados_dir) / '03_ocr_estadisticas.txt'
    with open(ruta_stats, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  ESTADÍSTICAS DE EXTRACCIÓN OCR CON TESSERACT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Fecha de ejecución:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tiempo total:           {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} min)\n")
        f.write(f"Workers utilizados:     {args.workers}\n")
        f.write(f"Velocidad promedio:     {total_imagenes/tiempo_total:.1f} imágenes/segundo\n\n")

        f.write("-" * 70 + "\n")
        f.write("  RESUMEN GENERAL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total de imágenes:      {total_imagenes}\n")
        f.write(f"Extracciones exitosas:  {total_exitosos} ({tasa_exito:.1f}%)\n")
        f.write(f"Extracciones fallidas:  {total_fallidos}\n\n")

        f.write("-" * 70 + "\n")
        f.write("  ESTADÍSTICAS POR CLASE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Clase':<30s} {'Total':>6s} {'Éxito':>6s} {'%':>7s} {'Prom.Raw':>10s} {'Prom.Proc':>10s}\n")
        f.write("-" * 70 + "\n")

        for clase in CLASES:
            if clase in estadisticas_clase:
                e = estadisticas_clase[clase]
                f.write(
                    f"{clase:<30s} {e['total']:>6d} {e['exitosos']:>6d} "
                    f"{e['tasa_exito']:>6.1f}% {e['avg_len_raw']:>10.1f} "
                    f"{e['avg_len_proc']:>10.1f}\n"
                )

        f.write("-" * 70 + "\n")
        f.write(
            f"{'TOTAL':<30s} {total_imagenes:>6d} {total_exitosos:>6d} "
            f"{tasa_exito:>6.1f}%\n"
        )
        f.write("=" * 70 + "\n")

    print(f"  [OK] Estadísticas guardadas en: {ruta_stats}")

    # --- Imprimir resumen en consola ---
    print("\n" + "=" * 70)
    print("  RESUMEN DE EXTRACCIÓN OCR")
    print("=" * 70)
    print(f"  Imágenes procesadas:  {total_imagenes}")
    print(f"  Extracciones exitosas:{total_exitosos} ({tasa_exito:.1f}%)")
    print(f"  Tiempo total:         {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    print(f"  Velocidad:            {total_imagenes/tiempo_total:.1f} img/s")
    print()

    print(f"  {'Clase':<25s} {'Éxito':>7s} {'Prom.Raw':>10s} {'Prom.Proc':>10s}")
    print("  " + "-" * 55)
    for clase in CLASES:
        if clase in estadisticas_clase:
            e = estadisticas_clase[clase]
            print(
                f"  {clase:<25s} {e['tasa_exito']:>6.1f}% "
                f"{e['avg_len_raw']:>10.1f} {e['avg_len_proc']:>10.1f}"
            )
    print("=" * 70)

    # --- Generar visualización de ejemplos ---
    ruta_ejemplos = Path(args.resultados_dir) / '03_ejemplos_ocr.png'
    generar_ejemplos_ocr(args.dataset_dir, resultados_por_clase, str(ruta_ejemplos))

    print("\n[INFO] ¡Extracción OCR completada exitosamente!")
    print(f"  - Texto crudo en:      {args.output_raw}/")
    print(f"  - Texto procesado en:  {args.output_text}/")
    print(f"  - Estadísticas en:     {ruta_stats}")
    print(f"  - Ejemplos en:         {ruta_ejemplos}")


# =============================================================================
#  PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    main()

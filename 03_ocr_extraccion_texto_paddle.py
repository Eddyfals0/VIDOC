#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
  03_ocr_extraccion_texto_paddle.py
  Proyecto Final - Recuperación de Información (8vo Semestre)
=============================================================================
  Descripción:
    Extrae texto de imágenes de documentos usando PaddleOCR y el algoritmo
    de reconstrucción de párrafos/burbujas de ocr_example.py.
    Preprocesa el texto siguiendo los patrones de los laboratorios del curso:
      1. Conversión a minúsculas
      2. Eliminación de caracteres especiales (regex)
      3. Eliminación de números
      4. Eliminación de espacios extra
      5. Eliminación de stopwords (inglés)
      6. Stemming con SnowballStemmer (inglés)

  Entradas:
    - dataset/{clase}/*.jpg  (imágenes de documentos)

  Salidas:
    - dataset_text_raw/{clase}/{archivo}.txt  (texto OCR agrupado y reconstruido)
    - dataset_text/{clase}/{archivo}.txt      (texto preprocesado)
    - resultados/03_ocr_estadisticas_paddle.txt (estadísticas detalladas)
    - resultados/03_ejemplos_ocr_paddle.png     (visualización de ejemplos)

  Uso:
    python 03_ocr_extraccion_texto_paddle.py
    python 03_ocr_extraccion_texto_paddle.py --dataset_dir dataset --workers 2
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
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuración de NLTK ---
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

# Clases del proyecto
from project_config import classes_with_files
CLASES = classes_with_files("dataset")

# Stopwords y stemmer
STOPWORDS_EN = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')

# Instancia global del modelo por proceso (lazy initialization)
_OCR_MODEL = None

def init_paddle_ocr():
    """
    Inicializa PaddleOCR de forma perezosa por proceso.
    """
    global _OCR_MODEL
    if _OCR_MODEL is None:
        try:
            from paddleocr import PaddleOCR
            _OCR_MODEL = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='en',
                enable_mkldnn=False
            )
        except Exception as e:
            print(f"\n[ERROR] No se pudo inicializar PaddleOCR: {e}")
            raise e

# =============================================================================
#  FUNCIONES DE EXTRACCIÓN OCR (PaddleOCR + Algoritmo ocr_example.py)
# =============================================================================

def extraer_texto_paddle(ruta_imagen):
    """
    Extrae texto de una imagen usando PaddleOCR y reconstruyendo el flujo
    de lectura según el algoritmo geométrico de ocr_example.py.

    Args:
        ruta_imagen (str): Ruta a la imagen

    Returns:
        str: Texto agrupado y formateado, o cadena vacía si falla
    """
    global _OCR_MODEL
    try:
        init_paddle_ocr()
        if _OCR_MODEL is None:
            return ""

        # Ejecutar PaddleOCR
        resultados = _OCR_MODEL.predict(str(ruta_imagen))
        if not resultados:
            return ""

        textos = []
        poligonos = []

        # 1. Normalizar formatos de salida de PaddleOCR
        # A) Formato diccionario personalizado de ocr_example.py:
        if isinstance(resultados[0], dict) and 'rec_texts' in resultados[0] and 'dt_polys' in resultados[0]:
            textos = resultados[0]['rec_texts']
            poligonos = resultados[0]['dt_polys']
        # B) Formato estándar de PaddleOCR (lista de líneas de texto y coordenadas):
        elif isinstance(resultados, list) and len(resultados) > 0:
            lineas = resultados[0] if isinstance(resultados[0], list) else resultados
            for linea in lineas:
                if linea is not None and len(linea) == 2:
                    poly, (text, conf) = linea
                    textos.append(text)
                    poligonos.append(poly)

        if not textos:
            return ""

        # 2. Extraer información espacial de cada línea
        items = []
        for poly, texto in zip(poligonos, textos):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            items.append({
                'text': texto,
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys),
                'center_y': sum(ys) / 4
            })

        # 3. Agrupar líneas en "burbujas" (bloques de texto) usando DSU (Disjoint Set Union)
        def misma_burbuja(i1, i2):
            dx = max(0, max(i1['min_x'] - i2['max_x'], i2['min_x'] - i1['max_x']))
            dy = max(0, max(i1['min_y'] - i2['max_y'], i2['min_y'] - i1['max_y']))
            # Están en el mismo renglón y muy cerca, o apilados verticalmente
            mismo_renglon = abs(i1['center_y'] - i2['center_y']) < 6 and dx < 20
            apilados = dx < 25 and dy < 50
            return mismo_renglon or apilados

        padres = {i: i for i in range(len(items))}
        def find(i):
            if padres[i] != i:
                padres[i] = find(padres[i])
            return padres[i]

        def union(i, j):
            raiz_i = find(i)
            raiz_j = find(j)
            if raiz_i != raiz_j:
                padres[raiz_i] = raiz_j

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if misma_burbuja(items[i], items[j]):
                    union(i, j)

        bloques_dict = {}
        for i, item in enumerate(items):
            raiz = find(i)
            if raiz not in bloques_dict:
                bloques_dict[raiz] = []
            bloques_dict[raiz].append(item)

        bloques = []
        for raiz, b_items in bloques_dict.items():
            min_x = min(i['min_x'] for i in b_items)
            max_x = max(i['max_x'] for i in b_items)
            min_y = min(i['min_y'] for i in b_items)
            max_y = max(i['max_y'] for i in b_items)
            bloques.append({
                'items': b_items,
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'center_x': (min_x + max_x) / 2,
                'center_y': (min_y + max_y) / 2
            })

        # 4. Agrupar bloques geométricos en filas
        bloques.sort(key=lambda b: b['center_y'])
        filas = []
        fila_actual = []
        y_actual = None

        for b in bloques:
            if y_actual is None:
                y_actual = b['center_y']
                fila_actual.append(b)
            elif abs(b['center_y'] - y_actual) < 180:  # Tolerancia horizontal
                fila_actual.append(b)
            else:
                fila_actual.sort(key=lambda b: b['center_x'])  # Izquierda a derecha
                filas.append(fila_actual)
                y_actual = b['center_y']
                fila_actual = [b]

        if fila_actual:
            fila_actual.sort(key=lambda b: b['center_x'])
            filas.append(fila_actual)

        # 5. Generar texto final estructurado por párrafos
        texto_final = []
        for fila in filas:
            for bloque in fila:
                bloque['items'].sort(key=lambda i: i['center_y'])

                lineas_burbuja = []
                linea_act = []
                y_linea = None
                for item in bloque['items']:
                    if y_linea is None:
                        y_linea = item['center_y']
                        linea_act.append(item)
                    elif abs(item['center_y'] - y_linea) < 8:
                        linea_act.append(item)
                    else:
                        linea_act.sort(key=lambda i: i['min_x'])
                        lineas_burbuja.append(" ".join(i['text'] for i in linea_act))
                        y_linea = item['center_y']
                        linea_act = [item]

                if linea_act:
                    linea_act.sort(key=lambda i: i['min_x'])
                    lineas_burbuja.append(" ".join(i['text'] for i in linea_act))

                for l in lineas_burbuja:
                    texto_final.append(l)
                # Separador de párrafo
                texto_final.append("")

        return "\n".join(texto_final).strip()

    except Exception as e:
        print(f"  [ERROR] PaddleOCR falló para {ruta_imagen}: {e}")
        return ""

# =============================================================================
#  FUNCIONES DE PREPROCESAMIENTO DE TEXTO
# =============================================================================

def preprocesar_texto(texto):
    """
    Preprocesa el texto extraído por OCR siguiendo los laboratorios de RI:
      1. Convertir a minúsculas
      2. Eliminar caracteres especiales
      3. Eliminar números
      4. Eliminar espacios extra
      5. Eliminar stopwords (inglés)
      6. Aplicar stemming
    """
    if not texto or len(texto.strip()) == 0:
        return ""

    # Paso 1: Convertir a minúsculas
    texto = texto.lower()

    # Paso 2: Eliminar caracteres especiales (dejar solo letras, números y espacios)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)

    # Paso 3: Eliminar números
    texto = re.sub(r'\d+', '', texto)

    # Paso 4: Eliminar espacios extra
    texto = re.sub(r'\s+', ' ', texto).strip()

    # Paso 5: Tokenizar y eliminar stopwords
    tokens = texto.split()
    tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 1]

    # Paso 6: Aplicar stemming
    tokens = [STEMMER.stem(t) for t in tokens]

    return ' '.join(tokens)

# =============================================================================
#  PROCESAMIENTO POR ARCHIVO (para paralelización en procesos)
# =============================================================================

def procesar_archivo(args_tupla):
    """
    Procesa una sola imagen, extrae texto con PaddleOCR, lo preprocesa y lo guarda.
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
        # Extraer texto usando el pipeline de PaddleOCR y ocr_example.py
        texto_raw = extraer_texto_paddle(ruta_imagen)

        # Preprocesar
        texto_procesado = preprocesar_texto(texto_raw)

        # Guardar en disco si hay texto
        if len(texto_raw) > 0:
            os.makedirs(os.path.dirname(ruta_raw), exist_ok=True)
            os.makedirs(os.path.dirname(ruta_procesado), exist_ok=True)

            with open(ruta_raw, 'w', encoding='utf-8') as f:
                f.write(texto_raw)

            with open(ruta_procesado, 'w', encoding='utf-8') as f:
                f.write(texto_procesado)

            resultado['exito'] = True
            resultado['len_raw'] = len(texto_raw)
            resultado['len_procesado'] = len(texto_procesado)
            resultado['texto_raw'] = texto_raw[:500]  # Resumen
            resultado['texto_procesado'] = texto_procesado[:500]

    except Exception as e:
        print(f"  [ERROR] Falló procesamiento de {ruta_imagen}: {e}")

    return resultado

# =============================================================================
#  FUNCIÓN PARA GENERAR VISUALIZACIÓN DE EJEMPLOS
# =============================================================================

def generar_ejemplos_ocr(resultados_por_clase, ruta_salida):
    """
    Dibuja una cuadrícula con ejemplos de OCR.
    """
    print("\n[INFO] Generando visualización de ejemplos OCR...")
    clases_ejemplo = []
    for clase in CLASES:
        if clase in resultados_por_clase:
            for r in resultados_por_clase[clase]:
                if r['exito'] and r['len_raw'] > 50:
                    clases_ejemplo.append((clase, r))
                    break
        if len(clases_ejemplo) >= 4:
            break

    if len(clases_ejemplo) == 0:
        print("  [ADVERTENCIA] No hay ejemplos con texto exitoso para graficar")
        return

    n_ejemplos = min(4, len(clases_ejemplo))
    fig, axes = plt.subplots(n_ejemplos, 2, figsize=(16, 5 * n_ejemplos))
    fig.suptitle('Ejemplos de Extracción OCR con PaddleOCR', fontsize=16, fontweight='bold')

    if n_ejemplos == 1:
        axes = [axes]

    for idx, (clase, resultado) in enumerate(clases_ejemplo[:n_ejemplos]):
        ruta_img = resultado['archivo']
        imagen = cv2.imread(ruta_img)

        if imagen is not None:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            axes[idx][0].imshow(imagen_rgb)
        else:
            axes[idx][0].text(0.5, 0.5, 'Imagen no cargada', ha='center', va='center')

        axes[idx][0].set_title(f'Clase: {clase}', fontsize=12, fontweight='bold')
        axes[idx][0].axis('off')

        texto_mostrar = resultado['texto_raw'][:400]
        if len(resultado['texto_raw']) > 400:
            texto_mostrar += '\n...(truncado)'

        axes[idx][1].text(
            0.05, 0.95, texto_mostrar,
            transform=axes[idx][1].transAxes,
            fontsize=7, verticalalignment='top',
            fontfamily='monospace',
            wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
        )
        axes[idx][1].set_title(
            f'Texto Agrupado por PaddleOCR ({resultado["len_raw"]} caracteres)',
            fontsize=12, fontweight='bold'
        )
        axes[idx][1].axis('off')

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Imagen de ejemplos guardada en: {ruta_salida}")

# =============================================================================
#  FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extracción de texto OCR de imágenes con PaddleOCR'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='dataset',
        help='Directorio raíz de imágenes (default: dataset)'
    )
    parser.add_argument(
        '--output_raw', type=str, default='dataset_text_raw',
        help='Salida de texto crudo reconstruido (default: dataset_text_raw)'
    )
    parser.add_argument(
        '--output_text', type=str, default='dataset_text',
        help='Salida de texto preprocesado (default: dataset_text)'
    )
    parser.add_argument(
        '--resultados_dir', type=str, default='resultados',
        help='Directorio para estadísticas (default: resultados)'
    )
    parser.add_argument(
        '--workers', type=int, default=2,
        help='Número de procesos paralelos. NOTA: PaddleOCR requiere mucha memoria, se recomiendan 1 o 2 (default: 2)'
    )
    parser.add_argument(
        '--max_per_class', type=int, default=0,
        help='Máximo por clase (0 = todas, default: 0)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_raw, exist_ok=True)
    os.makedirs(args.output_text, exist_ok=True)
    os.makedirs(args.resultados_dir, exist_ok=True)

    print("=" * 70)
    print("  EXTRACCIÓN DE TEXTO OCR CON PADDLEOCR")
    print("=" * 70)
    print(f"  Dataset:         {args.dataset_dir}")
    print(f"  Salida crudo:    {args.output_raw}")
    print(f"  Salida procesado:{args.output_text}")
    print(f"  Workers:         {args.workers}")
    print(f"  Máx por clase:   {'todas' if args.max_per_class == 0 else args.max_per_class}")
    print("=" * 70)

    # Probar que PaddleOCR se puede importar
    try:
        from paddleocr import PaddleOCR
        print("\n[OK] Dependencia de PaddleOCR importada correctamente.")
    except ImportError:
        print("\n[ERROR FATAL] La librería 'paddleocr' no está instalada en tu entorno Python.")
        print("  Puedes instalarla con: pip install paddleocr paddlepaddle")
        sys.exit(1)

    # Recopilar tareas
    print("\n[INFO] Escaneando imágenes en el dataset...")
    tareas = []
    conteo_por_clase = {}

    for clase in CLASES:
        dir_clase = Path(args.dataset_dir) / clase
        if not dir_clase.exists():
            continue

        imagenes = sorted(dir_clase.glob('*.jpg'))
        if args.max_per_class > 0:
            imagenes = imagenes[:args.max_per_class]

        conteo_por_clase[clase] = len(imagenes)

        for img_path in imagenes:
            nombre_base = img_path.stem
            ruta_raw = Path(args.output_raw) / clase / f"{nombre_base}.txt"
            ruta_procesado = Path(args.output_text) / clase / f"{nombre_base}.txt"

            # Sobrescribir siempre
            tareas.append((str(img_path), str(ruta_raw), str(ruta_procesado)))

    total_imagenes = len(tareas)
    print(f"\n[INFO] Total de imágenes a procesar (sobrescribiendo existentes): {total_imagenes}")
    for clase, conteo in conteo_por_clase.items():
        print(f"  - {clase:30s}: {conteo:5d} imágenes")

    if total_imagenes == 0:
        print("\n[INFO] No hay imágenes pendientes de procesamiento.")
        sys.exit(0)

    # Ejecutar con ProcessPoolExecutor
    print(f"\n[INFO] Iniciando procesamiento con {args.workers} workers...")
    inicio = time.time()

    resultados_todos = []
    resultados_por_clase = {clase: [] for clase in CLASES}

    # Nota: initializer=init_paddle_ocr carga el modelo antes en cada proceso
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_paddle_ocr) as executor:
        futuros = {executor.submit(procesar_archivo, tarea): tarea for tarea in tareas}

        with tqdm(total=total_imagenes, desc="PaddleOCR", unit="img") as pbar:
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

    # Estadísticas
    total_exitosos = sum(1 for r in resultados_todos if r['exito'])
    total_fallidos = total_imagenes - total_exitosos
    tasa_exito = (total_exitosos / total_imagenes) * 100 if total_imagenes > 0 else 0

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
        else:
            avg_len_raw = avg_len_proc = 0

        estadisticas_clase[clase] = {
            'total': n_total,
            'exitosos': n_exitosos,
            'tasa_exito': (n_exitosos / n_total) * 100 if n_total > 0 else 0,
            'avg_len_raw': avg_len_raw,
            'avg_len_proc': avg_len_proc
        }

    # Guardar reporte de estadísticas
    ruta_stats = Path(args.resultados_dir) / '03_ocr_estadisticas_paddle.txt'
    with open(ruta_stats, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  ESTADÍSTICAS DE EXTRACCIÓN OCR CON PADDLEOCR\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha de ejecución:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tiempo total:           {tiempo_total:.1f} segundos ({tiempo_total/60:.1f} min)\n")
        f.write(f"Workers utilizados:     {args.workers}\n")
        f.write(f"Velocidad promedio:     {total_imagenes/tiempo_total:.2f} imágenes/segundo\n\n")

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
        f.write(f"{'TOTAL':<30s} {total_imagenes:>6d} {total_exitosos:>6d} {tasa_exito:>6.1f}%\n")
        f.write("=" * 70 + "\n")

    print(f"  [OK] Estadísticas guardadas en: {ruta_stats}")

    # Imprimir en consola
    print("\n" + "=" * 70)
    print("  RESUMEN DE EXTRACCIÓN CON PADDLEOCR")
    print("=" * 70)
    print(f"  Imágenes procesadas:   {total_imagenes}")
    print(f"  Extracciones exitosas: {total_exitosos} ({tasa_exito:.1f}%)")
    print(f"  Tiempo total:          {tiempo_total:.1f}s")
    print(f"  Velocidad:             {total_imagenes/tiempo_total:.2f} img/s")
    print("=" * 70)

    # Generar gráfica de ejemplos
    ruta_ejemplos = Path(args.resultados_dir) / '03_ejemplos_ocr_paddle.png'
    generar_ejemplos_ocr(resultados_por_clase, str(ruta_ejemplos))

    print("\n[INFO] ¡Extracción con PaddleOCR completada con éxito!")
    print(f"  - Texto crudo reconstruido en: {args.output_raw}/")
    print(f"  - Texto preprocesado en:        {args.output_text}/")

if __name__ == '__main__':
    main()

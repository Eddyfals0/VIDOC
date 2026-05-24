#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
  04_tfidf_representacion.py
  Proyecto Final - Recuperación de Información (8vo Semestre)
=============================================================================
  Descripción:
    Construye la representación TF-IDF de los textos extraídos por OCR,
    siguiendo exactamente el patrón del Laboratorio 5:
      - Construcción manual del vocabulario
      - Cálculo de la matriz TF (frecuencia de términos)
      - Cálculo del vector IDF: IDF(t) = log10(N / DF(t)) + 1
      - Cálculo de la matriz TF-IDF: TF × IDF (element-wise)
      - Comparación con TfidfVectorizer de sklearn

  Entradas:
    - dataset_text/{clase}/*.txt  (textos preprocesados)

  Salidas:
    - features/vocabulario.txt       (vocabulario completo)
    - features/matriz_tf.npz         (matriz TF en formato sparse)
    - features/vector_idf.npy        (vector IDF)
    - features/matriz_tfidf.npz      (matriz TF-IDF manual, sparse)
    - features/tfidf_sklearn.npz     (matriz TF-IDF de sklearn, sparse)
    - features/labels_text.npy       (etiquetas de clase por documento)
    - resultados/04_top_terminos_tfidf.png  (visualización de términos top)

  Uso:
    python 04_tfidf_representacion.py
    python 04_tfidf_representacion.py --input_dir dataset_text --max_features 5000
=============================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# Clases del proyecto RVL-CDIP reducido a 14 categorias descargadas.
import os
from pathlib import Path

from project_config import classes_with_files

CLASES = classes_with_files("dataset_text", patterns=("*.txt",))
CLASSES = CLASES


# =============================================================================
#  FUNCIONES DE CARGA DE DATOS
# =============================================================================

def cargar_documentos(input_dir):
    """
    Carga todos los documentos de texto preprocesados del directorio de entrada.

    Args:
        input_dir (str): Directorio raíz con subcarpetas por clase

    Returns:
        tuple: (documentos, etiquetas, nombres_archivos)
            - documentos: lista de strings (texto de cada documento)
            - etiquetas: lista de strings (clase de cada documento)
            - nombres_archivos: lista de strings (ruta relativa)
    """
    documentos = []
    etiquetas = []
    nombres_archivos = []

    print("[INFO] Cargando documentos de texto preprocesados...")

    for clase in tqdm(CLASES, desc="Cargando clases"):
        dir_clase = Path(input_dir) / clase
        if not dir_clase.exists():
            print(f"  [ADVERTENCIA] No se encontró directorio: {dir_clase}")
            continue

        archivos = sorted(dir_clase.glob('*.txt'))

        for archivo in archivos:
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    texto = f.read().strip()

                # Solo incluir documentos con contenido
                if len(texto) > 0:
                    documentos.append(texto)
                    etiquetas.append(clase)
                    nombres_archivos.append(str(archivo.relative_to(input_dir)))
            except Exception as e:
                print(f"  [ERROR] No se pudo leer {archivo}: {e}")

    return documentos, etiquetas, nombres_archivos


# =============================================================================
#  FUNCIONES DE CONSTRUCCIÓN DE VOCABULARIO
# =============================================================================

def construir_vocabulario(documentos, min_df=2, max_df_ratio=0.95):
    """
    Construye el vocabulario a partir de todos los documentos.
    Filtra términos que aparecen en muy pocos o demasiados documentos.

    Args:
        documentos (list): Lista de textos preprocesados
        min_df (int): Frecuencia mínima de documento para incluir un término
        max_df_ratio (float): Proporción máxima de documentos (0-1)

    Returns:
        tuple: (vocabulario, term2idx)
            - vocabulario: lista de términos ordenados
            - term2idx: dict término → índice
    """
    print("\n[INFO] Construyendo vocabulario...")
    N = len(documentos)
    max_df = int(N * max_df_ratio)

    # Contar frecuencia de documento (DF) para cada término
    df_counter = Counter()
    for doc in tqdm(documentos, desc="Contando DF"):
        # Usar set() para contar cada término una sola vez por documento
        tokens_unicos = set(doc.split())
        df_counter.update(tokens_unicos)

    # Filtrar términos por frecuencia de documento
    vocabulario = sorted([
        termino for termino, df in df_counter.items()
        if min_df <= df <= max_df
    ])

    # Crear mapeo término → índice
    term2idx = {termino: idx for idx, termino in enumerate(vocabulario)}

    print(f"  Términos totales encontrados: {len(df_counter)}")
    print(f"  Filtrados (min_df={min_df}, max_df_ratio={max_df_ratio}): {len(df_counter) - len(vocabulario)}")
    print(f"  Vocabulario final: {len(vocabulario)} términos")

    return vocabulario, term2idx


# =============================================================================
#  FUNCIONES DE CÁLCULO TF-IDF (MANUAL, siguiendo Lab 5)
# =============================================================================

def calcular_tf(documentos, term2idx):
    """
    Calcula la matriz de Frecuencia de Términos (TF).
    TF(t,d) = frecuencia del término t en el documento d

    Siguiendo el patrón del Laboratorio 5.

    Args:
        documentos (list): Lista de textos
        term2idx (dict): Mapeo término → índice

    Returns:
        scipy.sparse.csr_matrix: Matriz TF de forma (n_docs, n_terms)
    """
    print("\n[INFO] Calculando matriz TF (frecuencia de términos)...")

    n_docs = len(documentos)
    n_terms = len(term2idx)

    # Construir matriz sparse usando listas de coordenadas (COO)
    filas = []
    columnas = []
    valores = []

    for doc_idx, doc in enumerate(tqdm(documentos, desc="Calculando TF")):
        # Contar frecuencia de cada término en el documento
        token_counts = Counter(doc.split())

        for termino, freq in token_counts.items():
            if termino in term2idx:
                filas.append(doc_idx)
                columnas.append(term2idx[termino])
                valores.append(freq)

    # Crear matriz sparse en formato COO y convertir a CSR
    tf_matrix = sparse.coo_matrix(
        (valores, (filas, columnas)),
        shape=(n_docs, n_terms),
        dtype=np.float64
    ).tocsr()

    return tf_matrix


def calcular_idf(documentos, term2idx):
    """
    Calcula el vector IDF (Inverse Document Frequency).
    Fórmula del Lab 5: IDF(t) = log10(N / DF(t)) + 1

    Args:
        documentos (list): Lista de textos
        term2idx (dict): Mapeo término → índice

    Returns:
        numpy.ndarray: Vector IDF de longitud n_terms
    """
    print("\n[INFO] Calculando vector IDF...")

    N = len(documentos)
    n_terms = len(term2idx)
    df = np.zeros(n_terms)

    # Calcular DF (Document Frequency) para cada término
    for doc in tqdm(documentos, desc="Calculando DF"):
        tokens_unicos = set(doc.split())
        for termino in tokens_unicos:
            if termino in term2idx:
                df[term2idx[termino]] += 1

    # Calcular IDF: log10(N / DF(t)) + 1
    # Evitar división por cero (aunque no debería ocurrir por el filtro de min_df)
    idf = np.log10(N / np.maximum(df, 1)) + 1

    return idf


def calcular_tfidf(tf_matrix, idf_vector):
    """
    Calcula la matriz TF-IDF multiplicando TF × IDF element-wise.
    Siguiendo el Lab 5: TF-IDF(t,d) = TF(t,d) × IDF(t)

    Args:
        tf_matrix (scipy.sparse.csr_matrix): Matriz TF
        idf_vector (numpy.ndarray): Vector IDF

    Returns:
        scipy.sparse.csr_matrix: Matriz TF-IDF
    """
    print("\n[INFO] Calculando matriz TF-IDF (TF × IDF)...")

    # Multiplicar cada columna de TF por su valor IDF correspondiente
    # Usar diag sparse para eficiencia
    idf_diag = sparse.diags(idf_vector)
    tfidf_matrix = tf_matrix.dot(idf_diag)

    return tfidf_matrix.tocsr()


# =============================================================================
#  FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# =============================================================================

def obtener_top_terminos_por_clase(tfidf_matrix, etiquetas, vocabulario, top_n=10):
    """
    Obtiene los top-N términos por TF-IDF promedio para cada clase.

    Args:
        tfidf_matrix: Matriz TF-IDF (sparse)
        etiquetas (list): Etiquetas de clase por documento
        vocabulario (list): Lista de términos
        top_n (int): Número de términos top a extraer

    Returns:
        dict: {clase: [(termino, score), ...]}
    """
    print("\n[INFO] Calculando top términos por clase...")

    top_por_clase = {}
    etiquetas_array = np.array(etiquetas)

    for clase in tqdm(CLASES, desc="Analizando clases"):
        # Obtener índices de documentos de esta clase
        indices = np.where(etiquetas_array == clase)[0]

        if len(indices) == 0:
            continue

        # Calcular TF-IDF promedio para esta clase
        submatriz = tfidf_matrix[indices]
        promedio = np.asarray(submatriz.mean(axis=0)).flatten()

        # Obtener los top-N índices por score
        top_indices = promedio.argsort()[-top_n:][::-1]

        top_por_clase[clase] = [
            (vocabulario[idx], promedio[idx])
            for idx in top_indices
            if promedio[idx] > 0
        ]

    return top_por_clase


def visualizar_top_terminos(top_por_clase, ruta_salida):
    """
    Genera una visualización de los top términos TF-IDF por clase.

    Args:
        top_por_clase (dict): Resultado de obtener_top_terminos_por_clase
        ruta_salida (str): Ruta donde guardar la imagen
    """
    print("\n[INFO] Generando visualización de top términos...")

    clases_con_datos = [c for c in CLASES if c in top_por_clase and len(top_por_clase[c]) > 0]
    n_clases = len(clases_con_datos)

    if n_clases == 0:
        print("  [ADVERTENCIA] No hay datos para visualizar")
        return

    # Configurar la grilla de subplots (4 columnas)
    n_cols = 4
    n_filas = (n_clases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_filas, n_cols, figsize=(20, 4 * n_filas))
    fig.suptitle(
        'Top 10 Términos por TF-IDF Promedio por Clase',
        fontsize=16, fontweight='bold', y=1.02
    )

    # Aplanar los axes para iterar fácilmente
    if n_filas == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_filas == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    # Paleta de colores para las barras
    colores = plt.cm.Set3(np.linspace(0, 1, n_clases))

    for idx, clase in enumerate(clases_con_datos):
        ax = axes_flat[idx]
        terminos_scores = top_por_clase[clase]

        if not terminos_scores:
            ax.set_visible(False)
            continue

        terminos = [t for t, _ in terminos_scores]
        scores = [s for _, s in terminos_scores]

        # Gráfica de barras horizontales (más legible para texto)
        barras = ax.barh(
            range(len(terminos)), scores,
            color=colores[idx], edgecolor='gray', alpha=0.85
        )

        ax.set_yticks(range(len(terminos)))
        ax.set_yticklabels(terminos, fontsize=8)
        ax.invert_yaxis()  # Mayor score arriba
        ax.set_xlabel('TF-IDF Promedio', fontsize=8)
        ax.set_title(clase.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7)

    # Ocultar axes vacíos
    for idx in range(n_clases, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Visualización guardada en: {ruta_salida}")


def imprimir_estadisticas(tf_matrix, idf_vector, tfidf_matrix, tfidf_sklearn,
                          vocabulario, etiquetas, top_por_clase):
    """
    Imprime estadísticas detalladas de las matrices generadas.
    """
    print("\n" + "=" * 70)
    print("  ESTADÍSTICAS DE REPRESENTACIÓN TF-IDF")
    print("=" * 70)

    n_docs, n_terms = tf_matrix.shape
    print(f"\n  Documentos totales:     {n_docs}")
    print(f"  Tamaño del vocabulario: {n_terms}")

    # Distribución de clases
    print(f"\n  Distribución de clases:")
    conteo_clases = Counter(etiquetas)
    for clase in CLASES:
        if clase in conteo_clases:
            print(f"    {clase:<30s}: {conteo_clases[clase]:5d} documentos")

    # Esparsidad de matrices
    nnz_tf = tf_matrix.nnz
    total_tf = n_docs * n_terms
    esparsidad_tf = (1 - nnz_tf / total_tf) * 100 if total_tf > 0 else 0

    nnz_tfidf = tfidf_matrix.nnz
    total_tfidf = n_docs * n_terms
    esparsidad_tfidf = (1 - nnz_tfidf / total_tfidf) * 100 if total_tfidf > 0 else 0

    nnz_sklearn = tfidf_sklearn.nnz
    total_sklearn = tfidf_sklearn.shape[0] * tfidf_sklearn.shape[1]
    esparsidad_sklearn = (1 - nnz_sklearn / total_sklearn) * 100 if total_sklearn > 0 else 0

    print(f"\n  Matriz TF:")
    print(f"    Forma:       {tf_matrix.shape}")
    print(f"    No-ceros:    {nnz_tf:,}")
    print(f"    Esparsidad:  {esparsidad_tf:.2f}%")

    print(f"\n  Vector IDF:")
    print(f"    Longitud:    {len(idf_vector)}")
    print(f"    Min:         {idf_vector.min():.4f}")
    print(f"    Max:         {idf_vector.max():.4f}")
    print(f"    Media:       {idf_vector.mean():.4f}")

    print(f"\n  Matriz TF-IDF (manual):")
    print(f"    Forma:       {tfidf_matrix.shape}")
    print(f"    No-ceros:    {nnz_tfidf:,}")
    print(f"    Esparsidad:  {esparsidad_tfidf:.2f}%")

    print(f"\n  Matriz TF-IDF (sklearn):")
    print(f"    Forma:       {tfidf_sklearn.shape}")
    print(f"    No-ceros:    {nnz_sklearn:,}")
    print(f"    Esparsidad:  {esparsidad_sklearn:.2f}%")

    # Top términos por clase
    print(f"\n  Top 10 términos TF-IDF por clase:")
    print("  " + "-" * 65)
    for clase in CLASES:
        if clase in top_por_clase and len(top_por_clase[clase]) > 0:
            terminos_str = ', '.join(
                f"{t}({s:.3f})" for t, s in top_por_clase[clase][:10]
            )
            print(f"  {clase:<25s}: {terminos_str}")

    print("=" * 70)


# =============================================================================
#  FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal: construye la representación TF-IDF del corpus.
    """
    # --- Argumentos de línea de comandos ---
    parser = argparse.ArgumentParser(
        description='Construcción de representación TF-IDF para documentos OCR (Lab 5)'
    )
    parser.add_argument(
        '--input_dir', type=str, default='dataset_text',
        help='Directorio con textos preprocesados (default: dataset_text)'
    )
    parser.add_argument(
        '--features_dir', type=str, default='features',
        help='Directorio de salida para features (default: features)'
    )
    parser.add_argument(
        '--resultados_dir', type=str, default='resultados',
        help='Directorio para visualizaciones (default: resultados)'
    )
    parser.add_argument(
        '--max_features', type=int, default=5000,
        help='Máximo de features para TfidfVectorizer de sklearn (default: 5000)'
    )
    parser.add_argument(
        '--min_df', type=int, default=2,
        help='Frecuencia mínima de documento para incluir un término (default: 2)'
    )
    parser.add_argument(
        '--max_df_ratio', type=float, default=0.95,
        help='Proporción máxima de documentos para un término (default: 0.95)'
    )
    parser.add_argument(
        '--top_n', type=int, default=10,
        help='Número de términos top por clase a mostrar (default: 10)'
    )
    args = parser.parse_args()

    # --- Crear directorios de salida ---
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.resultados_dir, exist_ok=True)

    print("=" * 70)
    print("  CONSTRUCCIÓN DE REPRESENTACIÓN TF-IDF")
    print("  (Siguiendo el patrón del Laboratorio 5)")
    print("=" * 70)
    print(f"  Entrada:         {args.input_dir}")
    print(f"  Features:        {args.features_dir}")
    print(f"  Max features:    {args.max_features}")
    print(f"  Min DF:          {args.min_df}")
    print(f"  Max DF ratio:    {args.max_df_ratio}")
    print("=" * 70)

    inicio_total = time.time()

    # =========================================================================
    #  PASO 1: Cargar documentos
    # =========================================================================
    documentos, etiquetas, nombres = cargar_documentos(args.input_dir)

    if len(documentos) == 0:
        print("\n[ERROR FATAL] No se encontraron documentos con texto.")
        print("  Asegúrate de haber ejecutado primero: python 03_ocr_extraccion_texto.py")
        sys.exit(1)

    print(f"\n  Documentos cargados: {len(documentos)}")
    print(f"  Clases encontradas: {len(set(etiquetas))}")

    # =========================================================================
    #  PASO 2: Construir vocabulario
    # =========================================================================
    vocabulario, term2idx = construir_vocabulario(
        documentos,
        min_df=args.min_df,
        max_df_ratio=args.max_df_ratio
    )

    # Guardar vocabulario
    ruta_vocab = Path(args.features_dir) / 'vocabulario.txt'
    with open(ruta_vocab, 'w', encoding='utf-8') as f:
        for termino in vocabulario:
            f.write(termino + '\n')
    print(f"  [OK] Vocabulario guardado en: {ruta_vocab}")

    # =========================================================================
    #  PASO 3: Calcular matriz TF
    # =========================================================================
    tf_matrix = calcular_tf(documentos, term2idx)

    # Guardar matriz TF en formato sparse
    ruta_tf = Path(args.features_dir) / 'matriz_tf.npz'
    sparse.save_npz(str(ruta_tf), tf_matrix)
    print(f"  [OK] Matriz TF guardada en: {ruta_tf} (forma: {tf_matrix.shape})")

    # =========================================================================
    #  PASO 4: Calcular vector IDF
    # =========================================================================
    idf_vector = calcular_idf(documentos, term2idx)

    # Guardar vector IDF
    ruta_idf = Path(args.features_dir) / 'vector_idf.npy'
    np.save(str(ruta_idf), idf_vector)
    print(f"  [OK] Vector IDF guardado en: {ruta_idf} (longitud: {len(idf_vector)})")

    # =========================================================================
    #  PASO 5: Calcular matriz TF-IDF (manual)
    # =========================================================================
    tfidf_matrix = calcular_tfidf(tf_matrix, idf_vector)

    # Guardar matriz TF-IDF manual
    ruta_tfidf = Path(args.features_dir) / 'matriz_tfidf.npz'
    sparse.save_npz(str(ruta_tfidf), tfidf_matrix)
    print(f"  [OK] Matriz TF-IDF (manual) guardada en: {ruta_tfidf}")

    # =========================================================================
    #  PASO 6: TF-IDF con sklearn (para comparación)
    # =========================================================================
    print("\n[INFO] Calculando TF-IDF con sklearn TfidfVectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        sublinear_tf=False,         # Sin transformación logarítmica del TF
        smooth_idf=False,           # Sin suavizado para coincidir con fórmula del lab
        norm=None                   # Sin normalización L2 para comparar directamente
    )

    tfidf_sklearn = vectorizer.fit_transform(documentos)

    # Guardar matriz TF-IDF de sklearn
    ruta_sklearn = Path(args.features_dir) / 'tfidf_sklearn.npz'
    sparse.save_npz(str(ruta_sklearn), tfidf_sklearn)
    print(f"  [OK] Matriz TF-IDF (sklearn) guardada en: {ruta_sklearn}")
    print(f"       Forma: {tfidf_sklearn.shape}")
    print(f"       Features de sklearn: {len(vectorizer.get_feature_names_out())}")

    # =========================================================================
    #  PASO 7: Guardar etiquetas
    # =========================================================================
    ruta_labels = Path(args.features_dir) / 'labels_text.npy'
    np.save(str(ruta_labels), np.array(etiquetas))
    print(f"  [OK] Etiquetas guardadas en: {ruta_labels}")

    ruta_names = Path(args.features_dir) / 'file_names_text.npy'
    np.save(str(ruta_names), np.array(nombres))
    print(f"  [OK] Nombres de documentos guardados en: {ruta_names}")

    # =========================================================================
    #  PASO 8: Análisis de top términos por clase
    # =========================================================================
    top_por_clase = obtener_top_terminos_por_clase(
        tfidf_matrix, etiquetas, vocabulario, top_n=args.top_n
    )

    # =========================================================================
    #  PASO 9: Visualización
    # =========================================================================
    ruta_viz = Path(args.resultados_dir) / '04_top_terminos_tfidf.png'
    visualizar_top_terminos(top_por_clase, str(ruta_viz))

    # =========================================================================
    #  PASO 10: Estadísticas
    # =========================================================================
    imprimir_estadisticas(
        tf_matrix, idf_vector, tfidf_matrix, tfidf_sklearn,
        vocabulario, etiquetas, top_por_clase
    )

    # --- Tiempo total ---
    tiempo_total = time.time() - inicio_total
    print(f"\n[INFO] ¡Representación TF-IDF completada en {tiempo_total:.1f} segundos!")
    print(f"  - Vocabulario:       {ruta_vocab}")
    print(f"  - Matriz TF:         {ruta_tf}")
    print(f"  - Vector IDF:        {ruta_idf}")
    print(f"  - Matriz TF-IDF:     {ruta_tfidf}")
    print(f"  - TF-IDF sklearn:    {ruta_sklearn}")
    print(f"  - Etiquetas:         {ruta_labels}")
    print(f"  - Visualización:     {ruta_viz}")


# =============================================================================
#  PUNTO DE ENTRADA
# =============================================================================

if __name__ == '__main__':
    main()

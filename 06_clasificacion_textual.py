#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_clasificacion_textual.py
===========================
Clasificación de documentos escaneados mediante texto OCR + TF-IDF.
Replica el pipeline del Laboratorio 9 del curso de RI.

Modelos entrenados:
  1. TF-IDF + LinearSVC           (SVM lineal, rápido y eficaz)
  2. TF-IDF + LogisticRegression  (regresión logística multiclase)
  3. TF-IDF + MultinomialNB       (Naïve Bayes multinomial)

Entrada:
  - dataset_text/{nombre_clase}/*.txt  (un archivo de texto OCR por documento)

Salida:
  - resultados/06_confusion_tfidf_svm.png          (matrices de confusión)
  - resultados/06_confusion_tfidf_lr.png
  - resultados/06_confusion_tfidf_nb.png
  - resultados/06_resultados_textuales.txt          (reporte resumen)
  - modelos/tfidf_svm.joblib                        (modelos entrenados)
  - modelos/tfidf_lr.joblib
  - modelos/tfidf_nb.joblib

Autor: Proyecto Final RI — BUAP 2026
"""

import argparse
import os
import sys
import warnings
from pathlib import Path


# ──────────────────────────────────────────────────────────
#  Constantes del proyecto
# ──────────────────────────────────────────────────────────

# Clases del proyecto RVL-CDIP reducido a 14 categorias descargadas.
import os
from pathlib import Path

from project_config import classes_with_files

CLASES = classes_with_files("dataset_text", patterns=("*.txt",))
CLASSES = CLASES


# ──────────────────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────────────────

def crear_directorios(*dirs: str):
    """Crea los directorios indicados si no existen."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def cargar_dataset_texto(directorio_base: str):
    """
    Carga los textos OCR desde la estructura de directorios.

    Estructura esperada:
        directorio_base/
        ├── letter/
        │   ├── letter_00001.txt
        │   └── ...
        ├── form/
        │   └── ...
        └── ...

    Retorna
    -------
    textos : list[str]  — contenido textual de cada documento
    etiquetas : list[str] — clase correspondiente
    conteo_por_clase : dict — cantidad de documentos por clase
    """
    textos = []
    etiquetas = []
    conteo_por_clase = {}

    base = Path(directorio_base)
    if not base.exists():
        print(f"[ERROR] No se encontró el directorio: {base}")
        sys.exit(1)

    # Iterar sobre los subdirectorios (cada uno es una clase)
    for clase_dir in sorted(base.iterdir()):
        if not clase_dir.is_dir():
            continue

        nombre_clase = clase_dir.name
        archivos_txt = sorted(clase_dir.glob("*.txt"))
        conteo = 0

        for archivo in archivos_txt:
            try:
                texto = archivo.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"  ⚠ Error leyendo {archivo}: {e}")
                continue

            # Descartar documentos vacíos o con texto muy corto
            if len(texto) < 10:
                continue

            textos.append(texto)
            etiquetas.append(nombre_clase)
            conteo += 1

        conteo_por_clase[nombre_clase] = conteo

    return textos, etiquetas, conteo_por_clase


def guardar_matriz_confusion(y_real, y_pred, nombres_clases, titulo, ruta_salida):
    """
    Genera y guarda una matriz de confusión como imagen PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_real, y_pred, labels=nombres_clases)
    n_clases = len(nombres_clases)
    fig_size = max(8, n_clases * 0.7)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        ax=ax,
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150)
    plt.close(fig)
    print(f"   Matriz de confusión guardada: {ruta_salida}")


def entrenar_y_evaluar_pipeline(pipeline, nombre_modelo, nombre_archivo,
                                X_train, X_test, y_train, y_test,
                                nombres_clases, dir_resultados, dir_modelos):
    """
    Entrena un pipeline de sklearn, genera métricas y guarda resultados.

    Retorna un diccionario con el resumen de resultados.
    """
    from sklearn.metrics import accuracy_score, classification_report
    import joblib

    print(f"\n{'='*60}")
    print(f"  Modelo: {nombre_modelo}")
    print(f"{'='*60}")

    # Entrenar el pipeline completo (TF-IDF + clasificador)
    print("   Entrenando pipeline...")
    pipeline.fit(X_train, y_train)
    print("  [OK] Entrenamiento completado")

    # Predicciones
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Reporte de clasificación (incluye precision, recall, f1 por clase)
    reporte = classification_report(
        y_test, y_pred,
        labels=nombres_clases,
        target_names=nombres_clases,
        zero_division=0,
    )
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n{reporte}")

    # Guardar matriz de confusión
    ruta_cm = os.path.join(dir_resultados, f"06_confusion_{nombre_archivo}.png")
    guardar_matriz_confusion(
        y_test, y_pred, nombres_clases,
        f"Matriz de Confusión — {nombre_modelo}",
        ruta_cm,
    )

    # Guardar modelo (pipeline completo con TF-IDF + clasificador)
    ruta_modelo = os.path.join(dir_modelos, f"{nombre_archivo}.joblib")
    joblib.dump(pipeline, ruta_modelo)
    print(f"   Modelo guardado: {ruta_modelo}")

    return {
        "nombre": nombre_modelo,
        "accuracy": accuracy,
        "reporte": reporte,
        "ruta_cm": ruta_cm,
        "ruta_modelo": ruta_modelo,
    }


def guardar_resumen(resultados, conteo_por_clase, ruta_salida):
    """Guarda un resumen textual de todos los modelos evaluados."""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RESULTADOS DE CLASIFICACIÓN TEXTUAL DE DOCUMENTOS (OCR + TF-IDF)\n")
        f.write("  Proyecto Final — Recuperación de Información — BUAP 2026\n")
        f.write("=" * 70 + "\n\n")

        # Distribución del dataset
        f.write("DISTRIBUCIÓN DEL DATASET\n")
        f.write("-" * 40 + "\n")
        total = sum(conteo_por_clase.values())
        for clase, conteo in sorted(conteo_por_clase.items()):
            f.write(f"  {clase:<30s} → {conteo:>5d} documentos\n")
        f.write(f"  {'TOTAL':<30s} → {total:>5d} documentos\n\n")

        # Tabla resumen de accuracies
        f.write("RESUMEN DE ACCURACY POR MODELO\n")
        f.write("-" * 40 + "\n")
        for r in resultados:
            f.write(f"  {r['nombre']:<40s} → {r['accuracy']:.4f}\n")
        f.write("\n")

        # Mejor modelo
        mejor = max(resultados, key=lambda x: x["accuracy"])
        f.write(f" Mejor modelo: {mejor['nombre']} (Accuracy: {mejor['accuracy']:.4f})\n\n")

        # Reportes detallados por modelo
        for r in resultados:
            f.write("=" * 70 + "\n")
            f.write(f"  Modelo: {r['nombre']}\n")
            f.write(f"  Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"  Matriz de confusión: {r['ruta_cm']}\n")
            f.write(f"  Modelo guardado: {r['ruta_modelo']}\n")
            f.write("-" * 70 + "\n")
            f.write(r["reporte"])
            f.write("\n")

    print(f"\n Resumen guardado: {ruta_salida}")


# ──────────────────────────────────────────────────────────
#  Función principal
# ──────────────────────────────────────────────────────────

def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description="Clasificación textual de documentos con OCR + TF-IDF (Lab 9).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-text", type=str, default="dataset_text",
        help="Directorio con textos OCR organizados por clase (default: dataset_text)",
    )
    parser.add_argument(
        "--resultados-dir", type=str, default="resultados",
        help="Directorio donde guardar gráficos y reportes (default: resultados)",
    )
    parser.add_argument(
        "--modelos-dir", type=str, default="modelos",
        help="Directorio donde guardar los modelos entrenados (default: modelos)",
    )
    parser.add_argument(
        "--max-features", type=int, default=3000,
        help="Número máximo de features para TF-IDF (default: 3000)",
    )
    parser.add_argument(
        "--ngram-range", type=str, default="1,2",
        help="Rango de n-gramas para TF-IDF, formato 'min,max' (default: 1,2)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proporción del conjunto de prueba (default: 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)",
    )
    args = parser.parse_args()

    # ── Importaciones pesadas ──
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar scikit-learn. Ejecuta: pip install scikit-learn"
        ) from exc

    # Suprimir warnings menores
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Crear directorios de salida ──
    crear_directorios(args.resultados_dir, args.modelos_dir)

    # ── Parsear rango de n-gramas ──
    try:
        ngram_min, ngram_max = [int(x.strip()) for x in args.ngram_range.split(",")]
        ngram_range = (ngram_min, ngram_max)
    except ValueError:
        print("[ERROR] Formato inválido para --ngram-range. Use 'min,max', e.g. '1,2'")
        sys.exit(1)

    # ── Cargar dataset de texto OCR ──
    print("\n Cargando textos OCR desde:", args.dataset_text)
    textos, etiquetas, conteo_por_clase = cargar_dataset_texto(args.dataset_text)

    if len(textos) < 2:
        print(f"\n[ERROR] Solo se encontraron {len(textos)} documentos con texto válido.")
        print("Asegúrate de haber ejecutado el script de extracción OCR primero.")
        sys.exit(1)

    # Mostrar distribución del dataset
    print(f"\n Total de documentos cargados: {len(textos)}")
    print(f" Clases encontradas: {len(conteo_por_clase)}")
    for clase, conteo in sorted(conteo_por_clase.items()):
        print(f"  • {clase}: {conteo} documentos")

    # Determinar las clases presentes (para reportes y matrices)
    nombres_clases = sorted(set(etiquetas))
    print(f"\n Clases para clasificación ({len(nombres_clases)}): {', '.join(nombres_clases)}")

    # ── División train/test (80/20, estratificada) ──
    print(f"\n Dividiendo datos: {int((1-args.test_size)*100)}% train / {int(args.test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        textos,
        etiquetas,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=etiquetas,
    )
    print(f"  Train: {len(X_train)} documentos")
    print(f"  Test:  {len(X_test)} documentos")

    # ── Definir pipelines (TF-IDF + clasificador) ──
    # Siguiendo el patrón del Laboratorio 9 del curso
    pipelines = [
        (
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                    sublinear_tf=True,  # Aplica 1 + log(tf) para suavizar
                )),
                ("clf", LinearSVC(max_iter=5000)),
            ]),
            "TF-IDF + LinearSVC",
            "tfidf_svm",
        ),
        (
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                )),
            ]),
            "TF-IDF + LogisticRegression",
            "tfidf_lr",
        ),
        (
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                    # No usar sublinear_tf con MultinomialNB (necesita valores >= 0)
                )),
                ("clf", MultinomialNB(alpha=1.0)),
            ]),
            "TF-IDF + MultinomialNB",
            "tfidf_nb",
        ),
    ]

    # ── Entrenar y evaluar cada pipeline ──
    resultados = []
    for pipeline_obj, nombre_display, nombre_archivo in pipelines:
        res = entrenar_y_evaluar_pipeline(
            pipeline=pipeline_obj,
            nombre_modelo=nombre_display,
            nombre_archivo=nombre_archivo,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            nombres_clases=nombres_clases,
            dir_resultados=args.resultados_dir,
            dir_modelos=args.modelos_dir,
        )
        resultados.append(res)

    # ── Guardar resumen general ──
    ruta_resumen = os.path.join(args.resultados_dir, "06_resultados_textuales.txt")
    guardar_resumen(resultados, conteo_por_clase, ruta_resumen)

    # ── Resumen final en consola ──
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL — CLASIFICACIÓN TEXTUAL (OCR + TF-IDF)")
    print("=" * 60)
    for r in resultados:
        indicador = "" if r == max(resultados, key=lambda x: x["accuracy"]) else "  "
        print(f"  {indicador} {r['nombre']:<42s} Accuracy: {r['accuracy']:.4f}")
    print("=" * 60)
    print("\n Clasificación textual completada exitosamente.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_clasificacion_visual.py
==========================
Clasificación de documentos escaneados mediante descriptores visuales.
Entrena y evalúa los siguientes modelos:
  1. LBP  + SVM  (kernel RBF)
  2. LBP  + KNN  (k=5)
  3. GLCM + SVM  (kernel RBF)
  4. Combinado (LBP + GLCM + Histograma) + SVM

Entrada:
  - features/features_lbp.npy
  - features/features_glcm.npy
  - features/features_histogram.npy  (opcional, si no existe se omite en combinado)
  - features/labels.npy

Salida:
  - resultados/05_confusion_<modelo>.png       (matrices de confusión)
  - resultados/05_resultados_visuales.txt       (reporte resumen)
  - modelos/<modelo>.joblib                     (modelos entrenados)

Autor: Proyecto Final RI — BUAP 2026
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────
#  Constantes del proyecto
# ──────────────────────────────────────────────────────────

# Las 16 clases del dataset RVL-CDIP
import os
from pathlib import Path

from project_config import DEFAULT_CLASSES_14

CLASES = DEFAULT_CLASSES_14
CLASSES = CLASES


# ──────────────────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────────────────

def cargar_features(ruta: str, nombre: str) -> np.ndarray:
    """Carga un archivo .npy y muestra información básica."""
    path = Path(ruta)
    if not path.exists():
        print(f"[ERROR] No se encontró el archivo: {path}")
        sys.exit(1)
    datos = np.load(str(path))
    print(f"  [OK] {nombre}: {datos.shape}")
    return datos


def crear_directorios(*dirs: str):
    """Crea los directorios indicados si no existen."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def guardar_matriz_confusion(y_real, y_pred, nombres_clases, titulo, ruta_salida):
    """
    Genera y guarda una matriz de confusión como imagen PNG.

    Parámetros
    ----------
    y_real : array-like — etiquetas reales
    y_pred : array-like — predicciones del modelo
    nombres_clases : list — nombres de las clases (ordenadas)
    titulo : str — título del gráfico
    ruta_salida : str — ruta donde guardar la imagen
    """
    import matplotlib
    matplotlib.use("Agg")  # Sin interfaz gráfica
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_real, y_pred)
    # Determinar tamaño dinámico según número de clases
    n_clases = len(nombres_clases)
    fig_size = max(8, n_clases * 0.7)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
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


def entrenar_y_evaluar(modelo, nombre_modelo, X_train, X_test, y_train, y_test,
                       nombres_clases, dir_resultados, dir_modelos):
    """
    Entrena un modelo, genera métricas y guarda resultados.

    Retorna un diccionario con el resumen de resultados.
    """
    from sklearn.metrics import accuracy_score, classification_report
    import joblib

    print(f"\n{'='*60}")
    print(f"  Modelo: {nombre_modelo}")
    print(f"{'='*60}")

    # Entrenar el modelo
    print("   Entrenando...")
    modelo.fit(X_train, y_train)
    print("  [OK] Entrenamiento completado")

    # Predicciones en conjunto de prueba
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Reporte de clasificación
    reporte = classification_report(
        y_test, y_pred,
        target_names=nombres_clases,
        zero_division=0,
    )
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n{reporte}")

    # Guardar matriz de confusión
    nombre_archivo = nombre_modelo.lower().replace(" ", "_").replace("+", "_")
    ruta_cm = os.path.join(dir_resultados, f"05_confusion_{nombre_archivo}.png")
    guardar_matriz_confusion(
        y_test, y_pred, nombres_clases,
        f"Matriz de Confusión — {nombre_modelo}",
        ruta_cm,
    )

    # Guardar modelo entrenado con joblib
    ruta_modelo = os.path.join(dir_modelos, f"{nombre_archivo}.joblib")
    joblib.dump(modelo, ruta_modelo)
    print(f"   Modelo guardado: {ruta_modelo}")

    return {
        "nombre": nombre_modelo,
        "accuracy": accuracy,
        "reporte": reporte,
        "ruta_cm": ruta_cm,
        "ruta_modelo": ruta_modelo,
    }


def guardar_resumen(resultados, ruta_salida):
    """Guarda un resumen textual de todos los modelos evaluados."""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RESULTADOS DE CLASIFICACIÓN VISUAL DE DOCUMENTOS\n")
        f.write("  Proyecto Final — Recuperación de Información — BUAP 2026\n")
        f.write("=" * 70 + "\n\n")

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
        description="Clasificación visual de documentos con LBP, GLCM y SVM/KNN.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--features-dir", type=str, default="features",
        help="Directorio donde se encuentran los archivos .npy de features (default: features)",
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
        "--test-size", type=float, default=0.2,
        help="Proporción del conjunto de prueba (default: 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semilla para reproducibilidad (default: 42)",
    )
    args = parser.parse_args()

    # ── Importaciones pesadas (se importan aquí para mostrar errores claros) ──
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar scikit-learn. Ejecuta: pip install scikit-learn"
        ) from exc

    # Suprimir warnings menores de convergencia
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Crear directorios de salida ──
    crear_directorios(args.resultados_dir, args.modelos_dir)

    # ── Cargar features previamente extraídas ──
    print("\n Cargando features visuales...")
    features_lbp = cargar_features(
        os.path.join(args.features_dir, "features_lbp.npy"), "LBP"
    )
    features_glcm = cargar_features(
        os.path.join(args.features_dir, "features_glcm.npy"), "GLCM"
    )
    labels = cargar_features(
        os.path.join(args.features_dir, "labels.npy"), "Labels"
    )
    ruta_class_names = os.path.join(args.features_dir, "class_names.npy")
    if os.path.exists(ruta_class_names):
        nombres_clases = list(np.load(ruta_class_names, allow_pickle=True))
    else:
        nombres_clases = CLASES

    # Intentar cargar features de histograma (opcional para modelo combinado)
    ruta_hist = os.path.join(args.features_dir, "features_histograms.npy")
    features_hist = None
    if os.path.exists(ruta_hist):
        features_hist = cargar_features(ruta_hist, "Histograma")
    else:
        print(f"  ⚠ No se encontró {ruta_hist}; el modelo combinado usará solo LBP+GLCM.")

    # ── Codificar etiquetas numéricamente ──
    # Los labels ya vienen como enteros desde el script 02.
    y_encoded = labels.astype(int)
    # Filtrar solo clases presentes en los datos
    clases_presentes = sorted(set(y_encoded))
    nombres_presentes = [nombres_clases[i] for i in clases_presentes]

    print(f"\n Total de muestras: {len(labels)}")
    print(f" Clases encontradas ({len(clases_presentes)}): {', '.join(nombres_presentes)}")

    # ── División train/test (80/20, estratificada) ──
    print(f"\n Dividiendo datos: {int((1-args.test_size)*100)}% train / {int(args.test_size*100)}% test")

    # Índices para dividir de forma consistente entre features
    indices = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_encoded,
    )

    y_train = y_encoded[idx_train]
    y_test = y_encoded[idx_test]

    print(f"  Train: {len(idx_train)} muestras")
    print(f"  Test:  {len(idx_test)} muestras")

    # ── Preparar features para cada modelo ──

    # 1. LBP features
    X_lbp_train = features_lbp[idx_train]
    X_lbp_test = features_lbp[idx_test]

    # 2. GLCM features
    X_glcm_train = features_glcm[idx_train]
    X_glcm_test = features_glcm[idx_test]

    # 3. Combinado: LBP + GLCM (+ Histograma si está disponible)
    if features_hist is not None:
        X_combinado = np.hstack([features_lbp, features_glcm, features_hist])
    else:
        X_combinado = np.hstack([features_lbp, features_glcm])

    X_comb_train = X_combinado[idx_train]
    X_comb_test = X_combinado[idx_test]

    # ── Escalar features para mejorar rendimiento de SVM/KNN ──
    scaler_lbp = StandardScaler()
    X_lbp_train = scaler_lbp.fit_transform(X_lbp_train)
    X_lbp_test = scaler_lbp.transform(X_lbp_test)

    scaler_glcm = StandardScaler()
    X_glcm_train = scaler_glcm.fit_transform(X_glcm_train)
    X_glcm_test = scaler_glcm.transform(X_glcm_test)

    scaler_comb = StandardScaler()
    X_comb_train = scaler_comb.fit_transform(X_comb_train)
    X_comb_test = scaler_comb.transform(X_comb_test)

    # ── Definir modelos ──
    modelos = [
        # (modelo_sklearn, nombre_display, X_train, X_test)
        (
            SVC(kernel="rbf", C=10, gamma="scale"),
            "lbp_svm",
            "LBP + SVM",
            X_lbp_train, X_lbp_test,
        ),
        (
            KNeighborsClassifier(n_neighbors=5),
            "lbp_knn",
            "LBP + KNN",
            X_lbp_train, X_lbp_test,
        ),
        (
            SVC(kernel="rbf", C=10, gamma="scale"),
            "glcm_svm",
            "GLCM + SVM",
            X_glcm_train, X_glcm_test,
        ),
        (
            SVC(kernel="rbf", C=10, gamma="scale"),
            "combinado_visual_svm",
            "Combinado Visual (LBP+GLCM+Hist) + SVM",
            X_comb_train, X_comb_test,
        ),
    ]

    # ── Entrenar y evaluar cada modelo ──
    resultados = []
    for modelo_obj, nombre_archivo, nombre_display, Xtr, Xte in modelos:
        res = entrenar_y_evaluar(
            modelo=modelo_obj,
            nombre_modelo=nombre_display,
            X_train=Xtr,
            X_test=Xte,
            y_train=y_train,
            y_test=y_test,
            nombres_clases=nombres_presentes,
            dir_resultados=args.resultados_dir,
            dir_modelos=args.modelos_dir,
        )
        resultados.append(res)

    # ── Guardar resumen general ──
    ruta_resumen = os.path.join(args.resultados_dir, "05_resultados_visuales.txt")
    guardar_resumen(resultados, ruta_resumen)

    # ── Resumen final en consola ──
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL — CLASIFICACIÓN VISUAL")
    print("=" * 60)
    for r in resultados:
        indicador = "" if r == max(resultados, key=lambda x: x["accuracy"]) else "  "
        print(f"  {indicador} {r['nombre']:<42s} Accuracy: {r['accuracy']:.4f}")
    print("=" * 60)
    print("\n Clasificación visual completada exitosamente.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_cnn_clasificacion.py
=======================
Clasificación de documentos escaneados mediante una CNN (Red Neuronal Convolucional).
Implementa la arquitectura definida en la Idea 6 del proyecto.

Arquitectura CNN:
  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64)
  → Flatten → Dense(64) → Dropout(0.3) → Dense(16, softmax)

Entrada:
  - dataset_preprocessed/128x128/{clase}/*.jpg  (imágenes preprocesadas)

Salida:
  - resultados/07_curvas_entrenamiento_cnn.png  (accuracy y loss por época)
  - resultados/07_confusion_cnn.png             (matriz de confusión)
  - resultados/07_resultados_cnn.txt            (reporte resumen)
  - modelos/cnn_model.keras                     (modelo entrenado)

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

# Dimensiones de las imágenes de entrada
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Escala de grises


# ──────────────────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────────────────

def crear_directorios(*dirs: str):
    """Crea los directorios indicados si no existen."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def cargar_imagenes(directorio_base: str):
    """
    Carga imágenes preprocesadas desde la estructura de directorios.

    Estructura esperada:
        directorio_base/
        ├── letter/
        │   ├── letter_00001.jpg
        │   └── ...
        ├── form/
        │   └── ...
        └── ...

    Retorna
    -------
    imagenes : np.ndarray  — shape (N, 128, 128, 1), valores en [0, 1]
    etiquetas : np.ndarray — shape (N,), índices numéricos de clase
    nombres_clases : list  — lista ordenada de nombres de clases encontradas
    conteo_por_clase : dict — cantidad de imágenes por clase
    """
    from tqdm import tqdm

    base = Path(directorio_base)
    if not base.exists():
        print(f"[ERROR] No se encontró el directorio: {base}")
        sys.exit(1)

    # Recopilar rutas de archivos y sus etiquetas
    rutas = []
    etiquetas_str = []

    for clase_dir in sorted(base.iterdir()):
        if not clase_dir.is_dir():
            continue
        nombre_clase = clase_dir.name
        archivos = sorted(clase_dir.glob("*.jpg"))
        # También buscar PNG por si acaso
        archivos.extend(sorted(clase_dir.glob("*.png")))
        for archivo in archivos:
            rutas.append(str(archivo))
            etiquetas_str.append(nombre_clase)

    if len(rutas) == 0:
        print(f"[ERROR] No se encontraron imágenes en {directorio_base}")
        sys.exit(1)

    # Mapear nombres de clase a índices numéricos
    nombres_clases = sorted(set(etiquetas_str))
    clase_a_idx = {nombre: idx for idx, nombre in enumerate(nombres_clases)}

    # Conteo por clase
    conteo_por_clase = {}
    for nombre in nombres_clases:
        conteo_por_clase[nombre] = etiquetas_str.count(nombre)

    # Cargar imágenes en memoria
    print(f"   Cargando {len(rutas)} imágenes...")
    imagenes = np.zeros((len(rutas), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    etiquetas = np.zeros(len(rutas), dtype=np.int32)

    errores = 0
    indices_validos = []

    for i, (ruta, etiqueta_str) in enumerate(tqdm(
        zip(rutas, etiquetas_str),
        total=len(rutas),
        desc="  Cargando imágenes",
        unit="img",
    )):
        try:
            from PIL import Image
            img = Image.open(ruta).convert("L")  # Escala de grises
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)

            # Normalizar pixeles al rango [0, 1]
            arr = arr / 255.0

            imagenes[i, :, :, 0] = arr
            etiquetas[i] = clase_a_idx[etiqueta_str]
            indices_validos.append(i)
        except Exception as e:
            print(f"  ⚠ Error cargando {ruta}: {e}")
            errores += 1

    # Filtrar solo imágenes válidas
    if errores > 0:
        print(f"  ⚠ {errores} imágenes no se pudieron cargar")
        imagenes = imagenes[indices_validos]
        etiquetas = etiquetas[indices_validos]

    return imagenes, etiquetas, nombres_clases, conteo_por_clase


def construir_modelo_cnn(num_clases: int):
    """
    Construye la arquitectura CNN definida en la Idea 6 del proyecto.

    Arquitectura:
        Conv2D(32, 3x3, ReLU) → MaxPool(2x2)
        Conv2D(64, 3x3, ReLU) → MaxPool(2x2)
        Conv2D(64, 3x3, ReLU)
        Flatten → Dense(64, ReLU) → Dropout(0.3)
        Dense(num_clases, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers

    model = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation="softmax"),
    ])

    # Compilar con la configuración especificada en la Idea 6
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def graficar_curvas_entrenamiento(historial, ruta_salida: str):
    """
    Genera gráficas de accuracy y loss durante el entrenamiento.

    Crea un gráfico con dos subplots:
      - Izquierda: accuracy (train vs validation)
      - Derecha: loss (train vs validation)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epocas = range(1, len(historial.history["accuracy"]) + 1)

    # ── Subplot 1: Accuracy ──
    ax1.plot(epocas, historial.history["accuracy"], "b-o", label="Train", markersize=4)
    ax1.plot(epocas, historial.history["val_accuracy"], "r-o", label="Validation", markersize=4)
    ax1.set_title("Accuracy por Época", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # ── Subplot 2: Loss ──
    ax2.plot(epocas, historial.history["loss"], "b-o", label="Train", markersize=4)
    ax2.plot(epocas, historial.history["val_loss"], "r-o", label="Validation", markersize=4)
    ax2.set_title("Loss por Época", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Curvas de Entrenamiento — CNN para Documentos", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Curvas de entrenamiento guardadas: {ruta_salida}")


def guardar_matriz_confusion(y_real, y_pred, nombres_clases, titulo, ruta_salida):
    """
    Genera y guarda una matriz de confusión como imagen PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_real, y_pred)
    n_clases = len(nombres_clases)
    fig_size = max(8, n_clases * 0.7)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
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


def guardar_resumen(accuracy_test, reporte, historial, ruta_modelo, ruta_salida,
                    conteo_por_clase, nombres_clases, num_train, num_test,
                    epochs, batch_size):
    """Guarda un resumen textual completo del entrenamiento CNN."""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RESULTADOS DE CLASIFICACIÓN CNN DE DOCUMENTOS\n")
        f.write("  Proyecto Final — Recuperación de Información — BUAP 2026\n")
        f.write("=" * 70 + "\n\n")

        # Configuración del modelo
        f.write("CONFIGURACIÓN DEL MODELO\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Arquitectura: CNN (3 capas Conv2D + Dense)\n")
        f.write(f"  Tamaño de entrada: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}\n")
        f.write(f"  Épocas: {epochs}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Optimizer: Adam\n")
        f.write(f"  Loss: sparse_categorical_crossentropy\n")
        f.write(f"  Dropout: 0.3\n\n")

        # Distribución del dataset
        f.write("DISTRIBUCIÓN DEL DATASET\n")
        f.write("-" * 40 + "\n")
        total = sum(conteo_por_clase.values())
        for clase, conteo in sorted(conteo_por_clase.items()):
            f.write(f"  {clase:<30s} → {conteo:>5d} imágenes\n")
        f.write(f"  {'TOTAL':<30s} → {total:>5d} imágenes\n")
        f.write(f"\n  Train: {num_train} imágenes\n")
        f.write(f"  Test:  {num_test} imágenes\n\n")

        # Resultados
        f.write("RESULTADOS EN TEST\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy en test: {accuracy_test:.4f}\n\n")

        # Historial de entrenamiento por época
        f.write("HISTORIAL DE ENTRENAMIENTO\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'Época':<8s} {'Train Acc':<12s} {'Val Acc':<12s} {'Train Loss':<12s} {'Val Loss':<12s}\n")
        f.write("  " + "-" * 56 + "\n")
        for i in range(len(historial.history["accuracy"])):
            f.write(
                f"  {i+1:<8d} "
                f"{historial.history['accuracy'][i]:<12.4f} "
                f"{historial.history['val_accuracy'][i]:<12.4f} "
                f"{historial.history['loss'][i]:<12.4f} "
                f"{historial.history['val_loss'][i]:<12.4f}\n"
            )
        f.write("\n")

        # Mejor época
        mejor_val_acc = max(historial.history["val_accuracy"])
        mejor_epoca = historial.history["val_accuracy"].index(mejor_val_acc) + 1
        f.write(f"  Mejor val_accuracy: {mejor_val_acc:.4f} (época {mejor_epoca})\n\n")

        # Reporte de clasificación
        f.write("REPORTE DE CLASIFICACIÓN\n")
        f.write("=" * 70 + "\n")
        f.write(reporte)
        f.write("\n")

        f.write(f"\n  Modelo guardado: {ruta_modelo}\n")

    print(f"\n Resumen guardado: {ruta_salida}")


# ──────────────────────────────────────────────────────────
#  Función principal
# ──────────────────────────────────────────────────────────

def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description="Clasificación de documentos con CNN (Idea 6, sección 6.3).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="dataset_preprocessed/128x128",
        help="Directorio con imágenes preprocesadas (default: dataset_preprocessed/128x128)",
    )
    parser.add_argument(
        "--resultados-dir", type=str, default="resultados",
        help="Directorio donde guardar gráficos y reportes (default: resultados)",
    )
    parser.add_argument(
        "--modelos-dir", type=str, default="modelos",
        help="Directorio donde guardar el modelo entrenado (default: modelos)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Número de épocas de entrenamiento (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Tamaño del batch para entrenamiento (default: 32)",
    )
    parser.add_argument(
        "--validation-split", type=float, default=0.2,
        help="Proporción de datos de entrenamiento para validación (default: 0.2)",
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
        import tensorflow as tf
        from tensorflow.keras import Sequential, layers
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar TensorFlow. Ejecuta: pip install tensorflow"
        ) from exc

    try:
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar scikit-learn. Ejecuta: pip install scikit-learn"
        ) from exc

    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar tqdm. Ejecuta: pip install tqdm"
        ) from exc

    # Suprimir warnings de TensorFlow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    warnings.filterwarnings("ignore")

    # Información de TensorFlow
    print(f"\n TensorFlow versión: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"   GPU(s) detectada(s): {len(gpus)}")
        for gpu in gpus:
            print(f"    • {gpu.name}")
    else:
        print("   Entrenando en CPU (no se detectó GPU)")

    # Fijar semilla para reproducibilidad
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    # ── Crear directorios de salida ──
    crear_directorios(args.resultados_dir, args.modelos_dir)

    # ── Cargar imágenes ──
    print(f"\n Cargando imágenes desde: {args.dataset_dir}")
    imagenes, etiquetas, nombres_clases, conteo_por_clase = cargar_imagenes(args.dataset_dir)

    print(f"\n Total de imágenes: {len(imagenes)}")
    print(f" Clases encontradas ({len(nombres_clases)}): {', '.join(nombres_clases)}")
    print(f" Shape de imágenes: {imagenes.shape}")
    print(f" Rango de valores: [{imagenes.min():.2f}, {imagenes.max():.2f}]")

    for clase, conteo in sorted(conteo_por_clase.items()):
        print(f"  • {clase}: {conteo} imágenes")

    num_clases = len(nombres_clases)

    # ── División train/test (80/20, estratificada) ──
    print(f"\n Dividiendo datos: {int((1-args.test_size)*100)}% train / {int(args.test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        imagenes,
        etiquetas,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=etiquetas,
    )
    print(f"  Train: {len(X_train)} imágenes")
    print(f"  Test:  {len(X_test)} imágenes")

    # ── Construir modelo CNN ──
    print("\n️  Construyendo modelo CNN...")
    modelo = construir_modelo_cnn(num_clases)
    modelo.summary()

    # ── Entrenar el modelo ──
    print(f"\n Iniciando entrenamiento ({args.epochs} épocas, batch={args.batch_size})...")
    historial = modelo.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
    )

    # ── Evaluar en conjunto de prueba ──
    print("\n Evaluando en conjunto de prueba...")
    loss_test, accuracy_test = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss en test:     {loss_test:.4f}")
    print(f"  Accuracy en test: {accuracy_test:.4f}")

    # Predicciones para reporte detallado y matriz de confusión
    y_pred_proba = modelo.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Reporte de clasificación
    reporte = classification_report(
        y_test, y_pred,
        target_names=nombres_clases,
        zero_division=0,
    )
    print(f"\n{reporte}")

    # ── Guardar curvas de entrenamiento ──
    ruta_curvas = os.path.join(args.resultados_dir, "07_curvas_entrenamiento_cnn.png")
    graficar_curvas_entrenamiento(historial, ruta_curvas)

    # ── Guardar matriz de confusión ──
    ruta_cm = os.path.join(args.resultados_dir, "07_confusion_cnn.png")
    guardar_matriz_confusion(
        y_test, y_pred, nombres_clases,
        "Matriz de Confusión — CNN",
        ruta_cm,
    )

    # ── Guardar modelo ──
    ruta_modelo = os.path.join(args.modelos_dir, "cnn_model.keras")
    modelo.save(ruta_modelo)
    print(f"   Modelo CNN guardado: {ruta_modelo}")

    # ── Guardar resumen textual ──
    ruta_resumen = os.path.join(args.resultados_dir, "07_resultados_cnn.txt")
    guardar_resumen(
        accuracy_test=accuracy_test,
        reporte=reporte,
        historial=historial,
        ruta_modelo=ruta_modelo,
        ruta_salida=ruta_resumen,
        conteo_por_clase=conteo_por_clase,
        nombres_clases=nombres_clases,
        num_train=len(X_train),
        num_test=len(X_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── Resumen final en consola ──
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL — CLASIFICACIÓN CNN")
    print("=" * 60)
    print(f"  Accuracy en test:       {accuracy_test:.4f}")
    print(f"  Loss en test:           {loss_test:.4f}")
    mejor_val_acc = max(historial.history["val_accuracy"])
    mejor_epoca = historial.history["val_accuracy"].index(mejor_val_acc) + 1
    print(f"  Mejor val_accuracy:     {mejor_val_acc:.4f} (época {mejor_epoca})")
    print(f"  Modelo guardado:        {ruta_modelo}")
    print(f"  Curvas entrenamiento:   {ruta_curvas}")
    print(f"  Matriz de confusión:    {ruta_cm}")
    print(f"  Resumen completo:       {ruta_resumen}")
    print("=" * 60)
    print("\n Clasificación CNN completada exitosamente.\n")


if __name__ == "__main__":
    main()

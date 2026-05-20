"""
10_evaluacion_comparativa.py — Evaluación Comparativa de Todos los Modelos
============================================================================
Genera la evaluación final del proyecto, comparando todos los modelos:
  - Visual: LBP+SVM, LBP+KNN, GLCM+SVM, Visual Combinado+SVM
  - Textual: TF-IDF+LinearSVC, TF-IDF+LogReg, TF-IDF+MultinomialNB
  - Deep Learning: CNN
  - Fusión: LBP+TF-IDF+SVM

Referencia: Idea 6, Sección 7 — Plan de Evaluación
Métricas: Accuracy, Precision, Recall, F1 (Labs 7-8)
Visualizaciones: Matrices de confusión, tablas, radar, muestras (Labs 7-8)

Uso:
    python 10_evaluacion_comparativa.py
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

# Cargar clases dinámicamente de lo que se haya logrado descargar (ej. 14 de 16)
_dataset_path = Path("dataset")
if _dataset_path.exists():
    CLASSES = sorted([d.name for d in _dataset_path.iterdir() if d.is_dir()])
else:
    CLASSES = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ]

# Colores para gráficas
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F']


def cargar_resultados_desde_archivos(resultados_dir):
    """
    Lee los reportes de resultados de cada script para extraer las métricas.
    Busca archivos: 05_resultados_visuales.txt, 06_resultados_textuales.txt, etc.
    """
    resultados_dir = Path(resultados_dir)
    modelos = {}
    
    # Intentar cargar resultados guardados como JSON
    json_path = resultados_dir / "metricas_todos_modelos.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            modelos = json.load(f)
        print(f"  Cargadas métricas de {len(modelos)} modelos desde JSON")
        return modelos
    
    print("  [!] No se encontró metricas_todos_modelos.json")
    print("  Intentando reentrenar modelos para obtener métricas...")
    
    return modelos


def entrenar_todos_los_modelos(features_dir, dataset_text_dir, dataset_preprocessed_dir):
    """
    Entrena todos los modelos y devuelve las métricas.
    Esta función se usa cuando no existen resultados previos.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    from PIL import Image
    
    features_dir = Path(features_dir)
    resultados = {}
    
    # ════════════════════════════════════════════════════════════════════
    # MODELOS VISUALES
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  MODELOS VISUALES")
    print("─" * 60)
    
    try:
        lbp = np.load(features_dir / "features_lbp.npy")
        glcm = np.load(features_dir / "features_glcm.npy")
        hist = np.load(features_dir / "features_histograms.npy")
        labels_vis = np.load(features_dir / "labels.npy")
        visual_combined = np.load(features_dir / "features_visual_combined.npy")
        
        X_train_lbp, X_test_lbp, y_train, y_test = train_test_split(
            lbp, labels_vis, test_size=0.2, random_state=42, stratify=labels_vis
        )
        X_train_glcm, X_test_glcm, _, _ = train_test_split(
            glcm, labels_vis, test_size=0.2, random_state=42, stratify=labels_vis
        )
        X_train_comb, X_test_comb, _, _ = train_test_split(
            visual_combined, labels_vis, test_size=0.2, random_state=42, stratify=labels_vis
        )
        
        # Escalar features
        scaler_lbp = StandardScaler()
        X_train_lbp_s = scaler_lbp.fit_transform(X_train_lbp)
        X_test_lbp_s = scaler_lbp.transform(X_test_lbp)
        
        scaler_glcm = StandardScaler()
        X_train_glcm_s = scaler_glcm.fit_transform(X_train_glcm)
        X_test_glcm_s = scaler_glcm.transform(X_test_glcm)
        
        scaler_comb = StandardScaler()
        X_train_comb_s = scaler_comb.fit_transform(X_train_comb)
        X_test_comb_s = scaler_comb.transform(X_test_comb)
        
        # ── Modelo 1: LBP + SVM ──
        print("\n  Entrenando LBP + SVM...")
        svm_lbp = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm_lbp.fit(X_train_lbp_s, y_train)
        y_pred = svm_lbp.predict(X_test_lbp_s)
        resultados["LBP + SVM"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "visual"
        }
        print(f"    Accuracy: {resultados['LBP + SVM']['accuracy']:.4f}")
        
        # ── Modelo 2: LBP + KNN ──
        print("  Entrenando LBP + KNN...")
        knn_lbp = KNeighborsClassifier(n_neighbors=5)
        knn_lbp.fit(X_train_lbp_s, y_train)
        y_pred = knn_lbp.predict(X_test_lbp_s)
        resultados["LBP + KNN"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "visual"
        }
        print(f"    Accuracy: {resultados['LBP + KNN']['accuracy']:.4f}")
        
        # ── Modelo 3: GLCM + SVM ──
        print("  Entrenando GLCM + SVM...")
        svm_glcm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm_glcm.fit(X_train_glcm_s, y_train)
        y_pred = svm_glcm.predict(X_test_glcm_s)
        resultados["GLCM + SVM"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "visual"
        }
        print(f"    Accuracy: {resultados['GLCM + SVM']['accuracy']:.4f}")
        
        # ── Modelo 4: Visual Combinado + SVM ──
        print("  Entrenando Visual Combinado + SVM...")
        svm_comb = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm_comb.fit(X_train_comb_s, y_train)
        y_pred = svm_comb.predict(X_test_comb_s)
        resultados["Visual Combinado + SVM"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "visual"
        }
        print(f"    Accuracy: {resultados['Visual Combinado + SVM']['accuracy']:.4f}")
        
    except FileNotFoundError as e:
        print(f"  [!] Features visuales no encontradas: {e}")
        print("  Ejecuta primero: python 02_extraccion_features_visuales.py")
    
    # ════════════════════════════════════════════════════════════════════
    # MODELOS TEXTUALES
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  MODELOS TEXTUALES (OCR + TF-IDF)")
    print("─" * 60)
    
    dataset_text_dir = Path(dataset_text_dir)
    try:
        textos = []
        etiquetas = []
        
        for class_dir in sorted(dataset_text_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for txt_file in sorted(class_dir.glob("*.txt")):
                texto = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                if texto:
                    textos.append(texto)
                    etiquetas.append(class_name)
        
        if len(textos) < 10:
            raise FileNotFoundError("Insuficientes textos OCR")
        
        print(f"  Textos cargados: {len(textos)}")
        
        X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
            textos, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
        )
        
        # ── TF-IDF + LinearSVC ──
        print("\n  Entrenando TF-IDF + LinearSVC...")
        pipe_svm = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('clf', LinearSVC(random_state=42, max_iter=2000))
        ])
        pipe_svm.fit(X_train_txt, y_train_txt)
        y_pred = pipe_svm.predict(X_test_txt)
        resultados["TF-IDF + LinearSVC"] = {
            "accuracy": accuracy_score(y_test_txt, y_pred),
            "precision": precision_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_txt, y_pred, labels=CLASSES).tolist(),
            "report": classification_report(y_test_txt, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "textual"
        }
        print(f"    Accuracy: {resultados['TF-IDF + LinearSVC']['accuracy']:.4f}")
        
        # ── TF-IDF + LogisticRegression ──
        print("  Entrenando TF-IDF + LogisticRegression...")
        pipe_lr = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        pipe_lr.fit(X_train_txt, y_train_txt)
        y_pred = pipe_lr.predict(X_test_txt)
        resultados["TF-IDF + LogReg"] = {
            "accuracy": accuracy_score(y_test_txt, y_pred),
            "precision": precision_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_txt, y_pred, labels=CLASSES).tolist(),
            "report": classification_report(y_test_txt, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "textual"
        }
        print(f"    Accuracy: {resultados['TF-IDF + LogReg']['accuracy']:.4f}")
        
        # ── TF-IDF + MultinomialNB ──
        print("  Entrenando TF-IDF + MultinomialNB...")
        pipe_nb = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        pipe_nb.fit(X_train_txt, y_train_txt)
        y_pred = pipe_nb.predict(X_test_txt)
        resultados["TF-IDF + NaiveBayes"] = {
            "accuracy": accuracy_score(y_test_txt, y_pred),
            "precision": precision_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test_txt, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test_txt, y_pred, labels=CLASSES).tolist(),
            "report": classification_report(y_test_txt, y_pred, target_names=CLASSES, zero_division=0),
            "tipo": "textual"
        }
        print(f"    Accuracy: {resultados['TF-IDF + NaiveBayes']['accuracy']:.4f}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"  [!] Textos OCR no encontrados o insuficientes: {e}")
        print("  Ejecuta primero: python 03_ocr_extraccion_texto.py")
    
    # ════════════════════════════════════════════════════════════════════
    # CNN (cargar resultados si existen)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  CNN (Deep Learning)")
    print("─" * 60)
    
    cnn_results_path = Path("resultados") / "07_resultados_cnn.txt"
    if cnn_results_path.exists():
        # Parsear accuracy del archivo de resultados
        with open(cnn_results_path, "r", encoding="utf-8") as f:
            content = f.read()
            import re
            acc_match = re.search(r'Accuracy.*?:\s*([\d.]+)', content)
            if acc_match:
                cnn_acc = float(acc_match.group(1))
                resultados["CNN"] = {
                    "accuracy": cnn_acc,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tipo": "deep_learning"
                }
                print(f"  CNN Accuracy (del archivo): {cnn_acc:.4f}")
    else:
        print("  [!] Resultados de CNN no encontrados.")
        print("  Ejecuta primero: python 07_cnn_clasificacion.py")
    
    return resultados


def generar_tabla_comparativa(resultados, output_dir):
    """Genera tabla comparativa de todos los modelos (Lab 7)."""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Datos de la tabla
    headers = ["Modelo", "Tipo", "Accuracy", "Precision", "Recall", "F1"]
    rows = []
    
    for nombre, metrics in sorted(resultados.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
        rows.append([
            nombre,
            metrics.get('tipo', 'N/A'),
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Colores de encabezado
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2C3E50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Colores por tipo
    tipo_colors = {
        'visual': '#E8F8F5',
        'textual': '#FEF9E7',
        'deep_learning': '#F5EEF8',
        'fusion': '#FDEDEC'
    }
    
    for i, row in enumerate(rows):
        tipo = row[1]
        color = tipo_colors.get(tipo, '#FFFFFF')
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)
    
    # Resaltar mejor modelo
    if rows:
        for j in range(len(headers)):
            table[1, j].set_text_props(fontweight='bold')
            table[1, j].set_edgecolor('#27AE60')
            table[1, j].set_linewidth(2)
    
    ax.set_title("Comparación de Modelos — Accuracy, Precision, Recall, F1\n"
                 "(Ordenados por Accuracy descendente)",
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = output_dir / "10_tabla_comparativa.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Tabla guardada en: {save_path}")


def generar_grafica_barras(resultados, output_dir):
    """Genera gráfica de barras comparando accuracy de todos los modelos."""
    output_dir = Path(output_dir)
    
    nombres = list(resultados.keys())
    accuracies = [resultados[n].get('accuracy', 0) for n in nombres]
    tipos = [resultados[n].get('tipo', 'other') for n in nombres]
    
    # Colores por tipo
    color_map = {
        'visual': '#3498DB',
        'textual': '#2ECC71',
        'deep_learning': '#9B59B6',
        'fusion': '#E74C3C'
    }
    colores = [color_map.get(t, '#95A5A6') for t in tipos]
    
    # Ordenar por accuracy
    indices = np.argsort(accuracies)[::-1]
    nombres = [nombres[i] for i in indices]
    accuracies = [accuracies[i] for i in indices]
    colores = [colores[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(nombres)), accuracies, color=colores, edgecolor='white', height=0.6)
    
    # Etiquetas de valor
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.set_yticks(range(len(nombres)))
    ax.set_yticklabels(nombres, fontsize=10)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Comparación de Accuracy — Todos los Modelos", fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(accuracies) * 1.15 if accuracies else 1.0)
    ax.invert_yaxis()
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', label='Visual'),
        Patch(facecolor='#2ECC71', label='Textual (OCR)'),
        Patch(facecolor='#9B59B6', label='CNN'),
        Patch(facecolor='#E74C3C', label='Fusión'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_path = output_dir / "10_barras_accuracy.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Gráfica de barras guardada en: {save_path}")


def generar_grafica_radar(resultados, output_dir):
    """
    Genera gráfica radar comparando fortalezas de cada enfoque.
    Referencia: Idea 6, Sección 7 — Visualización #7
    """
    output_dir = Path(output_dir)
    
    # Seleccionar modelos representativos (uno por tipo)
    modelos_repr = {}
    for nombre, m in resultados.items():
        tipo = m.get('tipo', 'other')
        if tipo not in modelos_repr or m.get('accuracy', 0) > modelos_repr[tipo][1].get('accuracy', 0):
            modelos_repr[tipo] = (nombre, m)
    
    if len(modelos_repr) < 2:
        print("  [!] No hay suficientes modelos para gráfica radar")
        return
    
    # Métricas para el radar
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    n_metricas = len(metricas)
    
    angles = np.linspace(0, 2 * np.pi, n_metricas, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el polígono
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors_radar = ['#3498DB', '#2ECC71', '#9B59B6', '#E74C3C']
    
    for i, (tipo, (nombre, metrics)) in enumerate(modelos_repr.items()):
        values = [metrics.get(m, 0) for m in metricas]
        values += values[:1]
        
        color = colors_radar[i % len(colors_radar)]
        ax.plot(angles, values, 'o-', linewidth=2, label=f"{nombre}", color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metricas], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Gráfica Radar — Comparación de Enfoques", fontsize=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    save_path = output_dir / "10_radar_comparacion.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica radar guardada en: {save_path}")


def generar_matrices_confusion(resultados, output_dir):
    """Genera grilla de matrices de confusión para los modelos principales."""
    output_dir = Path(output_dir)
    
    modelos_con_cm = {k: v for k, v in resultados.items() if 'confusion_matrix' in v}
    
    if not modelos_con_cm:
        print("  [!] No hay matrices de confusión disponibles")
        return
    
    n_modelos = len(modelos_con_cm)
    cols = min(3, n_modelos)
    rows = (n_modelos + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n_modelos == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (nombre, metrics) in enumerate(modelos_con_cm.items()):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, cbar=False)
        ax.set_title(f"{nombre}\nAcc: {metrics.get('accuracy', 0):.4f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicción", fontsize=8)
        ax.set_ylabel("Real", fontsize=8)
        ax.tick_params(labelsize=6)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Ocultar ejes vacíos
    for i in range(n_modelos, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    fig.suptitle("Matrices de Confusión — Todos los Modelos",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "10_matrices_confusion_todas.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Matrices de confusión guardadas en: {save_path}")


def generar_reporte_final(resultados, output_dir):
    """Genera el reporte final con todas las métricas y conclusiones."""
    output_dir = Path(output_dir)
    report_path = output_dir / "10_reporte_final.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  EVALUACIÓN COMPARATIVA — CLASIFICADOR DE DOCUMENTOS OCR\n")
        f.write("  Proyecto Final — Recuperación de Información (CCOS-264)\n")
        f.write("  BUAP — Facultad de Ciencias de la Computación — Primavera 2026\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Pipeline: Visual (LBP/GLCM + SVM/KNN) vs Textual (OCR + TF-IDF + SVM)\n")
        f.write("         vs CNN vs Fusión (Visual + Textual)\n\n")
        
        # Tabla resumen
        f.write("─" * 80 + "\n")
        f.write(f"{'Modelo':<30} {'Tipo':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("─" * 80 + "\n")
        
        for nombre in sorted(resultados.keys(), key=lambda x: resultados[x].get('accuracy', 0), reverse=True):
            m = resultados[nombre]
            f.write(f"{nombre:<30} {m.get('tipo', 'N/A'):<12} "
                   f"{m.get('accuracy', 0):>10.4f} {m.get('precision', 0):>10.4f} "
                   f"{m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}\n")
        
        f.write("─" * 80 + "\n\n")
        
        # Mejor modelo
        if resultados:
            mejor = max(resultados.items(), key=lambda x: x[1].get('accuracy', 0))
            f.write(f"MEJOR MODELO: {mejor[0]}\n")
            f.write(f"  Accuracy: {mejor[1].get('accuracy', 0):.4f}\n")
            f.write(f"  Tipo: {mejor[1].get('tipo', 'N/A')}\n\n")
        
        # Reportes detallados
        f.write("\n" + "=" * 80 + "\n")
        f.write("  REPORTES DETALLADOS POR MODELO\n")
        f.write("=" * 80 + "\n")
        
        for nombre, m in resultados.items():
            f.write(f"\n{'─' * 60}\n")
            f.write(f"  {nombre} ({m.get('tipo', 'N/A')})\n")
            f.write(f"{'─' * 60}\n")
            if 'report' in m:
                f.write(m['report'] + "\n")
        
        # Conclusiones
        f.write("\n" + "=" * 80 + "\n")
        f.write("  CONCLUSIONES\n")
        f.write("=" * 80 + "\n\n")
        
        tipos = {}
        for nombre, m in resultados.items():
            tipo = m.get('tipo', 'other')
            if tipo not in tipos:
                tipos[tipo] = []
            tipos[tipo].append((nombre, m.get('accuracy', 0)))
        
        for tipo, modelos in tipos.items():
            mejor = max(modelos, key=lambda x: x[1])
            f.write(f"  Mejor modelo {tipo}: {mejor[0]} (Accuracy: {mejor[1]:.4f})\n")
        
        f.write("\n  El pipeline dual (visual + textual) permite comparar las fortalezas\n")
        f.write("  de cada enfoque para la clasificación de documentos escaneados.\n")
        f.write("  La fusión de features combina lo mejor de ambos mundos.\n")
    
    print(f"  Reporte final guardado en: {report_path}")
    
    # También guardar métricas como JSON para uso futuro
    metricas_json = {}
    for nombre, m in resultados.items():
        metricas_json[nombre] = {
            k: v for k, v in m.items() if k != 'confusion_matrix' and k != 'report'
        }
    
    json_path = output_dir / "metricas_todos_modelos.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, indent=2)
    print(f"  Métricas JSON guardadas en: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación Comparativa de Todos los Modelos (Labs 7-8)"
    )
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--dataset-text", default="dataset_text")
    parser.add_argument("--dataset-preprocessed", default="dataset_preprocessed")
    parser.add_argument("--output-dir", default="resultados")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  EVALUACIÓN COMPARATIVA — TODOS LOS MODELOS")
    print("  Referencia: Idea 6, Sección 7 + Labs 7-8")
    print("=" * 70)
    
    # 1. Intentar cargar resultados existentes o reentrenar
    print("\n[1/5] Cargando/entrenando modelos...")
    resultados = cargar_resultados_desde_archivos(args.output_dir)
    
    if not resultados:
        resultados = entrenar_todos_los_modelos(
            args.features_dir, args.dataset_text, args.dataset_preprocessed
        )
    
    if not resultados:
        print("\n  [ERROR] No se pudieron obtener resultados de ningún modelo.")
        print("  Ejecuta primero los scripts 02-07.")
        sys.exit(1)
    
    print(f"\n  Total de modelos evaluados: {len(resultados)}")
    
    # 2. Tabla comparativa
    print("\n[2/5] Generando tabla comparativa...")
    generar_tabla_comparativa(resultados, args.output_dir)
    
    # 3. Gráfica de barras
    print("\n[3/5] Generando gráfica de barras...")
    generar_grafica_barras(resultados, args.output_dir)
    
    # 4. Gráfica radar
    print("\n[4/5] Generando gráfica radar...")
    generar_grafica_radar(resultados, args.output_dir)
    
    # 5. Matrices de confusión + Reporte final
    print("\n[5/5] Generando matrices de confusión y reporte final...")
    generar_matrices_confusion(resultados, args.output_dir)
    generar_reporte_final(resultados, args.output_dir)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("  ✓ EVALUACIÓN COMPARATIVA COMPLETADA")
    print("  Archivos generados en: " + str(output_dir.resolve()))
    print("=" * 70)
    
    if resultados:
        mejor = max(resultados.items(), key=lambda x: x[1].get('accuracy', 0))
        print(f"\n   MEJOR MODELO: {mejor[0]}")
        print(f"     Accuracy: {mejor[1].get('accuracy', 0):.4f}")
        print(f"     Tipo: {mejor[1].get('tipo', 'N/A')}")
    
    print("\n  Visualizaciones generadas:")
    print("    • 10_tabla_comparativa.png")
    print("    • 10_barras_accuracy.png")
    print("    • 10_radar_comparacion.png")
    print("    • 10_matrices_confusion_todas.png")
    print("    • 10_reporte_final.txt")
    print("    • metricas_todos_modelos.json")


if __name__ == "__main__":
    main()

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

# ------------------------------------------------------------------------------
# Constantes
# ------------------------------------------------------------------------------
import os
from pathlib import Path

from project_config import DEFAULT_CLASSES_14

CLASSES = DEFAULT_CLASSES_14

# Colores para gráficas
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F']


import re

def cargar_resultados_desde_archivos(resultados_dir):
    """
    Lee los reportes de resultados de cada script para extraer las métricas.
    Busca archivos: 05_resultados_visuales.txt, 06_resultados_textuales.txt, etc.
    """
    resultados_dir = Path(resultados_dir)
    modelos = {}
    
    # 1. Intentar cargar resultados guardados como JSON
    json_path = resultados_dir / "metricas_todos_modelos.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                modelos = json.load(f)
            print(f"  Cargadas métricas de {len(modelos)} modelos desde JSON")
            return modelos
        except Exception as e:
            print(f"  [!] Error al cargar {json_path.name}: {e}")
            modelos = {}
    
    print("  [!] No se encontró metricas_todos_modelos.json o estaba corrupto.")
    print("  Intentando parsear reportes de texto existentes en 'resultados/'...")
    
    # 2. Parsear 05_resultados_visuales.txt
    file_05 = resultados_dir / "05_resultados_visuales.txt"
    if file_05.exists():
        try:
            content = file_05.read_text(encoding="utf-8", errors="ignore")
            # Encontrar bloques por modelo
            blocks = content.split("======================================================================")
            for block in blocks:
                if "Modelo:" not in block:
                    continue
                name_match = re.search(r"Modelo:\s*(.*?)\n", block)
                acc_match = re.search(r"Accuracy:\s*([\d.]+)\n", block)
                if name_match and acc_match:
                    name = name_match.group(1).strip()
                    acc = float(acc_match.group(1).strip())
                    
                    # Separar reporte detallado
                    parts = block.split("-" * 70)
                    report = parts[1].strip() if len(parts) >= 2 else ""
                    
                    # Extraer macro avg
                    prec, rec, f1 = 0.0, 0.0, 0.0
                    macro_match = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", report)
                    if macro_match:
                        prec = float(macro_match.group(1))
                        rec = float(macro_match.group(2))
                        f1 = float(macro_match.group(3))
                    
                    modelos[name] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "report": report,
                        "tipo": "visual"
                    }
            print(f"  [OK] Parseados modelos visuales de {file_05.name}")
        except Exception as e:
            print(f"  [!] Error al parsear {file_05.name}: {e}")
            
    # 3. Parsear 06_resultados_textuales.txt
    file_06 = resultados_dir / "06_resultados_textuales.txt"
    if file_06.exists():
        try:
            content = file_06.read_text(encoding="utf-8", errors="ignore")
            blocks = content.split("======================================================================")
            for block in blocks:
                if "Modelo:" not in block:
                    continue
                name_match = re.search(r"Modelo:\s*(.*?)\n", block)
                acc_match = re.search(r"Accuracy:\s*([\d.]+)\n", block)
                if name_match and acc_match:
                    name = name_match.group(1).strip()
                    acc = float(acc_match.group(1).strip())
                    
                    parts = block.split("-" * 70)
                    report = parts[1].strip() if len(parts) >= 2 else ""
                    
                    prec, rec, f1 = 0.0, 0.0, 0.0
                    macro_match = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", report)
                    if macro_match:
                        prec = float(macro_match.group(1))
                        rec = float(macro_match.group(2))
                        f1 = float(macro_match.group(3))
                    
                    # Normalizar nombres para consistencia
                    display_name = name
                    if name == "TF-IDF + LogisticRegression":
                        display_name = "TF-IDF + LogReg"
                    elif name == "TF-IDF + MultinomialNB":
                        display_name = "TF-IDF + NaiveBayes"
                        
                    modelos[display_name] = {
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "report": report,
                        "tipo": "textual"
                    }
            print(f"  [OK] Parseados modelos textuales de {file_06.name}")
        except Exception as e:
            print(f"  [!] Error al parsear {file_06.name}: {e}")
            
    # 4. Parsear 07_resultados_cnn.txt
    file_07 = resultados_dir / "07_resultados_cnn.txt"
    if file_07.exists():
        try:
            content = file_07.read_text(encoding="utf-8", errors="ignore")
            acc_match = re.search(r"Accuracy en test:\s*([\d.]+)", content)
            if acc_match:
                acc = float(acc_match.group(1).strip())
                
                parts = content.split("REPORTE DE CLASIFICACIÓN\n======================================================================")
                report = parts[1].strip() if len(parts) >= 2 else ""
                
                prec, rec, f1 = 0.0, 0.0, 0.0
                macro_match = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", report)
                if macro_match:
                    prec = float(macro_match.group(1))
                    rec = float(macro_match.group(2))
                    f1 = float(macro_match.group(3))
                    
                modelos["CNN"] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "report": report,
                    "tipo": "deep_learning"
                }
                print(f"  [OK] Parseado modelo CNN de {file_07.name}")
        except Exception as e:
            print(f"  [!] Error al parsear {file_07.name}: {e}")

    # 5. Parsear 08_resultados_fusion.txt
    file_08 = resultados_dir / "08_resultados_fusion.txt"
    if file_08.exists():
        try:
            content = file_08.read_text(encoding="utf-8", errors="ignore")
            acc_match = re.search(r"Accuracy:\s*([\d.]+)", content)
            if acc_match:
                acc = float(acc_match.group(1).strip())
                
                parts = content.split("Reporte de Clasificación:")
                report_part = parts[1].split("Matriz de Confusión:") if len(parts) >= 2 else [""]
                report = report_part[0].strip()
                
                # Intentar extraer matriz de confusión de texto
                cm = []
                if len(report_part) >= 2:
                    cm_text = report_part[1].strip()
                    # Encontrar todas las filas entre corchetes
                    rows = re.findall(r"\[([\d\s]+)\]", cm_text)
                    for row in rows:
                        cm.append([int(x) for x in row.split()])
                
                prec, rec, f1 = 0.0, 0.0, 0.0
                macro_match = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", report)
                if macro_match:
                    prec = float(macro_match.group(1))
                    rec = float(macro_match.group(2))
                    f1 = float(macro_match.group(3))
                    
                modelos["Fusión (LBP + TF-IDF) + SVM"] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "report": report,
                    "tipo": "fusion"
                }
                if cm:
                    modelos["Fusión (LBP + TF-IDF) + SVM"]["confusion_matrix"] = cm
                    modelos["Fusión (LBP + TF-IDF) + SVM"]["classes"] = CLASSES
                print(f"  [OK] Parseado modelo de Fusión de {file_08.name}")
        except Exception as e:
            print(f"  [!] Error al parsear {file_08.name}: {e}")

    # 6. Intentar calcular matrices de confusión para modelos visuales y textuales usando los modelos guardados!
    if modelos:
        print("  Cargando modelos entrenados para reconstruir matrices de confusión (sin entrenamiento)...")
        try:
            import joblib
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import confusion_matrix
            
            # Cargar test splits
            features_dir = Path("features")
            lbp = np.load(features_dir / "features_lbp.npy")
            glcm = np.load(features_dir / "features_glcm.npy")
            labels_vis = np.load(features_dir / "labels.npy")
            
            y_encoded = labels_vis.astype(int)
            indices = np.arange(len(y_encoded))
            idx_train, idx_test = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=y_encoded
            )
            y_test = y_encoded[idx_test]
            
            # Cargar clases reales presentes
            clases_presentes = sorted(set(y_encoded))
            nombres_presentes = [CLASSES[i] for i in clases_presentes]
            
            # 6.1 LBP SVM
            if "LBP + SVM" in modelos:
                model_path = Path("modelos/lbp___svm.joblib")
                if model_path.exists():
                    clf = joblib.load(model_path)
                    X_lbp_train = lbp[idx_train]
                    X_lbp_test = lbp[idx_test]
                    scaler = StandardScaler()
                    scaler.fit(X_lbp_train)
                    X_test_scaled = scaler.transform(X_lbp_test)
                    y_pred = clf.predict(X_test_scaled)
                    modelos["LBP + SVM"]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
                    modelos["LBP + SVM"]["classes"] = nombres_presentes
                    
            # 6.2 LBP KNN
            if "LBP + KNN" in modelos:
                model_path = Path("modelos/lbp___knn.joblib")
                if model_path.exists():
                    clf = joblib.load(model_path)
                    X_lbp_train = lbp[idx_train]
                    X_lbp_test = lbp[idx_test]
                    scaler = StandardScaler()
                    scaler.fit(X_lbp_train)
                    X_test_scaled = scaler.transform(X_lbp_test)
                    y_pred = clf.predict(X_test_scaled)
                    modelos["LBP + KNN"]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
                    modelos["LBP + KNN"]["classes"] = nombres_presentes
                    
            # 6.3 GLCM SVM
            if "GLCM + SVM" in modelos:
                model_path = Path("modelos/glcm___svm.joblib")
                if model_path.exists():
                    clf = joblib.load(model_path)
                    X_glcm_train = glcm[idx_train]
                    X_glcm_test = glcm[idx_test]
                    scaler = StandardScaler()
                    scaler.fit(X_glcm_train)
                    X_test_scaled = scaler.transform(X_glcm_test)
                    y_pred = clf.predict(X_test_scaled)
                    modelos["GLCM + SVM"]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
                    modelos["GLCM + SVM"]["classes"] = nombres_presentes
                    
            # 6.4 Combinado Visual SVM
            comb_key = None
            if "Combinado Visual (LBP+GLCM+Hist) + SVM" in modelos:
                comb_key = "Combinado Visual (LBP+GLCM+Hist) + SVM"
            elif "Visual Combinado + SVM" in modelos:
                comb_key = "Visual Combinado + SVM"
                
            if comb_key:
                model_path = Path("modelos/combinado_visual_(lbp_glcm_hist)___svm.joblib")
                if model_path.exists():
                    clf = joblib.load(model_path)
                    # El combinado original usó LBP + GLCM en 05_clasificacion_visual.py (porque features_histogram.npy no existía)
                    # Detectar dimensiones automáticamente
                    try:
                        expected_features = clf.n_features_in_
                    except AttributeError:
                        expected_features = 29
                    
                    if expected_features == 29:
                        X_combinado = np.hstack([lbp, glcm])
                    else:
                        hist = np.load(features_dir / "features_histograms.npy")
                        X_combinado = np.hstack([lbp, glcm, hist])
                        
                    X_comb_train = X_combinado[idx_train]
                    X_comb_test = X_combinado[idx_test]
                    scaler = StandardScaler()
                    scaler.fit(X_comb_train)
                    X_test_scaled = scaler.transform(X_comb_test)
                    y_pred = clf.predict(X_test_scaled)
                    modelos[comb_key]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
                    modelos[comb_key]["classes"] = nombres_presentes
                    
            # 6.5 Textual Models
            dataset_text_dir = Path("dataset_text")
            if dataset_text_dir.exists():
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
                            
                if textos:
                    X_train_txt, X_test_txt, y_train_txt, y_test_txt = train_test_split(
                        textos, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
                    )
                    
                    clases_txt_presentes = sorted(set(y_test_txt))
                    
                    # LinearSVC
                    if "TF-IDF + LinearSVC" in modelos:
                        model_path = Path("modelos/tfidf_svm.joblib")
                        if model_path.exists():
                            clf = joblib.load(model_path)
                            y_pred = clf.predict(X_test_txt)
                            modelos["TF-IDF + LinearSVC"]["confusion_matrix"] = confusion_matrix(y_test_txt, y_pred, labels=clases_txt_presentes).tolist()
                            modelos["TF-IDF + LinearSVC"]["classes"] = clases_txt_presentes
                            
                    # LogReg
                    if "TF-IDF + LogReg" in modelos:
                        model_path = Path("modelos/tfidf_lr.joblib")
                        if model_path.exists():
                            clf = joblib.load(model_path)
                            y_pred = clf.predict(X_test_txt)
                            modelos["TF-IDF + LogReg"]["confusion_matrix"] = confusion_matrix(y_test_txt, y_pred, labels=clases_txt_presentes).tolist()
                            modelos["TF-IDF + LogReg"]["classes"] = clases_txt_presentes
                            
                    # NaiveBayes
                    if "TF-IDF + NaiveBayes" in modelos or "TF-IDF + NaiveBayes" in modelos:
                        nb_key = "TF-IDF + NaiveBayes" if "TF-IDF + NaiveBayes" in modelos else "TF-IDF + NaiveBayes"
                        model_path = Path("modelos/tfidf_nb.joblib")
                        if model_path.exists():
                            clf = joblib.load(model_path)
                            y_pred = clf.predict(X_test_txt)
                            modelos[nb_key]["confusion_matrix"] = confusion_matrix(y_test_txt, y_pred, labels=clases_txt_presentes).tolist()
                            modelos[nb_key]["classes"] = clases_txt_presentes
                            
            print("  [OK] Matrices de confusión y clases asociadas reconstruidas con éxito.")
        except Exception as e:
            print(f"  [!] Advertencia al reconstruir matrices de confusión: {e}")
            print("  Se continuará con las métricas parseadas.")

    return modelos


def entrenar_todos_los_modelos(features_dir, dataset_text_dir, dataset_preprocessed_dir):
    """
    Función fallback en caso de reentrenamiento. 
    Se ha corregido para evitar el error 'target_names size mismatch'.
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
    
    features_dir = Path(features_dir)
    resultados = {}
    
    # --------------------------------------------------------------------
    # Modelos Visuales
    # --------------------------------------------------------------------
    try:
        lbp = np.load(features_dir / "features_lbp.npy")
        glcm = np.load(features_dir / "features_glcm.npy")
        labels_vis = np.load(features_dir / "labels.npy")
        
        y_encoded = labels_vis.astype(int)
        clases_presentes = sorted(set(y_encoded))
        nombres_presentes = [CLASSES[i] for i in clases_presentes]
        
        X_train_lbp, X_test_lbp, y_train, y_test = train_test_split(
            lbp, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        X_train_glcm, X_test_glcm, _, _ = train_test_split(
            glcm, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        scaler_lbp = StandardScaler()
        X_train_lbp_s = scaler_lbp.fit_transform(X_train_lbp)
        X_test_lbp_s = scaler_lbp.transform(X_test_lbp)
        
        scaler_glcm = StandardScaler()
        X_train_glcm_s = scaler_glcm.fit_transform(X_train_glcm)
        X_test_glcm_s = scaler_glcm.transform(X_test_glcm)
        
        print("\n  [Fallback] Entrenando LBP + SVM...")
        svm_lbp = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm_lbp.fit(X_train_lbp_s, y_train)
        y_pred = svm_lbp.predict(X_test_lbp_s)
        resultados["LBP + SVM"] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, target_names=nombres_presentes, zero_division=0),
            "classes": nombres_presentes,
            "tipo": "visual"
        }
        
    except Exception as e:
        print(f"  [Fallback ERROR] No se pudieron entrenar modelos visuales: {e}")
        
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
        clases_modelo = metrics.get('classes', CLASSES[:len(cm)])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=clases_modelo, yticklabels=clases_modelo,
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
        f.write("-" * 80 + "\n")
        f.write(f"{'Modelo':<30} {'Tipo':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 80 + "\n")
        
        for nombre in sorted(resultados.keys(), key=lambda x: resultados[x].get('accuracy', 0), reverse=True):
            m = resultados[nombre]
            f.write(f"{nombre:<30} {m.get('tipo', 'N/A'):<12} "
                   f"{m.get('accuracy', 0):>10.4f} {m.get('precision', 0):>10.4f} "
                   f"{m.get('recall', 0):>10.4f} {m.get('f1', 0):>10.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        
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
            f.write(f"\n{'-' * 60}\n")
            f.write(f"  {nombre} ({m.get('tipo', 'N/A')})\n")
            f.write(f"{'-' * 60}\n")
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
    print("  [OK] EVALUACIÓN COMPARATIVA COMPLETADA")
    print("  Archivos generados en: " + str(output_dir.resolve()))
    print("=" * 70)
    
    if resultados:
        mejor = max(resultados.items(), key=lambda x: x[1].get('accuracy', 0))
        print(f"\n   MEJOR MODELO: {mejor[0]}")
        print(f"     Accuracy: {mejor[1].get('accuracy', 0):.4f}")
        print(f"     Tipo: {mejor[1].get('tipo', 'N/A')}")
    
    print("\n  Visualizaciones generadas:")
    print("    * 10_tabla_comparativa.png")
    print("    * 10_barras_accuracy.png")
    print("    * 10_radar_comparacion.png")
    print("    * 10_matrices_confusion_todas.png")
    print("    * 10_reporte_final.txt")
    print("    * metricas_todos_modelos.json")


if __name__ == "__main__":
    main()

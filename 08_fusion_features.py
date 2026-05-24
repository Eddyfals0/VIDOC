"""
08_fusion_features.py — Fusión de features visuales + textuales (BONUS)
=========================================================================
Concatena vectores LBP + TF-IDF en un solo vector para clasificación.
Referencia: Idea 6, Sección 6 — Modelo 4 (Fusión Visual + Textual)

Uso:
    python 08_fusion_features.py
    python 08_fusion_features.py --features-dir features --output-dir resultados
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

from project_config import DEFAULT_CLASSES_14

CLASSES = DEFAULT_CLASSES_14


def cargar_features_visuales(features_dir):
    """Carga features visuales (LBP) y sus etiquetas."""
    features_dir = Path(features_dir)
    
    lbp = np.load(features_dir / "features_lbp.npy")
    labels = np.load(features_dir / "labels.npy")
    class_names = np.load(features_dir / "class_names.npy", allow_pickle=True)
    file_names = np.load(features_dir / "file_names_visual.npy", allow_pickle=True)
    
    print(f"  Features LBP: {lbp.shape}")
    print(f"  Etiquetas visuales: {labels.shape}")
    
    return lbp, labels, class_names, file_names


def cargar_features_textuales(features_dir):
    """Carga features textuales (TF-IDF de sklearn) y sus etiquetas."""
    from scipy import sparse
    features_dir = Path(features_dir)
    
    tfidf = sparse.load_npz(features_dir / "tfidf_sklearn.npz")
    labels = np.load(features_dir / "labels_text.npy")
    file_names = np.load(features_dir / "file_names_text.npy", allow_pickle=True)
    
    print(f"  Features TF-IDF: {tfidf.shape}")
    print(f"  Etiquetas textuales: {labels.shape}")
    
    return tfidf, labels, file_names


def alinear_datasets(lbp, labels_vis, class_names, file_names_vis, tfidf, labels_txt, file_names_txt):
    """
    Alinea los datasets visual y textual para que tengan las mismas muestras.
    Si algún documento no tiene OCR exitoso, se excluye de la fusión.
    """
    from scipy import sparse

    def key(path):
        p = Path(str(path))
        return f"{p.parent.name}/{p.stem}"

    text_index = {key(path): i for i, path in enumerate(file_names_txt)}
    visual_indices = []
    text_indices = []
    labels_aligned = []

    for i, path in enumerate(file_names_vis):
        k = key(path)
        j = text_index.get(k)
        if j is None:
            continue
        label_name = str(class_names[int(labels_vis[i])])
        if label_name != str(labels_txt[j]):
            continue
        visual_indices.append(i)
        text_indices.append(j)
        labels_aligned.append(label_name)

    print(f"\n  Muestras visuales: {len(labels_vis)}")
    print(f"  Muestras textuales: {len(labels_txt)}")
    print(f"  Muestras alineadas por documento: {len(visual_indices)}")

    if not visual_indices:
        raise ValueError("No se encontraron documentos comunes entre features visuales y textuales.")

    lbp_aligned = lbp[visual_indices]
    tfidf_aligned = tfidf[text_indices]
    labels_aligned = np.array(labels_aligned)

    return lbp_aligned, tfidf_aligned, labels_aligned


def fusionar_features(lbp, tfidf):
    """
    Concatena features visuales + textuales en un solo vector.
    Referencia: Idea 6 — np.hstack([features_lbp, features_tfidf])
    Optimización: Mantiene la representación sparse para entrenamiento ultra-rápido.
    """
    from scipy import sparse
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalizar features visuales (escalar a [0,1])
    scaler_vis = MinMaxScaler()
    lbp_norm = scaler_vis.fit_transform(lbp)
    
    # Asegurar que TF-IDF sea sparse
    if not sparse.issparse(tfidf):
        tfidf_sparse = sparse.csr_matrix(tfidf)
    else:
        tfidf_sparse = tfidf
    
    # Concatenar de forma rala (sparse hstack)
    features_fusionadas = sparse.hstack([sparse.csr_matrix(lbp_norm), tfidf_sparse], format="csr")
    
    print(f"\n  Features LBP normalizadas (dense): {lbp_norm.shape}")
    print(f"  Features TF-IDF (sparse): {tfidf_sparse.shape}")
    print(f"  Features fusionadas (sparse): {features_fusionadas.shape}")
    
    return features_fusionadas


def entrenar_y_evaluar(features, labels, output_dir):
    """
    Entrena SVM sobre features fusionadas y evalúa.
    Referencia: Idea 6 — Modelo 4: svm_fusion = SVC(kernel='rbf')
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import MaxAbsScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import joblib
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelos_dir = Path("modelos")
    modelos_dir.mkdir(parents=True, exist_ok=True)
    
    # División train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")
    
    # Escalar features preservando dispersión
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Modelo: Fusión (LBP + TF-IDF) + SVM ──
    print("\n  Entrenando SVM con features fusionadas...")
    svm_fusion = LinearSVC(C=1.0, dual=False, random_state=42)
    svm_fusion.fit(X_train_scaled, y_train)
    y_pred = svm_fusion.predict(X_test_scaled)
    
    # Filtrar solo clases presentes en los datos de prueba
    nombres_presentes = sorted(set(y_test))
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=nombres_presentes, target_names=nombres_presentes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=nombres_presentes)
    
    print(f"\n  Accuracy Fusión (LBP+TF-IDF) + SVM: {accuracy:.4f}")
    print(f"\n{report}")
    
    # Guardar reporte
    report_path = output_dir / "08_resultados_fusion.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RESULTADOS — FUSIÓN DE FEATURES (LBP + TF-IDF) + SVM\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Modelo: LinearSVC(C=1.0, dual=False)\n")
        f.write(f"Features: LBP ({features.shape[1]} dims) = visual + textual\n")
        f.write(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write(report + "\n\n")
        f.write("Matriz de Confusión:\n")
        f.write(str(cm) + "\n")
    
    print(f"  Reporte guardado en: {report_path}")
    
    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=nombres_presentes, yticklabels=nombres_presentes, ax=ax)
    ax.set_title(f"Matriz de Confusión — Fusión (LBP+TF-IDF) + SVM\nAccuracy: {accuracy:.4f}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    cm_path = output_dir / "08_confusion_fusion.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Matriz de confusión guardada en: {cm_path}")
    
    # Guardar modelo
    model_path = modelos_dir / "fusion_svm.joblib"
    joblib.dump({"model": svm_fusion, "scaler": scaler}, model_path)
    print(f"  Modelo guardado en: {model_path}")
    
    return accuracy, report


def main():
    parser = argparse.ArgumentParser(
        description="Fusión de features visuales + textuales para clasificación (BONUS)"
    )
    parser.add_argument("--features-dir", default="features",
                        help="Directorio con features extraídas (default: features)")
    parser.add_argument("--output-dir", default="resultados",
                        help="Directorio de resultados (default: resultados)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FUSIÓN DE FEATURES — VISUAL (LBP) + TEXTUAL (TF-IDF)")
    print("  Referencia: Idea 6, Modelo 4 (Bonus)")
    print("=" * 70)
    
    # 1. Cargar features
    print("\n[1/4] Cargando features visuales...")
    lbp, labels_vis, class_names, file_names_vis = cargar_features_visuales(args.features_dir)
    
    print("\n[2/4] Cargando features textuales...")
    tfidf, labels_txt, file_names_txt = cargar_features_textuales(args.features_dir)
    
    # 2. Alinear datasets
    print("\n[3/4] Alineando datasets...")
    lbp_aligned, tfidf_aligned, labels = alinear_datasets(
        lbp, labels_vis, class_names, file_names_vis, tfidf, labels_txt, file_names_txt
    )
    
    # 3. Fusionar
    print("\n[3/4] Fusionando features (LBP + TF-IDF)...")
    features_fusionadas = fusionar_features(lbp_aligned, tfidf_aligned)
    
    # 4. Entrenar y evaluar
    print("\n[4/4] Entrenando y evaluando modelo de fusión...")
    accuracy, report = entrenar_y_evaluar(features_fusionadas, labels, args.output_dir)
    
    print("\n" + "=" * 70)
    print(f"  [OK] Fusión completada — Accuracy: {accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

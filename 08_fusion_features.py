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


def cargar_features_visuales(features_dir):
    """Carga features visuales (LBP) y sus etiquetas."""
    features_dir = Path(features_dir)
    
    lbp = np.load(features_dir / "features_lbp.npy")
    labels = np.load(features_dir / "labels.npy")
    class_names = np.load(features_dir / "class_names.npy", allow_pickle=True)
    
    print(f"  Features LBP: {lbp.shape}")
    print(f"  Etiquetas visuales: {labels.shape}")
    
    return lbp, labels, class_names


def cargar_features_textuales(features_dir):
    """Carga features textuales (TF-IDF de sklearn) y sus etiquetas."""
    from scipy import sparse
    features_dir = Path(features_dir)
    
    tfidf = sparse.load_npz(features_dir / "tfidf_sklearn.npz")
    labels = np.load(features_dir / "labels_text.npy")
    
    print(f"  Features TF-IDF: {tfidf.shape}")
    print(f"  Etiquetas textuales: {labels.shape}")
    
    return tfidf, labels


def alinear_datasets(lbp, labels_vis, tfidf, labels_txt):
    """
    Alinea los datasets visual y textual para que tengan las mismas muestras.
    Si algún documento no tiene OCR exitoso, se excluye de la fusión.
    """
    # Encontrar índices comunes
    n_vis = len(labels_vis)
    n_txt = len(labels_txt)
    
    # Si tienen el mismo tamaño, asumimos alineación por orden
    n = min(n_vis, n_txt)
    
    print(f"\n  Muestras visuales: {n_vis}")
    print(f"  Muestras textuales: {n_txt}")
    print(f"  Muestras para fusión: {n}")
    
    lbp_aligned = lbp[:n]
    tfidf_aligned = tfidf[:n]
    labels_aligned = labels_vis[:n]
    
    return lbp_aligned, tfidf_aligned, labels_aligned


def fusionar_features(lbp, tfidf):
    """
    Concatena features visuales + textuales en un solo vector.
    Referencia: Idea 6 — np.hstack([features_lbp, features_tfidf])
    """
    from scipy import sparse
    
    # Normalizar features visuales (escalar a [0,1])
    from sklearn.preprocessing import MinMaxScaler
    scaler_vis = MinMaxScaler()
    lbp_norm = scaler_vis.fit_transform(lbp)
    
    # Convertir TF-IDF sparse a dense si es necesario
    if sparse.issparse(tfidf):
        tfidf_dense = tfidf.toarray()
    else:
        tfidf_dense = tfidf
    
    # Normalizar features textuales
    scaler_txt = MinMaxScaler()
    tfidf_norm = scaler_txt.fit_transform(tfidf_dense)
    
    # Concatenar: [features_lbp | features_tfidf]
    features_fusionadas = np.hstack([lbp_norm, tfidf_norm])
    
    print(f"\n  Features LBP normalizadas: {lbp_norm.shape}")
    print(f"  Features TF-IDF normalizadas: {tfidf_norm.shape}")
    print(f"  Features fusionadas: {features_fusionadas.shape}")
    
    return features_fusionadas


def entrenar_y_evaluar(features, labels, output_dir):
    """
    Entrena SVM sobre features fusionadas y evalúa.
    Referencia: Idea 6 — Modelo 4: svm_fusion = SVC(kernel='rbf')
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
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
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Modelo: Fusión (LBP + TF-IDF) + SVM ──
    print("\n  Entrenando SVM con features fusionadas...")
    svm_fusion = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_fusion.fit(X_train_scaled, y_train)
    y_pred = svm_fusion.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  Accuracy Fusión (LBP+TF-IDF) + SVM: {accuracy:.4f}")
    print(f"\n{report}")
    
    # Guardar reporte
    report_path = output_dir / "08_resultados_fusion.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  RESULTADOS — FUSIÓN DE FEATURES (LBP + TF-IDF) + SVM\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Modelo: SVC(kernel='rbf', C=10, gamma='scale')\n")
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
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
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
    lbp, labels_vis, class_names = cargar_features_visuales(args.features_dir)
    
    print("\n[2/4] Cargando features textuales...")
    tfidf, labels_txt = cargar_features_textuales(args.features_dir)
    
    # 2. Alinear datasets
    print("\n[3/4] Alineando datasets...")
    lbp_aligned, tfidf_aligned, labels = alinear_datasets(
        lbp, labels_vis, tfidf, labels_txt
    )
    
    # 3. Fusionar
    print("\n[3/4] Fusionando features (LBP + TF-IDF)...")
    features_fusionadas = fusionar_features(lbp_aligned, tfidf_aligned)
    
    # 4. Entrenar y evaluar
    print("\n[4/4] Entrenando y evaluando modelo de fusión...")
    accuracy, report = entrenar_y_evaluar(features_fusionadas, labels, args.output_dir)
    
    print("\n" + "=" * 70)
    print(f"  ✓ Fusión completada — Accuracy: {accuracy:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

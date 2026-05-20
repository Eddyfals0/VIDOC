import argparse
from pathlib import Path


def load_text_dataset(dataset_dir):
    texts = []
    labels = []

    for class_dir in sorted(Path(dataset_dir).iterdir()):
        if not class_dir.is_dir():
            continue

        for text_path in sorted(class_dir.glob("*.txt")):
            text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(class_dir.name)

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Entrena un clasificador textual OCR con TF-IDF + SVM.")
    parser.add_argument("--dataset-text", default="dataset_text")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--max-features", type=int, default=3000)
    args = parser.parse_args()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta instalar scikit-learn. Instala dependencias con: pip install scikit-learn"
        ) from exc

    texts, labels = load_text_dataset(args.dataset_text)
    if len(texts) < 2:
        raise SystemExit(
            f"No hay suficientes textos en {args.dataset_text}. "
            "Primero genera archivos .txt con OCR por clase."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=args.max_features, lowercase=True)),
            ("svm", LinearSVC()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Reporte de clasificacion:")
    print(classification_report(y_test, predictions))
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, predictions, labels=sorted(set(labels))))
    print("Orden de clases:")
    print(sorted(set(labels)))


if __name__ == "__main__":
    main()

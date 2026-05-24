"""
Configuracion compartida del proyecto.

El proyecto se replantea para trabajar con las 14 clases descargadas de
RVL-CDIP. Las carpetas vacias o incompletas se ignoran en cada etapa.
"""

from pathlib import Path


DEFAULT_CLASSES_14 = [
    "advertisement",
    "budget",
    "email",
    "file_folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "news_article",
    "presentation",
    "questionnaire",
    "scientific_publication",
    "scientific_report",
    "specification",
]

EXCLUDED_CLASSES = {"memo", "resume"}
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def classes_with_files(base_dir, patterns=IMAGE_EXTENSIONS, excluded=EXCLUDED_CLASSES):
    """Return sorted class folders that contain at least one matching file."""
    base = Path(base_dir)
    if not base.exists():
        return DEFAULT_CLASSES_14.copy()

    classes = []
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir() or class_dir.name in excluded:
            continue
        if any(class_dir.glob(pattern) for pattern in patterns):
            classes.append(class_dir.name)

    return classes or DEFAULT_CLASSES_14.copy()


def classes_from_dataset(base_dir="dataset"):
    """Return the 14-class project scope, ignoring partial empty classes."""
    return classes_with_files(base_dir)

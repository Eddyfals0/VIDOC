# Dataset simplificado

Se descargo una muestra pequena de `sabaridsnfuji/rvl-cdip-filtered`, un subconjunto
de RVL-CDIP con 4 clases: `letter`, `form`, `email` y `resume`.

## Carpetas creadas

```text
dataset_simple/
├── email/
├── form/
├── letter/
├── resume/
└── labels.csv

dataset_simple_preview/
├── email/
├── form/
├── letter/
└── resume/
```

- `dataset_simple/` conserva las imagenes originales en `.tif` para entrenamiento.
- `dataset_simple_preview/` contiene copias `.png` solo para revisar visualmente.
- `dataset_simple/labels.csv` tiene la etiqueta, clase, ruta local y metadatos de cada imagen.

## Tamano actual

La muestra actual tiene 32 documentos:

| Clase | Imagenes |
|-------|----------|
| `letter` | 8 |
| `form` | 8 |
| `email` | 8 |
| `resume` | 8 |

## Regenerar o ampliar

```bash
python download_rvl_cdip_sample.py --samples-per-class 8 --output dataset_simple
```

Para una muestra mas grande:

```bash
python download_rvl_cdip_sample.py --samples-per-class 50 --output dataset_simple
```

## Siguiente paso: texto OCR

Para entrenar la rama textual del proyecto hace falta crear textos OCR por documento.
La estructura recomendada es:

```text
dataset_text/
├── email/
│   └── email_001.txt
├── form/
│   └── form_001.txt
├── letter/
│   └── letter_001.txt
└── resume/
    └── resume_001.txt
```

En este entorno no estan instalados `paddleocr`, `pytesseract`, `Pillow`,
`scikit-learn` ni `tesseract`, asi que la descarga del dataset quedo lista, pero
el paso OCR/modelo textual necesita instalar dependencias o usar el entorno donde
ya hayas corrido OCR antes.

Cuando ya existan los `.txt`, puedes entrenar el baseline textual con:

```bash
python train_text_classifier.py --dataset-text dataset_text
```

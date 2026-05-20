# Clasificador de Imágenes de Documentos (OCR + RI)

> **Proyecto Final — Recuperación de Información (CCOS-264)**  
> **BUAP — Facultad de Ciencias de la Computación — Primavera 2026**

---

## 1. Definición del Problema

En instituciones educativas, empresas y gobierno se manejan grandes volúmenes de documentos escaneados (facturas, credenciales, actas, recibos, formularios). Actualmente, estos documentos se almacenan como imágenes sin metadatos textuales, lo que imposibilita buscar en su contenido.

**Pregunta central:** ¿Se puede construir un sistema que **clasifique documentos escaneados por tipo** (factura, credencial, acta, etc.) y permita **buscar texto dentro de ellos**, combinando descriptores visuales, OCR y técnicas de RI?

### Alcance
- Clasificación multiclase de imágenes de documentos (4-5 categorías)
- Extracción de texto con OCR + indexación con TF-IDF
- Búsqueda textual sobre el contenido extraído
- Comparación de enfoque visual vs textual vs combinado

---

## 2. Justificación

- Integra **3 unidades** del catálogo que rara vez se ven juntas:
  - **Unidad 2:** Preprocesamiento textual, TF-IDF, evaluación
  - **Unidad 4:** Imágenes digitales, descriptores (LBP, GLCM), matching
  - **Unidad 5:** Clasificación supervisada (SVM, KNN)
- **Interdisciplinario:** combina procesamiento de imágenes + NLP + RI clásica
- **Aplicación real:** digitalización de documentos es un problema vigente en México
- Cubre la Unidad 4 (multimedia) que no se trabajó en los laboratorios

---

## 3. Técnicas del Curso Aplicadas

| Técnica | Fuente en el curso | Uso en el proyecto |
|---------|--------------------|--------------------|
| Descriptores de imágenes (LBP) | Unidad 4.2 | Extraer features visuales de documentos |
| Texturas GLCM | Unidad 4.2 | Descriptores de textura para clasificación |
| Correspondencia de plantillas | Unidad 4.7 | Matching visual entre documentos similares |
| OCR (extracción de texto) | Unidad 4 (implícito) | Convertir imágenes a texto buscable |
| Preprocesamiento de texto | Labs 3-4 | Limpieza del texto extraído por OCR |
| TF-IDF | Lab 5 | Representación vectorial del texto OCR |
| Similitud coseno | Lab 6 | Búsqueda de documentos por contenido |
| Precisión, Recall, F1 | Lab 7 | Evaluación de clasificadores |
| SVM, KNN | Unidad 5.1 | Clasificación supervisada |
| CNN (Deep Learning) | Extensión DL | Clasificación de imágenes de documentos |

---

## 4. Metodología (Pipeline)

```
┌─────────────────────────────────────────────────────┐
│           PIPELINE DUAL: VISUAL + TEXTUAL           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. RECOLECCIÓN DE DATOS                            │
│     └─ Imágenes de documentos (4-5 categorías)      │
│                                                     │
│  2. RAMA VISUAL (Clasificación por imagen)          │
│     ├─ Preprocesamiento de imagen                   │
│     │   (resize, escala de grises, normalización)   │
│     ├─ Extracción de descriptores                   │
│     │   ├─ LBP (Local Binary Patterns)              │
│     │   ├─ GLCM (texturas estadísticas)             │
│     │   └─ Histogramas de intensidad                 │
│     ├─ Clasificación con SVM / KNN                  │
│     └─ Clasificación con CNN simple                 │
│                                                     │
│  3. RAMA TEXTUAL (Búsqueda por contenido)           │
│     ├─ OCR con Tesseract                            │
│     ├─ Preprocesamiento del texto extraído          │
│     │   (limpieza, normalización, stop words)       │
│     ├─ Representación TF-IDF                        │
│     ├─ Similitud coseno para búsqueda               │
│     └─ Clasificación con TF-IDF + SVM               │
│                                                     │
│  4. EVALUACIÓN COMPARATIVA                          │
│     ├─ Visual (LBP+SVM) vs Textual (OCR+TF-IDF)    │
│     ├─ CNN vs métodos clásicos                      │
│     ├─ Métricas: Accuracy, Precision, Recall, F1    │
│     └─ Matrices de confusión                        │
│                                                     │
│  5. (BONUS) FUSIÓN DE FEATURES                      │
│     └─ Concatenar features visuales + textuales     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 5. Dataset

### Opción A: Dataset público
**RVL-CDIP** (subconjunto) — https://huggingface.co/datasets/aharley/rvl_cdip
- 16 categorías de documentos (usar solo 4-5)
- Miles de imágenes etiquetadas
- Categorías sugeridas: `letter`, `form`, `invoice`, `resume`, `memo`

### Opción A simplificada para prototipo
Para arrancar sin descargar el RVL-CDIP completo, se puede usar el dataset filtrado:
https://huggingface.co/datasets/sabaridsnfuji/rvl-cdip-filtered

En este repositorio se creó una muestra local en `dataset_simple/` con 4 clases:
`letter`, `form`, `email`, `resume`.

La muestra se puede regenerar o ampliar con:

```bash
python download_rvl_cdip_sample.py --samples-per-class 8 --output dataset_simple
```

Para entrenar la rama textual, el siguiente paso es correr OCR sobre cada imagen y guardar
un `.txt` por documento usando la misma estructura de clases.

### Opción B: Dataset propio (más original)
Recopilar ~200 imágenes de documentos en español:

| Categoría | Ejemplos | Cantidad |
|-----------|----------|----------|
| Factura | Facturas de servicios, compras | ~50 |
| Credencial | INE, credencial escolar, licencia | ~50 |
| Recibo | Recibos de pago, nómina | ~50 |
| Acta | Acta de nacimiento, constancia | ~50 |
| Formulario | Solicitudes, formatos oficiales | ~50 |

### Estructura del dataset

```
dataset/
├── facturas/
│   ├── factura_001.png
│   ├── factura_002.png
│   └── ...
├── credenciales/
│   ├── credencial_001.png
│   └── ...
├── recibos/
│   └── ...
├── actas/
│   └── ...
└── formularios/
    └── ...
```

---

## 6. Modelos a Comparar

### Modelo 1: LBP + SVM (Clasificación visual clásica)
```python
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

def extraer_lbp(imagen_gris, radius=3, n_points=24):
    lbp = local_binary_pattern(imagen_gris, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                           range=(0, n_points + 2), density=True)
    return hist

# Clasificar con SVM sobre histogramas LBP
svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
```
- **Relación:** Unidad 4.2 (descriptores locales) + Unidad 5.1.3 (SVM)

### Modelo 2: OCR + TF-IDF + SVM (Clasificación textual)
```python
import pytesseract
from PIL import Image

def extraer_texto_ocr(ruta_imagen):
    img = Image.open(ruta_imagen)
    texto = pytesseract.image_to_string(img, lang='spa')
    return limpiar_texto(texto)

# Pipeline: texto OCR → TF-IDF → SVM
pipeline_ocr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('svm', LinearSVC())
])
```
- **Relación:** Unidad 4 (OCR) + Lab 5 (TF-IDF) + Unidad 5.1.3 (SVM)

### Modelo 3: CNN Simple (Deep Learning)
```python
model_cnn = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_clases, activation='softmax')
])
```
- **Relación:** Componente de Deep Learning requerido por la rúbrica

### (Bonus) Modelo 4: Fusión Visual + Textual
```python
# Concatenar features LBP + TF-IDF en un solo vector
features_combinadas = np.hstack([features_lbp, features_tfidf])
svm_fusion = SVC(kernel='rbf')
svm_fusion.fit(features_combinadas, labels)
```

---

## 7. Plan de Evaluación

### Métricas por modelo

| Modelo | Entrada | Métricas |
|--------|---------|----------|
| LBP + SVM | Imagen | Accuracy, Precision, Recall, F1 (por clase) |
| OCR + TF-IDF + SVM | Texto de OCR | Accuracy, Precision, Recall, F1 (por clase) |
| CNN | Imagen | Accuracy, Precision, Recall, F1 (por clase) |
| Fusión | Imagen + Texto | Accuracy, Precision, Recall, F1 (por clase) |

### Visualizaciones planeadas
1. **Matrices de confusión** para cada modelo
2. **Tabla comparativa** de accuracy por clase y modelo
3. **Ejemplos de OCR:** texto extraído vs texto real
4. **Histogramas LBP** de cada categoría de documento
5. **Curvas de entrenamiento** del CNN
6. **Muestras de documentos** correctamente/incorrectamente clasificados
7. **Gráfica radar** comparando fortalezas de cada enfoque

---

## 8. Componente de Búsqueda (RI pura)

Además de clasificar, el sistema permitirá **buscar documentos por contenido textual**:

```
Entrada del usuario: "factura servicio eléctrico marzo 2026"
                          ↓
               Preprocesamiento de query
                          ↓
              Representación TF-IDF de la query
                          ↓
       Similitud coseno contra todos los documentos OCR
                          ↓
         Ranking de documentos más relevantes
                          ↓
Salida: Top-5 documentos con mayor similitud + imagen
```

Esto replica directamente el pipeline de los **Labs 5 y 6** pero aplicado a texto extraído de imágenes.

---

## 9. Cronograma

| Semana | Fechas | Actividad |
|--------|--------|-----------|
| 1 | 7-11 mayo | Propuesta + recolección de imágenes + setup OCR |
| 2 | 12-18 mayo | Extracción LBP + OCR + modelos clásicos (SVM, KNN) |
| 3 | 19-25 mayo | CNN + fusión + evaluación comparativa + documento |
| — | 26 mayo | **Entrega final** |

---

## 10. Herramientas

| Herramienta | Uso |
|-------------|-----|
| Python 3.10+ | Lenguaje principal |
| Tesseract OCR | Extracción de texto de imágenes |
| pytesseract | Wrapper Python para Tesseract |
| Pillow / OpenCV | Manipulación de imágenes |
| scikit-image | Descriptores LBP, GLCM |
| scikit-learn | TF-IDF, SVM, KNN, métricas |
| TensorFlow / Keras | CNN |
| Matplotlib / Seaborn | Visualizaciones |
| Jupyter Notebook / Colab | Desarrollo |

---

## 11. Originalidad

- **Pipeline dual** (visual + textual) es poco común en proyectos universitarios
- Cubre la **Unidad 4 (multimedia)** que generalmente se omite en los proyectos
- Combina **3 disciplinas:** visión por computadora + NLP + RI
- Aplicación directa a problemas de **digitalización institucional**
- La fusión de features visual+textual aporta un análisis innovador

---

## 12. Referencias

1. Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern Information Retrieval*. 2nd Edition.
2. Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with LBP*. IEEE TPAMI.
4. Camastra, F., & Vinciarelli, A. (2008). *Machine Learning for Audio, Image and Video Analysis*. Springer.
5. Harley, A. W., Ufkes, A., & Derpanis, K. G. (2015). *Evaluation of Deep CNNs for Document Image Classification*. ICDAR.

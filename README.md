# Clasificador de Documentos OCR

Proyecto final de Recuperación de Información basado en la **Idea 6: Clasificador de Imágenes de Documentos (OCR + RI)**.

El alcance final se replanteo a **14 clases de RVL-CDIP** porque solo esas categorias quedaron disponibles con datos suficientes. Las clases incompletas `memo` y `resume` se excluyen automaticamente del pipeline.

El objetivo es trabajar con imágenes de documentos escaneados para:

- clasificar documentos por tipo con enfoques visuales y textuales,
- extraer texto con OCR,
- indexar el contenido con TF-IDF,
- buscar documentos por similitud textual,
- comparar resultados entre métodos clásicos, CNN y fusión de características.

## Qué incluye este repositorio

Este repositorio conserva el código del proyecto y sus resultados finales. No se suben datasets completos ni carpetas intermedias de procesamiento.

Archivos y carpetas principales:

- `01_preprocesamiento_imagenes.py` a `10_evaluacion_comparativa.py`: pipeline completo del proyecto.
- `run_pipeline.py`: ejecuta la parte final del flujo de evaluación.
- `resultados/`: gráficas, reportes y evidencias generadas por los modelos.
- `modelos/`: modelos entrenados en formato serializado.
- `features/`: representaciones generadas para visualización, OCR y fusión.
- `project_config.py`: configuracion compartida con las 14 clases validas.
- `PROYECTO_14_CLASES_EXPLICACION.md`: explicacion programa por programa y relacion con la problematica de la Idea 6.

## Cómo funciona el proyecto

### 1. Preprocesamiento de imágenes

`01_preprocesamiento_imagenes.py` normaliza las imágenes de documentos. El proceso típico incluye lectura, conversión a escala de grises, redimensionamiento y mejora de contraste.

### 2. Extracción de características visuales

`02_extraccion_features_visuales.py` obtiene descriptores para clasificación clásica:

- LBP para textura local,
- GLCM para textura estadística,
- histogramas de intensidad.

### 3. OCR y extracción de texto

`03_ocr_extraccion_texto.py` aplica OCR sobre cada imagen y guarda el texto por documento. Esa salida se usa para búsqueda y clasificación textual.

### 4. Representación TF-IDF

`04_tfidf_representacion.py` convierte los textos OCR en vectores TF-IDF para poder comparar documentos por contenido.

### 5. Clasificación visual

`05_clasificacion_visual.py` entrena modelos como SVM y KNN usando las características visuales extraídas de las imágenes.

### 6. Clasificación textual

`06_clasificacion_textual.py` entrena clasificadores sobre texto OCR usando TF-IDF y modelos como LinearSVC, Logistic Regression y Naive Bayes.

### 7. CNN

`07_cnn_clasificacion.py` entrena una red convolucional sobre imágenes preprocesadas para comparar aprendizaje profundo contra los métodos clásicos.

### 8. Fusión de características

`08_fusion_features.py` combina rasgos visuales y textuales para evaluar un enfoque híbrido.

### 9. Motor de búsqueda

`09_motor_busqueda.py` permite hacer consultas sobre el texto OCR usando TF-IDF y similitud coseno, devolviendo los documentos más relevantes.

### 10. Evaluación comparativa

`10_evaluacion_comparativa.py` consolida métricas y genera comparaciones entre modelos.

## Resultados generados

Las salidas ya producidas por el proyecto se guardan en `resultados/`. Algunos ejemplos actuales son:

- `01_muestras_preprocesamiento.png`
- `02_histogramas_lbp.png`
- `03_ejemplos_ocr.png`
- `04_top_terminos_tfidf.png`
- `05_confusion_lbp___svm.png`
- `05_confusion_glcm___svm.png`
- `06_confusion_tfidf_svm.png`
- `06_confusion_tfidf_lr.png`
- `06_confusion_tfidf_nb.png`
- `05_resultados_visuales.txt`
- `06_resultados_textuales.txt`

Los modelos entrenados quedan en `modelos/` y las matrices/características intermedias en `features/`.

## Estructura esperada de datos

El proyecto trabaja con documentos organizados por clase. En esta version final se usan 14 clases:

```text
dataset/
├── advertisement/
├── budget/
├── email/
├── file_folder/
├── form/
├── handwritten/
├── invoice/
├── letter/
├── news_article/
├── presentation/
├── questionnaire/
├── scientific_publication/
├── scientific_report/
└── specification/
```

También se usan carpetas derivadas durante el proceso, como `dataset_preprocessed/`, `dataset_text/` y `dataset_text_raw/`, pero esas carpetas se consideran salidas intermedias y no deberían subirse al repositorio.

## Flujo recomendado de ejecución

1. Preprocesar imágenes.
2. Extraer features visuales.
3. Ejecutar OCR.
4. Construir TF-IDF.
5. Entrenar clasificadores visuales y textuales.
6. Entrenar la CNN.
7. Probar la fusión de features.
8. Ejecutar búsquedas sobre el texto OCR.
9. Generar la evaluación comparativa final.

## Nota sobre GitHub

Este repositorio está preparado para versionar el código y los resultados finales. Los datasets originales, los archivos temporales y las salidas intermedias quedan excluidos para mantener el repositorio ligero.

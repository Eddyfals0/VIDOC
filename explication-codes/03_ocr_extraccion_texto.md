# Explicación Detallada: `03_ocr_extraccion_texto.py` (Tesseract) y `03_ocr_extraccion_texto_paddle.py` (PaddleOCR)

Este módulo representa el puente de unión entre la **Visión por Computadora (Procesamiento de Imágenes)** y la **Recuperación de Información (Procesamiento de Lenguaje Natural)**. Su propósito es digitalizar el contenido textual impreso en los documentos mediante **OCR (Optical Character Recognition)** y transformarlo en un formato vectorial indexable (bag-of-words / TF-IDF) siguiendo los rigurosos estándares teóricos de los laboratorios del curso.

---

## 1. Preprocesamiento Físico de Imágenes para OCR

Antes de ejecutar el motor de OCR, la imagen pasa por una etapa de optimización crítica para maximizar el reconocimiento de caracteres:

```python
def preprocesar_imagen(ruta_imagen):
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    _, imagen_bin = cv2.threshold(
        imagen, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return imagen_bin
```

### Binarización Global Adaptativa de Otsu (`cv2.THRESH_OTSU`)
El motor de OCR requiere distinguir con máxima nitidez el texto (primer plano) del papel (fondo). La umbralización simple define un valor estático (ej. 127) para separar blanco y negro, lo cual falla bajo condiciones cambiantes de luz.
*   **¿Cómo funciona la Binarización de Otsu?**
    *   Asume que el histograma de la imagen en escala de grises es **bimodal** (tiene dos picos dominantes correspondientes al fondo y al texto).
    *   Calcula dinámicamente el valor de umbral óptimo $t$ minimizando la varianza intra-clase de las intensidades de ambos grupos, definida como una suma ponderada de varianzas de las dos clases:
        $$\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)$$
        donde $\omega_0$ y $\omega_1$ son las probabilidades de las dos clases separadas por el umbral $t$, y $\sigma_i^2$ son las varianzas de estas clases.
    *   El resultado es una imagen puramente monocromática (binaria) libre de degradados y sombras, ideal para el motor de OCR.

---

## 2. Configuración y Extracción del Motor Tesseract

Tesseract es ejecutado con parámetros altamente optimizados para la lectura de documentos estructurados en inglés:

```python
config_tesseract = '--psm 6 --oem 3'
texto = pytesseract.image_to_string(imagen_bin, lang='eng', config=config_tesseract)
```

*   **PSM (Page Segmentation Mode) = 6:** Configura al analizador para asumir que la imagen es **un único bloque de texto uniforme**. Esto desactiva análisis estructurales pesados y previene que el texto se rompa en párrafos inconexos debido a pequeños ruidos o logos.
*   **OEM (OCR Engine Mode) = 3:** Ejecuta el motor por defecto de Tesseract (actualmente basado en Redes Neuronales LSTM recurrentes con decodificación de conexión temporal de CTC), ofreciendo la mejor relación entre velocidad y precisión.

---

## 3. Pipeline de Preprocesamiento de Texto (Alineación con Labs 2 y 9)

Una vez que el OCR extrae el texto "crudo" (que suele contener errores tipográficos, signos de puntuación extraños y conectores vacíos), se aplica la limpieza y reducción morfológica exigida en los laboratorios de RI:

```python
def preprocesar_texto(texto):
```

### Paso 1: Normalización de Caja (Minúsculas)
```python
texto = texto.lower()
```
Evita la duplicación semántica de términos (ej: *"Documento"* y *"documento"* se unifican).

### Paso 2: Eliminación de Caracteres Especiales (Regex)
```python
texto = re.sub(r'[^a-z0-9\s]', '', texto)
```
Utiliza expresiones regulares para retener únicamente caracteres alfanuméricos y espacios simples, eliminando signos de puntuación, guiones y símbolos especiales producidos por imperfecciones en el OCR.

### Paso 3: Eliminación de Dígitos Numéricos
```python
texto = re.sub(r'\d+', '', texto)
```
Aunque los números son importantes en tablas, en la búsqueda semántica general e indexación TF-IDF suelen generar vocabulario excesivo de baja relevancia (como folios, fechas de minutos o montos).

### Paso 4: Colapso de Espacios
```python
texto = re.sub(r'\s+', ' ', texto).strip()
```
Unifica múltiples tabulaciones, saltos de línea y espacios redundantes en separadores individuales.

### Paso 5: Tokenización y Filtro de Stopwords
```python
tokens = texto.split()
tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 1]
```
Divide la cadena de texto en listas de palabras (tokens) y elimina las Stopwords inglesas cargadas de NLTK (palabras comunes de nulo valor semántico como *"the", "is", "at"*). También filtra palabras con longitud menor o igual a 1.

### Paso 6: Reducción de Flexión Morfológica (Stemming)
```python
tokens = [STEMMER.stem(t) for t in tokens]
```
Aplica el algoritmo de derivación de Snowball (`SnowballStemmer`) para reducir cada token a su raíz morfológica común. Por ejemplo, los términos *"classification"*, *"classifying"* y *"classified"* se reducen a la raíz *"class"*, permitiendo que el motor de búsqueda recupere documentos relevantes sin importar la conjugación lingüística exacta.

---

## 4. Paralelización mediante Procesos Múltiples

Dado que la extracción OCR es computacionalmente costosa (ligada al uso intensivo de CPU por el procesamiento de imágenes e inferencia de redes neuronales), el script implementa multiprocesamiento nativo mediante `ProcessPoolExecutor` de Python:

```python
with ProcessPoolExecutor(max_workers=args.workers) as executor:
    futuros = {executor.submit(procesar_archivo, tarea): tarea for tarea in tareas}
```
Esto distribuye las imágenes entre múltiples núcleos físicos de la CPU (Workers), reduciendo el tiempo de procesamiento total linealmente con el número de núcleos disponibles.

---

## 5. Implementación Avanzada: `03_ocr_extraccion_texto_paddle.py`

En documentos con estructuras complejas (múltiples columnas, tablas incrustadas o inclinaciones de página), Tesseract tradicionalmente lee en horizontal de lado a lado, rompiendo la coherencia de los textos en columnas paralelas.

Para solventar esto, se provee la variante de **PaddleOCR**, la cual no solo utiliza un modelo de detección de texto basado en Deep Learning (DBNet) y reconocimiento (SVTR), sino que incorpora un **algoritmo geométrico sofisticado de reconstrucción de orden de lectura** que emula la lectura humana:

### Algoritmo de Reconstrucción de Párrafos / Burbujas

#### 1. Extracción de Coordenadas
Obtiene los polígonos delimitadores de cada línea de texto detectada por el modelo de PaddleOCR.

#### 2. DSU (Disjoint Set Union - Estructura de Conjuntos Disjuntos)
Para agrupar textos fragmentados en bloques coherentes ("burbujas de lectura"), se implementa una DSU que evalúa si dos líneas $i1$ e $i2$ pertenecen geométricamente al mismo párrafo:

```python
def misma_burbuja(i1, i2):
    dx = max(0, max(i1['min_x'] - i2['max_x'], i2['min_x'] - i1['max_x']))
    dy = max(0, max(i1['min_y'] - i2['max_y'], i2['min_y'] - i1['max_y']))
    
    # Están en la misma línea horizontal y muy cerca, o apilados verticalmente a corta distancia
    mismo_renglon = abs(i1['center_y'] - i2['center_y']) < 6 and dx < 20
    apilados = dx < 25 and dy < 50
    return mismo_renglon or apilados
```
La DSU realiza una unión dinámica (`union(i, j)`) de todas las cajas colindantes, resolviendo la segmentación semántica de párrafos sin importar la rotación ni el espaciado.

#### 3. Ordenamiento Geométrico Bidimensional
Una vez formados los bloques agregados, se ordenan primero verticalmente (`center_y`) para mapear el flujo por renglones, y de izquierda a derecha (`center_x`) para segmentar correctamente las columnas independientes antes de concatenar el texto final.

---

## 6. Salidas Estructuradas de los Scripts

Ambos scripts generan una infraestructura limpia en disco al finalizar su ejecución:

*   **`dataset_text_raw/{clase}/{archivo}.txt`:** Texto crudo extraído directamente por el OCR (útil para auditoría visual).
*   **`dataset_text/{clase}/{archivo}.txt`:** Texto completamente preprocesado, normalizado y derivado (listo para cargar en la matriz TF-IDF de `04_tfidf_representacion.py`).
*   **`resultados/03_ocr_estadisticas.txt`:** Estadísticas detalladas de velocidad de lectura (img/seg), volumen de caracteres promedio por clase y tasas de éxito.
*   **`resultados/03_ejemplos_ocr.png`:** Panel comparativo visual que superpone las imágenes originales analizadas al lado de su respectivo texto extraído.

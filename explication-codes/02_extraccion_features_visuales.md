# Explicación Detallada: `02_extraccion_features_visuales.py`

Este módulo se encarga del **análisis morfológico y de texturas** de los documentos digitales. A diferencia de las aproximaciones puramente textuales (OCR), la clasificación visual clásica asume que diferentes clases de documentos poseen **patrones visuales únicos e identificables** (por ejemplo, los correos electrónicos tienen mucho espacio en blanco arriba y texto en bloque, las facturas presentan tablas y listas compactas, y los artículos científicos constan de columnas paralelas con gráficos).

Para capturar estas firmas visuales, el script extrae tres familias de características de las imágenes preprocesadas a resolución **256x256**:
1.  **LBP (Local Binary Patterns):** Patrones de microtextura local.
2.  **GLCM (Gray-Level Co-occurrence Matrix):** Estadísticas de distribución espacial de niveles de gris.
3.  **Histograma de Intensidad:** Perfil global de contraste e iluminación.

---

## 1. Fundamentos Teóricos de los Descriptores

### A. LBP (Local Binary Patterns - Patrones Binarios Locales)
LBP es un operador de textura altamente eficiente que describe la estructura espacial local de una imagen. Es robusto frente a cambios de contraste e iluminación global.

#### Algoritmo Matemático:
1.  Para un píxel central $c$ con intensidad $g_c$, se seleccionan $P$ vecinos espaciales distribuidos uniformemente sobre una circunferencia de radio $R$.
2.  Cada vecino $p$ con intensidad $g_p$ se compara con el central aplicando una umbralización binaria:
    $$s(g_p - g_c) = \begin{cases} 1 & \text{si } g_p \ge g_c \\ 0 & \text{si } g_p < g_c \end{cases}$$
3.  Se construye el código LBP sumando las potencias de dos de los umbrales del vecindario:
    $$LBP_{P, R} = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p$$
4.  **Patrones Uniformes (`uniform`):** Un patrón LBP se considera uniforme si su secuencia binaria circular contiene como máximo dos transiciones de $0 \to 1$ o de $1 \to 0$. Por ejemplo, `00011100` es uniforme (2 transiciones), mientras que `01010100` no lo es (6 transiciones). El método uniforme reduce el número de combinaciones posibles para $P=24$ de $2^{24}$ a únicamente **26 patrones fundamentales** (que representan bordes, esquinas, valles y manchas), compactando dramáticamente el vector de características y evitando el sobreajuste del clasificador.

---

### B. GLCM (Gray-Level Co-occurrence Matrix - Matriz de Co-ocurrencia de Niveles de Gris)
La GLCM cuantifica las relaciones espaciales entre píxeles adyacentes de la imagen. Es una matriz bidimensional donde cada celda $P(i, j | d, \theta)$ representa la frecuencia con la que un píxel con nivel de gris $i$ se encuentra a una distancia $d$ y en una dirección angular $\theta$ de otro píxel con nivel de gris $j$.

#### Parámetros Configurados:
*   **Cuantización a 64 niveles:** Reducir la imagen original de 256 niveles de gris a 64 (`imagen // 4`) hace que la matriz GLCM sea mucho más densa, reduce drásticamente el uso de memoria de la GPU/CPU y disminuye el impacto del ruido de digitalización.
*   **Distancias ($d$):** $[1, 2, 3]$ píxeles de separación espacial.
*   **Ángulos ($\theta$):** $[0, \pi/4, \pi/2, 3\pi/4]$ radianes (equivalente a direcciones $0^\circ$, $45^\circ$, $90^\circ$, y $135^\circ$).
*   Se calculan **12 matrices de co-ocurrencia** distintas (3 distancias $\times$ 4 direcciones).

#### Propiedades Estadísticas Extraídas de cada Matriz:
Para cada una de las 12 matrices normalizadas, se extraen 5 descriptores clásicos de Haralick:
1.  **Contraste (Contrast):** Mide la cantidad de variaciones locales de intensidad (alto si hay bordes definidos).
    $$\text{Contraste} = \sum_{i,j} |i - j|^2 P(i,j)$$
2.  **Disimilitud (Dissimilarity):** Similar al contraste, pero con crecimiento lineal en lugar de cuadrático.
    $$\text{Disimilitud} = \sum_{i,j} |i - j| P(i,j)$$
3.  **Homogeneidad (Homogeneity):** Mide la proximidad de la distribución de elementos de la GLCM a la diagonal (alto si hay áreas uniformes sin texto).
    $$\text{Homogeneidad} = \sum_{i,j} \frac{P(i,j)}{1 + |i - j|^2}$$
4.  **Energía o Segundo Momento Angular (Energy):** Suma de los cuadrados de las probabilidades de la GLCM. Mide la uniformidad de la textura (alto si la textura es periódica o constante).
    $$\text{Energía} = \sqrt{\sum_{i,j} P(i,j)^2}$$
5.  **Correlación (Correlation):** Mide la dependencia lineal de los niveles de gris de píxeles vecinos (alto si hay patrones lineales continuos).
    $$\text{Correlación} = \sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$$

El vector GLCM final tiene un tamaño de:
$$\text{Tamaño} = 5 \text{ propiedades} \times 3 \text{ distancias} \times 4 \text{ ángulos} = 60 \text{ dimensiones}$$

---

### C. Histograma de Intensidad
Representa la distribución global de niveles de gris de la imagen usando **256 bins**. Es de vital importancia para medir la proporción de tinta versus el fondo del papel (densidad de caracteres tipográficos, márgenes o presencia de imágenes impresas). El histograma se normaliza dividiendo cada elemento por la cantidad total de píxeles para que los valores sean invariantes frente al tamaño de la imagen.

---

## 2. El Vector de Características Combinado (Fusión Temprana)

Para alimentar a los clasificadores clásicos (como Support Vector Machines - SVM o K-Nearest Neighbors - KNN), el script aplica una **fusión temprana a nivel de características** (Early Feature Fusion), concatenando linealmente los descriptores individuales de cada imagen:

$$\mathbf{x}_{\text{visual}} = \begin{bmatrix} \mathbf{x}_{\text{LBP}} & \mathbf{x}_{\text{GLCM}} & \mathbf{x}_{\text{Hist}} \end{bmatrix}$$

*   **Dimensionalidad Total:** $26 \text{ (LBP)} + 60 \text{ (GLCM)} + 256 \text{ (Histograma)} = 342 \text{ dimensiones}$.

---

## 3. Explicación del Código Paso a Paso

El script encapsula la extracción matemática mediante las siguientes funciones clave utilizando la biblioteca científica `scikit-image`:

### A. Extracción de LBP
```python
def extraer_lbp(imagen: np.ndarray) -> np.ndarray:
    lbp_mapa = local_binary_pattern(
        imagen,
        P=LBP_N_POINTS,    # 24 puntos vecinos
        R=LBP_RADIUS,      # Radio 3
        method=LBP_METHOD  # 'uniform'
    )
    # Construcción de histograma normalizado
    histograma, _ = np.histogram(
        lbp_mapa.ravel(),
        bins=LBP_N_BINS,   # 26 bins
        range=(0, LBP_N_BINS),
        density=True       # Normalización probabilística
    )
    return histograma.astype(np.float32)
```

### B. Extracción de GLCM
```python
def extraer_glcm(imagen: np.ndarray) -> np.ndarray:
    # Cuantización de intensidad a 64 niveles de gris
    imagen_cuantizada = (imagen // 4).astype(np.uint8)
    
    # Cálculo de las matrices de co-ocurrencia
    glcm = graycomatrix(
        imagen_cuantizada,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )
    
    # Extracción y aplanado de los 5 descriptores estadísticos
    features = []
    for prop in GLCM_PROPERTIES:
        valores = graycoprops(glcm, prop)  # Retorna matriz (3 distancias x 4 ángulos)
        features.extend(valores.ravel().tolist())
        
    return np.array(features, dtype=np.float32)
```

---

## 4. Estructura de Almacenamiento NumPy en Disco

Los vectores de características y etiquetas se agrupan en matrices multidimensionales y se guardan como archivos binarios comprimidos independientes en formato NumPy (`.npy`) en el directorio `features/`. Esto optimiza la carga en memoria durante la etapa de clasificación en los scripts posteriores:

```python
archivos_salida = {
    "features_lbp.npy": features_lbp,                     # Shape: (N, 26)
    "features_glcm.npy": features_glcm,                   # Shape: (N, 60)
    "features_histograms.npy": features_hist,             # Shape: (N, 256)
    "features_visual_combined.npy": features_combined,    # Shape: (N, 342)
    "labels.npy": labels,                                 # Shape: (N,)
    "class_names.npy": class_names,                       # Nombres de las 14 clases
    "file_names_visual.npy": np.array(nombres_archivos),  # Trazabilidad de archivos
}
```

---

## 5. Visualización: Histogramas LBP Promedio por Categoría

Una sección destacada del script es `generar_visualizacion_lbp`. Esta función toma las características LBP y etiquetas numéricas, calcula la media y la desviación estándar para cada una de las 14 clases de documentos del dataset, y genera un panel comparativo:

```python
lbp_promedio = features_lbp[mascara].mean(axis=0)
lbp_std = features_lbp[mascara].std(axis=0)

bins = np.arange(LBP_N_BINS)
ax.bar(bins, lbp_promedio, color=colores[idx], alpha=0.7, width=0.8)
ax.fill_between(
    bins,
    np.maximum(lbp_promedio - lbp_std, 0),
    lbp_promedio + lbp_std,
    alpha=0.2, color=colores[idx]
)
```
La gráfica se guarda en `resultados/02_histogramas_lbp.png` y muestra la densidad y variabilidad de la microtextura por categoría. Permite validar visualmente si una clase se distingue de otra de forma analítica antes de entrenar los modelos matemáticos.

# 🚀 Guía de Ejecución: PaddleOCR + GPU NVIDIA (RTX 3050 Ti)

Esta guía explica detalladamente cómo ejecutar la versión ultra-rápida de extracción de texto OCR utilizando **PaddleOCR** acelerado por hardware con los núcleos CUDA de tu tarjeta gráfica **NVIDIA GeForce RTX 3050 Ti**.

---

## 💻 1. Entorno de Conda Detectado
Tu entorno con soporte GPU de PaddlePaddle se encuentra en:
* **Ruta absoluta:** `C:\Users\Eduar\.conda\envs\paddle_env`
* **Nombre del entorno:** `paddle_env`

---

## 🏃‍♂️ 2. Cómo Ejecutar el Extractor OCR (PaddleOCR)

Abre una terminal de **PowerShell** en la carpeta raíz de tu proyecto (`VIDOC`) y elige uno de los siguientes métodos de ejecución:

### Método A: Activando el entorno Conda (Recomendado)
```powershell
conda activate paddle_env
python 03_ocr_extraccion_texto_paddle.py --workers 2
```

### Método B: Ejecución directa (Sin activar nada)
```powershell
& "C:\Users\Eduar\.conda\envs\paddle_env\python.exe" 03_ocr_extraccion_texto_paddle.py --workers 2
```

---

## 📊 3. Cómo Monitorear el Progreso en Tiempo Real

Como el proceso actual se lanzó en segundo plano para no interrumpir tu terminal de desarrollo, puedes **sintonizar y ver el avance en vivo** abriendo otra terminal de PowerShell y corriendo este comando:

```powershell
Get-Content -Path "C:\Users\Eduar\.gemini\antigravity-ide\brain/511593cd-87c4-4c1d-a6e7-b1628925ae2f/.system_generated/tasks/task-311.log" -Wait
```
*(Este comando actuará como una transmisión en vivo de la barra de progreso `tqdm` de PaddleOCR).*

---

## 🔄 4. Qué Hacer Después (Actualizar el Proyecto Completo)

Una vez que termine la barra de progreso al 100%, para actualizar todos tus modelos entrenados, métricas y gráficas comparativas con el nuevo texto de súper alta precisión, **copia, pega y ejecuta esta secuencia de comandos en tu terminal activada con `paddle_env`**:

```powershell
# 1. Regenerar la matriz de representación TF-IDF
python 04_tfidf_representacion.py

# 2. Entrenar y actualizar los modelos de clasificación textual (SVM, Logistic Regression, Naive Bayes)
python 06_clasificacion_textual.py

# 3. Entrenar y actualizar el clasificador de Fusión Híbrida (Visual + Textual)
python 08_fusion_features.py

# 4. Generar de nuevo el reporte de métricas consolidado y actualizar todas las gráficas en resultados/
python 10_evaluacion_comparativa.py
```

*(Con esta secuencia, todos tus resultados se actualizarán automáticamente con las nuevas y mejores métricas generadas gracias a PaddleOCR).*

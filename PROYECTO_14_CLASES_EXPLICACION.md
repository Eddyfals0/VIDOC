# Replanteamiento del Proyecto: Clasificador de Documentos OCR con 14 Clases

## Contexto del replanteamiento

La propuesta original de la Idea 6 planteaba construir un sistema capaz de clasificar documentos escaneados y permitir busquedas sobre su contenido mediante OCR y tecnicas de Recuperacion de Informacion. RVL-CDIP contiene 16 categorias, pero en este proyecto se trabajo con las 14 categorias que si quedaron disponibles de forma usable:

- `advertisement`
- `budget`
- `email`
- `file_folder`
- `form`
- `handwritten`
- `invoice`
- `letter`
- `news_article`
- `presentation`
- `questionnaire`
- `scientific_publication`
- `scientific_report`
- `specification`

Las clases `memo` y `resume` se excluyen del alcance final porque no se cuenta con suficientes imagenes procesadas para entrenar y evaluar modelos de forma justa. Esta decision no debilita la idea central: el problema sigue siendo multiclase, con volumen suficiente y con comparacion entre enfoques visuales, textuales, CNN y fusion.

## Problematica que resuelve

En empresas, escuelas e instituciones se almacenan documentos escaneados como imagenes. El problema es que una imagen por si sola no permite buscar texto, filtrar por tipo documental ni organizar automaticamente archivos. El proyecto resuelve esa problematica con un pipeline doble:

1. Una rama visual clasifica documentos por apariencia usando descriptores de imagen.
2. Una rama textual extrae texto con OCR, lo representa con TF-IDF y permite clasificar o buscar documentos por contenido.
3. Una etapa comparativa mide que enfoque funciona mejor segun accuracy, precision, recall y F1.

## Como se adapta el proyecto a 14 clases

El archivo `project_config.py` centraliza el alcance del proyecto. Define las 14 clases validas y excluye carpetas incompletas. Los scripts principales ahora toman las clases desde los datos realmente disponibles, evitando que carpetas vacias generen etiquetas incorrectas o reportes con clases fantasma.

Tambien se corrigio la preparacion de features visuales para guardar `file_names_visual.npy` y la representacion TF-IDF para guardar `file_names_text.npy`. Esto permite que la fusion visual-textual alinee documentos por nombre y clase, no solamente por posicion dentro de un arreglo.

## Explicacion de cada programa

### `project_config.py`

Este archivo no entrena modelos ni genera resultados, pero es importante porque define la configuracion comun del proyecto. Aqui se declara la lista oficial de 14 clases usadas y se excluyen las clases incompletas `memo` y `resume`.

Entrada:

- Carpetas del dataset local, como `dataset/`, `dataset_preprocessed/` o `dataset_text/`.

Proceso:

- Revisa que clases tienen archivos reales.
- Ignora carpetas vacias o incompletas.
- Devuelve una lista ordenada de clases validas.

Salida:

- Lista de clases usada por los demas programas.

Importancia:

Antes, algunos scripts podian tomar las 16 carpetas originales aunque dos clases no tuvieran datos suficientes. Eso podia provocar etiquetas incorrectas, matrices de confusion confusas y comparaciones injustas. Con este archivo, todo el proyecto queda alineado al mismo alcance: 14 clases reales.

### `01_preprocesamiento_imagenes.py`

Este programa prepara las imagenes originales para que todos los modelos reciban documentos con una forma consistente. Los documentos escaneados pueden venir con diferentes tamanos, resoluciones, niveles de contraste o iluminacion. Si se entrenan modelos directamente con esas diferencias, el clasificador puede aprender ruido en lugar de aprender el tipo de documento.

Entrada:

- Imagenes originales organizadas por clase en `dataset/{clase}/`.

Proceso:

- Recorre solamente las 14 clases validas.
- Lee cada imagen.
- Convierte la imagen a escala de grises.
- Redimensiona la imagen a tamanos fijos.
- Aplica mejora de contraste con CLAHE.
- Guarda versiones preprocesadas en carpetas separadas.

Salida:

- `128x128`: usada por la CNN.
- `256x256`: usada por descriptores visuales clasicos.
- Imagen comparativa en `resultados/01_muestras_preprocesamiento.png`.

Importancia:

Resuelve la primera parte de la problematica: convertir documentos escaneados heterogeneos en imagenes normalizadas. Esto permite comparar documentos de forma justa y preparar entradas adecuadas para las ramas visuales del sistema.

### `02_extraccion_features_visuales.py`

Este programa convierte cada imagen preprocesada en vectores numericos. Los modelos clasicos como SVM o KNN no pueden trabajar directamente con imagenes como archivos; necesitan una representacion en forma de numeros.

Entrada:

- Imagenes preprocesadas en `dataset_preprocessed/256x256/{clase}/`.

Proceso:

- Recorre las 14 clases con imagenes validas.
- Calcula LBP para capturar patrones locales de textura.
- Calcula GLCM para capturar relaciones entre tonos vecinos.
- Calcula histogramas de intensidad para resumir distribucion de valores de gris.
- Combina los descriptores visuales en una sola matriz.
- Guarda tambien las etiquetas y nombres de archivo para mantener trazabilidad.

Salida:

- `features/features_lbp.npy`
- `features/features_glcm.npy`
- `features/features_histograms.npy`
- `features/features_visual_combined.npy`
- `features/labels.npy`
- `features/class_names.npy`
- `features/file_names_visual.npy`
- Grafica `resultados/02_histogramas_lbp.png`

Importancia:

Este programa cubre la rama visual clasica de la Idea 6. Permite evaluar si los documentos pueden clasificarse por su apariencia: bloques de texto, tablas, encabezados, distribucion del contenido, espacios en blanco y textura general.

### `03_ocr_extraccion_texto.py`

Este programa extrae texto de las imagenes usando Tesseract OCR. Es una de las piezas centrales del proyecto porque transforma documentos escaneados, que originalmente son imagenes no buscables, en texto procesable.

Entrada:

- Imagenes originales en `dataset/{clase}/`.

Proceso:

- Recorre las 14 clases validas.
- Preprocesa cada imagen para mejorar OCR, por ejemplo convirtiendo a escala de grises y binarizando.
- Ejecuta Tesseract para obtener texto.
- Guarda una version cruda del OCR.
- Limpia el texto eliminando ruido, caracteres innecesarios, stopwords y aplicando normalizacion.
- Registra estadisticas por clase.

Salida:

- Texto crudo en `dataset_text_raw/`.
- Texto limpiado en `dataset_text/`.
- Estadisticas en `resultados/03_ocr_estadisticas.txt`.
- Ejemplos visuales en `resultados/03_ejemplos_ocr.png`.

Importancia:

Esta etapa conecta vision por computadora con Recuperacion de Informacion. Sin OCR, el sistema solo podria clasificar por apariencia. Con OCR, tambien puede buscar por palabras, clasificar por contenido y construir representaciones TF-IDF.

### `04_tfidf_representacion.py`

Este programa convierte los textos OCR en vectores TF-IDF. TF-IDF da mas peso a palabras importantes para un documento y menos peso a palabras frecuentes que aparecen en muchos documentos.

Entrada:

- Archivos `.txt` limpios en `dataset_text/{clase}/`.

Proceso:

- Carga todos los textos con contenido.
- Construye un vocabulario.
- Calcula frecuencia de terminos.
- Calcula IDF para medir que tan distintivo es cada termino.
- Construye matriz TF-IDF manual.
- Construye tambien una matriz TF-IDF con `TfidfVectorizer` de scikit-learn.
- Guarda etiquetas y nombres de documentos.
- Obtiene los terminos mas representativos por clase.

Salida:

- `features/vocabulario.txt`
- `features/matriz_tf.npz`
- `features/vector_idf.npy`
- `features/matriz_tfidf.npz`
- `features/tfidf_sklearn.npz`
- `features/labels_text.npy`
- `features/file_names_text.npy`
- `resultados/04_top_terminos_tfidf.png`

Importancia:

Resuelve la representacion textual del problema. Gracias a este programa, los documentos dejan de ser solo texto plano y se convierten en vectores comparables. Esto se usa en clasificacion textual, busqueda por similitud y fusion visual-textual.

### `05_clasificacion_visual.py`

Este programa entrena y evalua modelos clasicos usando solamente las caracteristicas visuales extraidas en el paso 2. No usa OCR ni contenido textual.

Entrada:

- Features visuales en `features/`.
- Etiquetas en `features/labels.npy`.
- Nombres de clases en `features/class_names.npy`.

Proceso:

- Carga LBP, GLCM e histogramas.
- Divide los datos en entrenamiento y prueba con particion estratificada.
- Escala las features para que SVM y KNN trabajen correctamente.
- Entrena varios clasificadores:

- LBP + SVM.
- LBP + KNN.
- GLCM + SVM.
- Combinado visual + SVM.

Salida:

- Modelos guardados en `modelos/`.
- Reporte `resultados/05_resultados_visuales.txt`.
- Matrices de confusion por modelo.

Importancia:

Permite responder una pregunta clave: que tan lejos se puede llegar clasificando documentos solo por estructura visual. Esto sirve como linea base clasica contra la rama textual y la CNN.

### `06_clasificacion_textual.py`

Este programa clasifica documentos usando el texto extraido por OCR. A diferencia del script visual, aqui el tipo de documento se predice a partir de palabras y terminos representativos.

Entrada:

- Textos OCR limpios en `dataset_text/{clase}/`.

Proceso:

- Carga textos y etiquetas.
- Divide datos en entrenamiento y prueba.
- Para cada modelo, crea un pipeline con TF-IDF y clasificador.
- Entrena tres modelos:

- TF-IDF + LinearSVC.
- TF-IDF + Logistic Regression.
- TF-IDF + Multinomial Naive Bayes.

Salida:

- Modelos guardados en `modelos/tfidf_*.joblib`.
- Reporte `resultados/06_resultados_textuales.txt`.
- Matrices de confusion de los modelos textuales.

Importancia:

Este script evalua la hipotesis principal de la rama textual: si el OCR es suficientemente bueno, el contenido del documento deberia ayudar a identificar su categoria. En los resultados preliminares, esta fue la rama con mejor desempeno.

### `07_cnn_clasificacion.py`

Este programa entrena una red neuronal convolucional simple usando las imagenes preprocesadas de `128x128`. A diferencia de LBP o GLCM, la CNN aprende automaticamente sus propios filtros visuales.

Entrada:

- Imagenes en `dataset_preprocessed/128x128/{clase}/`.

Proceso:

- Carga imagenes en escala de grises.
- Normaliza pixeles al rango `[0, 1]`.
- Divide en entrenamiento y prueba.
- Construye una CNN con capas convolucionales, pooling, capa densa y dropout.
- Entrena el modelo durante varias epocas.
- Evalua en el conjunto de prueba.

Salida:

- Modelo `modelos/cnn_model.keras`.
- Reporte `resultados/07_resultados_cnn.txt`.
- Matriz de confusion `resultados/07_confusion_cnn.png`.
- Curvas de entrenamiento `resultados/07_curvas_entrenamiento_cnn.png`.

Importancia:

Sirve para comparar aprendizaje profundo contra metodos clasicos. Tambien muestra si una CNN sencilla puede aprender patrones visuales de documentos sin usar OCR. Si hay sobreajuste, las curvas de entrenamiento ayudan a justificar mejoras futuras.

### `08_fusion_features.py`

Este programa une dos fuentes de informacion: la apariencia visual del documento y el contenido textual extraido por OCR.

Entrada:

- LBP visual en `features/features_lbp.npy`.
- Etiquetas visuales en `features/labels.npy`.
- Nombres visuales en `features/file_names_visual.npy`.
- TF-IDF textual en `features/tfidf_sklearn.npz`.
- Etiquetas textuales en `features/labels_text.npy`.
- Nombres textuales en `features/file_names_text.npy`.

Proceso:

- Carga features visuales y textuales.
- Alinea documentos por nombre de archivo y clase.
- Normaliza las features visuales.
- Concatena LBP + TF-IDF en una matriz sparse.
- Divide en entrenamiento y prueba.
- Entrena un LinearSVC sobre la representacion fusionada.

Salida:

- Modelo `modelos/fusion_svm.joblib`.
- Reporte `resultados/08_resultados_fusion.txt`.
- Matriz de confusion `resultados/08_confusion_fusion.png`.

Importancia:

Evalua si combinar vision y texto mejora el desempeno. Esta parte es especialmente relevante para documentos donde el OCR falla parcialmente, pero la estructura visual sigue siendo informativa, o al reves.

### `09_motor_busqueda.py`

Este programa implementa el componente de busqueda del proyecto. No busca clasificar primero, sino recuperar documentos relevantes a partir de una consulta escrita por el usuario.

Entrada:

- Textos OCR en `dataset_text/`.
- Consulta del usuario, por ejemplo `"invoice payment total amount"`.

Proceso:

- Carga todos los documentos OCR.
- Construye un indice TF-IDF.
- Preprocesa la consulta.
- Convierte la consulta al mismo espacio vectorial.
- Calcula similitud coseno entre la consulta y todos los documentos.
- Ordena los documentos por relevancia.

Salida:

- Ranking top-k de documentos relevantes.
- Reporte `resultados/09_resultados_busqueda.txt` cuando se ejecutan consultas de ejemplo.
- Visualizacion de resultados cuando hay imagenes asociadas.

Importancia:

Este script resuelve directamente la problematica de Recuperacion de Informacion: permite encontrar documentos escaneados por contenido textual, aunque originalmente fueran imagenes.

### `10_evaluacion_comparativa.py`

Este programa junta las metricas de todos los enfoques y genera una comparacion final. Es la etapa que permite analizar resultados y construir conclusiones.

Entrada:

- Reportes de `resultados/`.
- Modelos guardados en `modelos/`.
- Features y etiquetas guardadas en `features/`.

Proceso:

- Lee o reconstruye metricas de los modelos.
- Organiza resultados por tipo: visual, textual, deep learning y fusion.
- Genera tablas y graficas comparativas.
- Identifica el mejor modelo global y el mejor por familia.
- Produce un reporte final.

Salida:

- `resultados/metricas_todos_modelos.json`
- `resultados/10_reporte_final.txt`
- `resultados/10_barras_accuracy.png`
- `resultados/10_radar_comparacion.png`
- `resultados/10_matrices_confusion_todas.png`
- `resultados/10_tabla_comparativa.png`

Importancia:

Convierte todos los experimentos en evidencia comparable. Sin esta etapa, solo habria resultados aislados; con ella se puede defender que enfoque funciona mejor y por que.

### `run_pipeline.py`

Este archivo sirve como ejecutor auxiliar para correr la etapa final del proyecto sin escribir manualmente todos los comandos.

Entrada:

- Resultados, modelos y features ya generados.

Proceso:

- Lanza la evaluacion comparativa final.

Salida:

- Actualiza los reportes y graficas de la evaluacion.

Importancia:

Es util para repetir la evaluacion cuando ya se entrenaron los modelos. Ayuda a mantener ordenado el flujo de ejecucion y reduce errores al correr comandos manualmente.

## Interpretacion actual de resultados

Los resultados son buenos como primera version academica porque estan muy por encima del azar para 14 clases. El mejor enfoque observado fue textual: OCR + TF-IDF + Logistic Regression. Esto tiene sentido porque muchas categorias documentales se distinguen por palabras y estructura textual.

Sin embargo, antes de presentar resultados finales conviene regenerar features y reportes despues del replanteamiento a 14 clases. La version anterior podia arrastrar etiquetas de clases incompletas, por lo que las metricas antiguas deben tratarse como preliminares.

## Flujo recomendado actualizado

```bash
python 01_preprocesamiento_imagenes.py
python 02_extraccion_features_visuales.py
python 03_ocr_extraccion_texto.py
python 04_tfidf_representacion.py
python 05_clasificacion_visual.py
python 06_clasificacion_textual.py
python 07_cnn_clasificacion.py
python 08_fusion_features.py
python 09_motor_busqueda.py
python 10_evaluacion_comparativa.py
```

Si ya existen salidas antiguas, lo ideal es regenerarlas para que `features/`, `modelos/` y `resultados/` reflejen exactamente el nuevo alcance de 14 clases.

import time
from paddleocr import PaddleOCR

# 1. Inicializa el modelo (la primera vez descargará unos modelos ligeros, luego será instantáneo)
# Le decimos que lea en español ('es') y apagamos clasificadores que pueden rotar el cómic por accidente
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, lang='es', enable_mkldnn=False) 

# 2. Le pasamos la ruta de nuestra imagen
ruta_imagen = 'dataset_simple_preview\email\email_008.png'

print("Iniciando análisis OCR...")
inicio = time.time()
# En la nueva versión se sugiere usar predict() y no se requiere 'cls' porque ya está en init
resultados = ocr.predict(ruta_imagen)
fin = time.time()
tiempo_transcurrido = fin - inicio
print(f"\n--- Análisis completado en {tiempo_transcurrido:.2f} segundos ---\n")

# 3. Agrupamos y guardamos el texto imitando el orden de lectura de un cómic
try:
    if isinstance(resultados[0], dict) and 'rec_texts' in resultados[0] and 'dt_polys' in resultados[0]:
        textos = resultados[0]['rec_texts']
        poligonos = resultados[0]['dt_polys']
        
        # 3.1. Extraer información espacial de cada línea detectada
        items = []
        for poly, texto in zip(poligonos, textos):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            items.append({
                'text': texto, 
                'min_x': min(xs), 
                'max_x': max(xs), 
                'min_y': min(ys),
                'max_y': max(ys),
                'center_y': sum(ys) / 4
            })
            
        # 3.2. Agrupar líneas en "Burbujas" usando componentes conexos
        def misma_burbuja(i1, i2):
            dx = max(0, max(i1['min_x'] - i2['max_x'], i2['min_x'] - i1['max_x']))
            dy = max(0, max(i1['min_y'] - i2['max_y'], i2['min_y'] - i1['max_y']))
            # Están en el mismo renglón y MUY cerca horizontalmente, o apilados verticalmente
            mismo_renglon = abs(i1['center_y'] - i2['center_y']) < 6 and dx < 20
            apilados = dx < 25 and dy < 50
            return mismo_renglon or apilados

        padres = {i: i for i in range(len(items))}
        def find(i):
            if padres[i] != i:
                padres[i] = find(padres[i])
            return padres[i]
            
        def union(i, j):
            raiz_i = find(i)
            raiz_j = find(j)
            if raiz_i != raiz_j:
                padres[raiz_i] = raiz_j

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if misma_burbuja(items[i], items[j]):
                    union(i, j)

        bloques_dict = {}
        for i, item in enumerate(items):
            raiz = find(i)
            if raiz not in bloques_dict:
                bloques_dict[raiz] = []
            bloques_dict[raiz].append(item)

        bloques = []
        for raiz, b_items in bloques_dict.items():
            min_x = min(i['min_x'] for i in b_items)
            max_x = max(i['max_x'] for i in b_items)
            min_y = min(i['min_y'] for i in b_items)
            max_y = max(i['max_y'] for i in b_items)
            bloques.append({
                'items': b_items,
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'center_x': (min_x + max_x) / 2,
                'center_y': (min_y + max_y) / 2
            })

        # 3.3. Agrupar los bloques (burbujas) en Filas (Paneles)
        bloques.sort(key=lambda b: b['center_y'])
        filas = []
        fila_actual = []
        y_actual = None
        
        for b in bloques:
            if y_actual is None:
                y_actual = b['center_y']
                fila_actual.append(b)
            elif abs(b['center_y'] - y_actual) < 180: # Tolerancia grande para paneles en la misma fila horizontal
                fila_actual.append(b)
            else:
                fila_actual.sort(key=lambda b: b['center_x']) # Leer de izquierda a derecha
                filas.append(fila_actual)
                y_actual = b['center_y']
                fila_actual = [b]
                
        if fila_actual:
            fila_actual.sort(key=lambda b: b['center_x'])
            filas.append(fila_actual)

        # 3.4. Generar el texto final reconstruyendo las frases por burbuja
        texto_final = []
        for fila in filas:
            for bloque in fila:
                bloque['items'].sort(key=lambda i: i['center_y'])
                
                lineas_burbuja = []
                linea_act = []
                y_linea = None
                for item in bloque['items']:
                    if y_linea is None:
                        y_linea = item['center_y']
                        linea_act.append(item)
                    elif abs(item['center_y'] - y_linea) < 8: # Reducido para no agrupar renglones consecutivos
                        linea_act.append(item)
                    else:
                        linea_act.sort(key=lambda i: i['min_x'])
                        lineas_burbuja.append(" ".join(i['text'] for i in linea_act))
                        y_linea = item['center_y']
                        linea_act = [item]
                
                if linea_act:
                    linea_act.sort(key=lambda i: i['min_x'])
                    lineas_burbuja.append(" ".join(i['text'] for i in linea_act))
                
                # Escribir todas las líneas de esta burbuja
                for l in lineas_burbuja:
                    texto_final.append(l)
                texto_final.append("") # Línea en blanco entre burbujas
        
        # Guardar archivo
        nombre_archivo_salida = 'resultado_ocr.txt'
        with open(nombre_archivo_salida, 'w', encoding='utf-8') as archivo_txt:
            for linea in texto_final:
                print(linea)
                archivo_txt.write(linea + '\n')
                
        print(f"\n--- ¡Texto agrupado guardado en '{nombre_archivo_salida}'! ---")
    else:
        print("Estructura de resultado distinta a la esperada:", resultados[0].keys())
except Exception as e:
    print(f"Error al procesar los resultados: {e}")

import time
from paddleocr import PaddleOCR

class ComicOCR:
    def __init__(self, lang='es', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False):
        """
        Inicializa el modelo PaddleOCR con la configuración deseada.
        """
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            lang=lang,
            enable_mkldnn=enable_mkldnn
        )

    def analyze_image(self, image_path):
        """
        Realiza el análisis OCR de una imagen y devuelve el texto estructurado 
        en formato de lista de líneas (strings).
        """
        print(f"Iniciando análisis OCR en '{image_path}'...")
        inicio = time.time()
        resultados = self.ocr.predict(image_path)
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        print(f"--- Análisis completado en {tiempo_transcurrido:.2f} segundos ---")
        
        return self._group_text(resultados)

    def _group_text(self, resultados):
        """
        Agrupa y ordena el texto detectado imitando el orden de lectura de un cómic.
        """
        try:
            if isinstance(resultados[0], dict) and 'rec_texts' in resultados[0] and 'dt_polys' in resultados[0]:
                textos = resultados[0]['rec_texts']
                poligonos = resultados[0]['dt_polys']
                
                # 1. Extraer información espacial de cada línea detectada
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
                    
                # 2. Agrupar líneas en "Burbujas" usando componentes conexos
                def misma_burbuja(i1, i2):
                    dx = max(0, max(i1['min_x'] - i2['max_x'], i2['min_x'] - i1['max_x']))
                    dy = max(0, max(i1['min_y'] - i2['max_y'], i2['min_y'] - i1['max_y']))
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

                # 3. Agrupar los bloques (burbujas) en Filas (Paneles)
                bloques.sort(key=lambda b: b['center_y'])
                filas = []
                fila_actual = []
                y_actual = None
                
                for b in bloques:
                    if y_actual is None:
                        y_actual = b['center_y']
                        fila_actual.append(b)
                    elif abs(b['center_y'] - y_actual) < 180:
                        fila_actual.append(b)
                    else:
                        fila_actual.sort(key=lambda b: b['center_x'])
                        filas.append(fila_actual)
                        y_actual = b['center_y']
                        fila_actual = [b]
                        
                if fila_actual:
                    fila_actual.sort(key=lambda b: b['center_x'])
                    filas.append(fila_actual)

                # 4. Generar el texto final reconstruyendo las frases por burbuja
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
                            elif abs(item['center_y'] - y_linea) < 8:
                                linea_act.append(item)
                            else:
                                linea_act.sort(key=lambda i: i['min_x'])
                                lineas_burbuja.append(" ".join(i['text'] for i in linea_act))
                                y_linea = item['center_y']
                                linea_act = [item]
                        
                        if linea_act:
                            linea_act.sort(key=lambda i: i['min_x'])
                            lineas_burbuja.append(" ".join(i['text'] for i in linea_act))
                        
                        for l in lineas_burbuja:
                            texto_final.append(l)
                        texto_final.append("") # Línea en blanco entre burbujas
                
                return texto_final
            else:
                print("Estructura de resultado distinta a la esperada:", resultados[0].keys())
                return []
        except Exception as e:
            print(f"Error al procesar los resultados: {e}")
            return []

    def get_text(self, image_path):
        """
        Extrae el texto de la imagen y lo devuelve como una única cadena de texto (string).
        """
        lineas = self.analyze_image(image_path)
        return "\n".join(lineas)

    def to_txt(self, image_path, output_path='resultado_ocr.txt'):
        """
        Extrae el texto de la imagen y lo guarda en un archivo .txt.
        """
        lineas = self.analyze_image(image_path)
        if not lineas:
            print("No se encontró texto para guardar.")
            return False

        try:
            with open(output_path, 'w', encoding='utf-8') as archivo_txt:
                for linea in lineas:
                    print(linea)
                    archivo_txt.write(linea + '\n')
            print(f"\n--- ¡Texto guardado exitosamente en '{output_path}'! ---")
            return True
        except Exception as e:
            print(f"Error al escribir en el archivo: {e}")
            return False

# =============================================================================
# Ejemplo de uso:
# =============================================================================
if __name__ == '__main__':
    # Inicializar el lector
    comic_reader = ComicOCR()
    
    ruta = 'dataset_simple_preview\email\email_001.png'
    
    # 1. Si quieres guardar el texto en un archivo
    comic_reader.to_txt(ruta, 'resultado_ocr.txt')
    
    # 2. Si quieres obtener el texto directamente en una variable
    # texto_extraido = comic_reader.get_text(ruta)
    # print(texto_extraido)

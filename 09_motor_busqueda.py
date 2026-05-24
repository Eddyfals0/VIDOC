"""
09_motor_busqueda.py — Motor de Búsqueda Textual por Contenido OCR
====================================================================
Replica el pipeline de los Labs 5 y 6: TF-IDF + Similitud Coseno
para buscar documentos por contenido textual.

Referencia: Idea 6, Sección 8 — Componente de Búsqueda (RI pura)

Pipeline:
    Query del usuario → Preprocesamiento → TF-IDF → Similitud Coseno → Ranking

Uso:
    python 09_motor_busqueda.py
    python 09_motor_busqueda.py --query "invoice payment march"
    python 09_motor_busqueda.py --interactive
"""

import argparse
import re
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path

from project_config import DEFAULT_CLASSES_14

CLASSES = DEFAULT_CLASSES_14


def cargar_documentos(dataset_text_dir):
    """Carga todos los documentos de texto OCR."""
    dataset_text_dir = Path(dataset_text_dir)
    documentos = []
    rutas = []
    clases = []
    
    for class_dir in sorted(dataset_text_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        
        for txt_file in sorted(class_dir.glob("*.txt")):
            texto = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if texto:
                documentos.append(texto)
                rutas.append(str(txt_file))
                clases.append(class_name)
    
    print(f"  Documentos cargados: {len(documentos)}")
    print(f"  Clases encontradas: {len(set(clases))}")
    
    return documentos, rutas, clases


def preprocesar_query(query):
    """
    Preprocesa la query del usuario (mismo pipeline que Lab 2/9).
    1. Lowercase
    2. Remover caracteres especiales
    3. Remover números
    4. Remover stopwords
    5. Stemming
    """
    from nltk.corpus import stopwords
    from nltk.stem.snowball import SnowballStemmer
    
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    
    # Lowercase
    texto = query.lower()
    # Remover caracteres especiales
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    # Remover números
    texto = re.sub(r'\d+', '', texto)
    # Tokenizar
    tokens = texto.split()
    # Remover stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    # Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)


def construir_indice(documentos):
    """
    Construye el índice TF-IDF usando sklearn.
    Referencia: Lab 5 — TfidfVectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(documentos)
    
    print(f"  Vocabulario: {len(vectorizer.vocabulary_)} términos")
    print(f"  Matriz TF-IDF: {tfidf_matrix.shape}")
    print(f"  Sparsidad: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")
    
    return vectorizer, tfidf_matrix


def buscar(query, vectorizer, tfidf_matrix, documentos, rutas, clases, top_k=5):
    """
    Busca documentos similares a la query usando similitud coseno.
    Referencia: Lab 6 — sklearn.metrics.pairwise.cosine_similarity
    
    Pipeline:
        Query → Preprocesamiento → TF-IDF → Similitud Coseno → Ranking
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Preprocesar query
    query_procesada = preprocesar_query(query)
    
    if not query_procesada.strip():
        print("  [!] La query quedó vacía después del preprocesamiento.")
        return []
    
    # Vectorizar query
    query_vector = vectorizer.transform([query_procesada])
    
    # Calcular similitud coseno contra todos los documentos
    similitudes = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Obtener Top-K resultados (similitud > 0)
    indices_ordenados = np.argsort(similitudes)[::-1]
    
    resultados = []
    for idx in indices_ordenados:
        if similitudes[idx] <= 0:
            break
        if len(resultados) >= top_k:
            break
        
        resultados.append({
            "rank": len(resultados) + 1,
            "similitud": float(similitudes[idx]),
            "ruta": rutas[idx],
            "clase": clases[idx],
            "texto_preview": documentos[idx][:200] + "..." if len(documentos[idx]) > 200 else documentos[idx],
        })
    
    return resultados


def mostrar_resultados(query, resultados):
    """Muestra los resultados de búsqueda en formato legible."""
    print(f"\n{'═' * 70}")
    print(f"  BÚSQUEDA: \"{query}\"")
    print(f"{'═' * 70}")
    
    if not resultados:
        print("  No se encontraron documentos relevantes.")
        return
    
    for r in resultados:
        print(f"\n  [{r['rank']}] Similitud: {r['similitud']:.4f} | Clase: {r['clase']}")
        print(f"      Archivo: {Path(r['ruta']).name}")
        print(f"      Preview: {r['texto_preview'][:100]}...")
    
    print(f"\n{'═' * 70}")


def visualizar_resultados(query, resultados, dataset_dir, output_dir):
    """
    Genera visualización de los resultados de búsqueda.
    Muestra las imágenes de los documentos encontrados.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(dataset_dir)
    
    n_results = min(len(resultados), 5)
    if n_results == 0:
        return
    
    fig, axes = plt.subplots(1, n_results, figsize=(4 * n_results, 6))
    if n_results == 1:
        axes = [axes]
    
    fig.suptitle(f'Búsqueda: "{query}"', fontsize=14, fontweight='bold')
    
    for i, r in enumerate(resultados[:n_results]):
        ax = axes[i]
        
        # Intentar encontrar la imagen correspondiente
        txt_path = Path(r['ruta'])
        # Construir ruta de imagen: dataset/{clase}/{nombre}.jpg
        img_name = txt_path.stem + ".jpg"
        img_path = dataset_dir / r['clase'] / img_name
        
        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img, cmap='gray')
        else:
            ax.text(0.5, 0.5, "Imagen\nno encontrada",
                    ha='center', va='center', fontsize=10,
                    transform=ax.transAxes)
        
        ax.set_title(f"[{r['rank']}] {r['clase']}\nSim: {r['similitud']:.4f}",
                     fontsize=9, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "09_resultados_busqueda.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Visualización guardada en: {save_path}")


def modo_interactivo(vectorizer, tfidf_matrix, documentos, rutas, clases, dataset_dir, output_dir):
    """Modo interactivo de búsqueda (como un mini buscador)."""
    print("\n" + "=" * 70)
    print("  MOTOR DE BÚSQUEDA INTERACTIVO")
    print("  Escribe tu consulta y presiona Enter.")
    print("  Escribe 'salir' para terminar.")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n   Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.lower() in ('salir', 'exit', 'quit', 'q'):
            print("  ¡Hasta luego!")
            break
        
        if not query:
            continue
        
        resultados = buscar(query, vectorizer, tfidf_matrix, documentos, rutas, clases, top_k=5)
        mostrar_resultados(query, resultados)
        
        if resultados:
            visualizar_resultados(query, resultados, dataset_dir, output_dir)


def ejecutar_queries_ejemplo(vectorizer, tfidf_matrix, documentos, rutas, clases, dataset_dir, output_dir):
    """Ejecuta queries de ejemplo para demostrar el motor de búsqueda."""
    queries_ejemplo = [
        "invoice payment total amount",
        "dear sir letter regarding",
        "meeting agenda schedule",
        "scientific report research results",
        "budget financial report expenses",
        "email subject from to",
        "scientific research method results",
        "form please fill name address",
    ]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivo de resultados
    report_path = output_dir / "09_resultados_busqueda.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  MOTOR DE BÚSQUEDA — RESULTADOS DE QUERIES DE EJEMPLO\n")
        f.write("  Pipeline: Query → Preprocesamiento → TF-IDF → Coseno → Ranking\n")
        f.write("  Referencia: Labs 5 y 6 del curso\n")
        f.write("=" * 70 + "\n\n")
        
        for query in queries_ejemplo:
            resultados = buscar(query, vectorizer, tfidf_matrix,
                              documentos, rutas, clases, top_k=5)
            
            mostrar_resultados(query, resultados)
            
            f.write(f"\nQuery: \"{query}\"\n")
            f.write(f"Query procesada: \"{preprocesar_query(query)}\"\n")
            f.write("-" * 50 + "\n")
            
            if not resultados:
                f.write("  Sin resultados relevantes.\n")
            else:
                for r in resultados:
                    f.write(f"  [{r['rank']}] Sim={r['similitud']:.4f} | "
                           f"Clase={r['clase']} | {Path(r['ruta']).name}\n")
            f.write("\n")
    
    print(f"\n  Resultados guardados en: {report_path}")
    
    # Visualizar último resultado
    if queries_ejemplo:
        last_results = buscar(queries_ejemplo[0], vectorizer, tfidf_matrix,
                            documentos, rutas, clases, top_k=5)
        if last_results:
            visualizar_resultados(queries_ejemplo[0], last_results,
                                dataset_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Motor de Búsqueda Textual — TF-IDF + Similitud Coseno (Labs 5-6)"
    )
    parser.add_argument("--dataset-text", default="dataset_text",
                        help="Directorio con textos OCR preprocesados")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Directorio con imágenes originales")
    parser.add_argument("--output-dir", default="resultados",
                        help="Directorio de resultados")
    parser.add_argument("--query", type=str, default=None,
                        help="Query para buscar (opcional)")
    parser.add_argument("--interactive", action="store_true",
                        help="Modo interactivo")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Número de resultados (default: 5)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MOTOR DE BÚSQUEDA TEXTUAL")
    print("  Pipeline: TF-IDF + Similitud Coseno")
    print("  Referencia: Labs 5 y 6 del curso de RI")
    print("=" * 70)
    
    # 1. Cargar documentos
    print("\n[1/3] Cargando documentos OCR...")
    documentos, rutas, clases = cargar_documentos(args.dataset_text)
    
    if len(documentos) == 0:
        print("  [ERROR] No se encontraron documentos de texto.")
        print(f"  Verifica que existan archivos .txt en {args.dataset_text}/")
        sys.exit(1)
    
    # 2. Construir índice TF-IDF
    print("\n[2/3] Construyendo índice TF-IDF...")
    vectorizer, tfidf_matrix = construir_indice(documentos)
    
    # 3. Buscar
    print("\n[3/3] Ejecutando búsquedas...")
    
    if args.query:
        # Búsqueda individual
        resultados = buscar(args.query, vectorizer, tfidf_matrix,
                          documentos, rutas, clases, top_k=args.top_k)
        mostrar_resultados(args.query, resultados)
        if resultados:
            visualizar_resultados(args.query, resultados, args.dataset_dir, args.output_dir)
    
    elif args.interactive:
        # Modo interactivo
        modo_interactivo(vectorizer, tfidf_matrix, documentos, rutas, clases,
                        args.dataset_dir, args.output_dir)
    
    else:
        # Ejecutar queries de ejemplo
        ejecutar_queries_ejemplo(vectorizer, tfidf_matrix, documentos, rutas, clases,
                                args.dataset_dir, args.output_dir)
    
    print("\n" + "=" * 70)
    print("  [OK] Motor de búsqueda completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

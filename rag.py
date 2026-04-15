"""
rag.py - Sistema RAG con FAISS para recomendación de películas y series
Usa FAISS como retriever (diferente al ChromaDB del profe)
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import json
import shutil
import hashlib

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
PELICULAS_PATH = "./peliculas.json"
SERIES_PATH = "./series.json"
FAISS_PATH = "./faiss_index"
INDEX_META_PATH = "./faiss_index/catalogo_meta.json"


def _cargar_catalogo(path: str, tipo: str) -> list:
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    documentos = []
    for item in items:
        if tipo == "pelicula":
            contenido = (
                f"Tipo: Película\n"
                f"Título: {item.get('titulo', 'Sin título')}\n"
                f"Género: {item.get('genero', 'N/A')}\n"
                f"Año: {item.get('año', 'N/A')}\n"
                f"Ritmo: {item.get('ritmo', 'N/A')}\n"
                f"Emociones: {', '.join(item.get('emociones', []))}\n"
                f"Descripción: {item.get('descripcion', 'Sin descripción')}\n"
                f"Director: {item.get('director', 'N/A')}\n"
                f"Duración: {item.get('duracion', 'N/A')} minutos"
            )
        else:
            contenido = (
                f"Tipo: Serie\n"
                f"Título: {item.get('titulo', 'Sin título')}\n"
                f"Género: {item.get('genero', 'N/A')}\n"
                f"Año: {item.get('año', 'N/A')}\n"
                f"Ritmo: {item.get('ritmo', 'N/A')}\n"
                f"Emociones: {', '.join(item.get('emociones', []))}\n"
                f"Descripción: {item.get('descripcion', 'Sin descripción')}\n"
                f"Creador: {item.get('creador', 'N/A')}\n"
                f"Temporadas: {item.get('temporadas', 'N/A')}\n"
                f"Episodios: {item.get('episodios', 'N/A')}"
            )

        documentos.append(
            Document(
                page_content=contenido,
                metadata={"titulo": item.get("titulo", "Sin título"), "tipo": tipo}
            )
        )

    return documentos


def _firma_catalogo() -> dict:
    def info(path: str) -> dict:
        if not os.path.exists(path):
            return {"exists": False, "sha256": None, "size": 0}

        with open(path, "rb") as f:
            contenido = f.read()

        return {
            "exists": True,
            "sha256": hashlib.sha256(contenido).hexdigest(),
            "size": len(contenido)
        }

    return {
        "version": 2,
        "peliculas": info(PELICULAS_PATH),
        "series": info(SERIES_PATH)
    }


def _indice_actualizado() -> bool:
    if not os.path.exists(FAISS_PATH):
        return False
    if not os.path.exists(INDEX_META_PATH):
        return False

    try:
        with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
            meta_guardada = json.load(f)
    except Exception:
        return False

    return meta_guardada == _firma_catalogo()


def _indice_faiss_existe() -> bool:
    return os.path.exists(os.path.join(FAISS_PATH, "index.faiss"))


# ─────────────────────────────────────────────
# 1. CREAR BASE VECTORIAL FAISS (solo una vez)
# ─────────────────────────────────────────────
def create_vector_db():
    force_reindex = os.getenv("FORCE_REINDEX", "0") == "1"

    if _indice_actualizado():
        return

    if _indice_faiss_existe() and not force_reindex:
        return

    docs = []
    docs.extend(_cargar_catalogo(PELICULAS_PATH, "pelicula"))
    docs.extend(_cargar_catalogo(SERIES_PATH, "serie"))

    if not docs:
        raise ValueError("No hay contenido disponible en peliculas.json o series.json para indexar.")

    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)

    os.makedirs(FAISS_PATH, exist_ok=True)
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(_firma_catalogo(), f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 2. CARGAR FAISS Y CREAR RETRIEVER
# ─────────────────────────────────────────────
def load_vectorstore():
    if not _indice_actualizado():
        create_vector_db()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ─────────────────────────────────────────────
# 3. BUSCAR PELÍCULAS SEGÚN CRITERIOS
# ─────────────────────────────────────────────
def buscar_contenido(criterios: str, tipo: str = "ambas") -> list:
    """Busca en FAISS el contenido más relevante según criterios y tipo."""
    vectorstore = load_vectorstore()

    docs = vectorstore.max_marginal_relevance_search(criterios, k=12, fetch_k=24)

    if tipo in {"pelicula", "serie"}:
        docs_filtrados = [doc for doc in docs if doc.metadata.get("tipo") == tipo]
    else:
        docs_filtrados = docs

    docs_finales = docs_filtrados[:3] if docs_filtrados else docs[:3]
    return [doc.page_content for doc in docs_finales]

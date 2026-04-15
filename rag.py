"""
rag.py - Sistema RAG con FAISS para recomendación de películas
Usa FAISS como retriever (diferente al ChromaDB del profe)
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
import os
import json

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
JSON_PATH = "./peliculas.json"
FAISS_PATH = "./faiss_index"


# ─────────────────────────────────────────────
# 1. CREAR BASE VECTORIAL FAISS (solo una vez)
# ─────────────────────────────────────────────
def create_vector_db():
    if os.path.exists(FAISS_PATH):
        print("La base vectorial FAISS ya existe. No se recrea.")
        return

    print("Creando base vectorial FAISS...")

    # Cargar el JSON de películas
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        peliculas = json.load(f)

    # Convertir cada película en un documento de texto
    from langchain_core.documents import Document

    docs = []
    for p in peliculas:
        contenido = (
            f"Título: {p['titulo']}\n"
            f"Género: {p['genero']}\n"
            f"Año: {p['año']}\n"
            f"Ritmo: {p['ritmo']}\n"
            f"Emociones: {', '.join(p['emociones'])}\n"
            f"Descripción: {p['descripcion']}\n"
            f"Director: {p['director']}\n"
            f"Duración: {p['duracion']} minutos"
        )
        docs.append(Document(page_content=contenido, metadata={"titulo": p["titulo"]}))

    # Embeddings con Gemini (igual que el profe)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # ✅ FAISS en lugar de ChromaDB (retriever diferente al del profe)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_PATH)

    print("Base vectorial FAISS creada.")


# ─────────────────────────────────────────────
# 2. CARGAR FAISS Y CREAR RETRIEVER
# ─────────────────────────────────────────────
def load_retriever():
    if not os.path.exists(FAISS_PATH):
        create_vector_db()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Cargar FAISS desde disco
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retriever con FAISS - search_type="mmr" para mayor variedad
    retriever = vectorstore.as_retriever(
        search_type="mmr",          # MMR: Maximal Marginal Relevance (variedad + relevancia)
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    return retriever


# ─────────────────────────────────────────────
# 3. BUSCAR PELÍCULAS SEGÚN CRITERIOS
# ─────────────────────────────────────────────
def buscar_peliculas(criterios: str) -> list:
    """Busca en FAISS las películas más relevantes según los criterios dados."""
    retriever = load_retriever()
    docs = retriever.invoke(criterios)
    return [doc.page_content for doc in docs]

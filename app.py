"""
app.py - Interfaz Streamlit para el Recomendador de Películas y Series con IA
Estilo del profe: igual a app.py del proyecto RAG de clase
"""

import os

import streamlit as st
from chain import recomendar

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch IA",
    page_icon="🎬",
    layout="wide"
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .titulo-principal {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #F5C518;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .subtitulo {
        text-align: center;
        color: #aaa;
        font-size: 1.1rem;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .mood-chip {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #F5C518;
        color: #F5C518;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin: 0.2rem;
        cursor: pointer;
    }

    .flujo-box {
        background: #1a1a2e;
        border-left: 3px solid #F5C518;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #ccc;
    }

    .flujo-step {
        color: #F5C518;
        font-weight: 600;
    }

    .stApp {
        background-color: #0d0d0d;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="titulo-principal">CineMatch IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Dime cómo te sientes y te recomiendo la película o serie perfecta</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS: Chat vs Investigación
# ─────────────────────────────────────────────
tab_chat, tab_investigacion = st.tabs(["Chat", "Investigación"])

# ─────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────
with tab_chat:
    col_left, col_chat = st.columns([0.45, 2.55])

    with col_left:
        st.markdown("### Ejemplos de búsqueda")
        ejemplos = [
            "Quiero algo para llorar a mares",
            "Necesito adrenalina pura",
            "Algo que me haga pensar mucho",
            "Quiero reírme y sentirme bien",
            "Estoy melancólico y nostálgico",
            "Algo que me impacte fuerte",
            "Quiero una serie de misterio para maratonear",
            "Quiero una historia de amor bonita"
        ]
        for e in ejemplos:
            if st.button(e, key=e, use_container_width=True):
                st.session_state.mood_seleccionado = e

        st.markdown("---")
        st.markdown("### Flujo de procesamiento")
        st.markdown("""
        <div class="flujo-box">
            <span class="flujo-step">Paso 1</span><br>
            Interpreta tu emoción
        </div>
        <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
        <div class="flujo-box">
            <span class="flujo-step">Paso 2</span><br>
            Traduce a criterios de búsqueda
        </div>
        <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
        <div class="flujo-box">
            <span class="flujo-step">Paso 3</span><br>
            Busca en RAG con FAISS (MMR)
        </div>
        <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
        <div class="flujo-box">
            <span class="flujo-step">Paso 4</span><br>
            Genera tu recomendación
        </div>
        """, unsafe_allow_html=True)

    with col_chat:
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = []

        if "last_error" not in st.session_state:
            st.session_state.last_error = None

        for msg in st.session_state.mensajes:
            with st.chat_message(msg["rol"]):
                st.markdown(msg["contenido"])

        mood_inicial = st.session_state.pop("mood_seleccionado", None)
        prompt = st.chat_input("¿Cómo te sientes hoy? ¿Qué quieres ver?")
        entrada = mood_inicial or prompt

        if entrada:
            st.session_state.mensajes.append({"rol": "user", "contenido": entrada})
            with st.chat_message("user"):
                st.markdown(entrada)

            with st.chat_message("assistant"):
                with st.spinner("Buscando la mejor opción para ti..."):
                    try:
                        recomendacion = recomendar(entrada)
                        st.markdown(recomendacion)
                        st.session_state.mensajes.append({
                            "rol": "assistant",
                            "contenido": recomendacion
                        })
                        st.session_state.last_error = None
                    except Exception as e:
                        st.session_state.last_error = f"{type(e).__name__}: {e}"
                        st.error("No pude generar la recomendación. Revisa la configuración o los logs.")
                        if os.getenv("STREAMLIT_DEBUG", "0") == "1":
                            st.exception(e)

        if st.session_state.last_error:
            st.warning(st.session_state.last_error)

# ─────────────────────────────────────────────
# TAB 2: INVESTIGACIÓN (Retrievers)
# ─────────────────────────────────────────────
with tab_investigacion:
    st.markdown("""
    <style>
    .retriever-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .retriever-title {
        color: #F5C518;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .retriever-type {
        display: inline-block;
        background: #252540;
        border: 1px solid #F5C518;
        color: #F5C518;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    
    .retriever-description {
        color: #ccc;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .intro-section {
        background: #1a1a2e;
        border-left: 3px solid #F5C518;
        padding: 1.5rem;
        border-radius: 4px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## Investigación: Retrievers en LangChain")
    st.markdown("Una exploración de diferentes estrategias de recuperación de documentos en sistemas RAG")
    
    st.markdown("""
    <div class="intro-section">
    <h3 style="margin-top: 0; color: #F5C518;">Contexto del proyecto</h3>
    <p style="color: #aaa; margin-bottom: 0;">
    Este proyecto investigó 5 retrievers diferentes a los presentados en clase (similarity básico y MultiQueryRetriever).
    El sistema actual utiliza <strong>FAISS con búsqueda MMR</strong> para recuperar documentos variados y relevantes.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Concepto fundamental
    with st.expander("¿Qué es un Retriever?", expanded=True):
        st.markdown("""
        Un retriever es el componente del sistema RAG responsable de **recuperar documentos relevantes** de una base de datos 
        dada una consulta del usuario. Diferentes retrievers usan estrategias distintas para determinar qué documentos son "relevantes".
        
        **Rol crítico:** El retriever actúa como puente entre la pregunta del usuario y el conocimiento contenido en los documentos.
        La calidad del retriever impacta directamente en la calidad de las respuestas finales del sistema.
        """)
    
    st.markdown("")
    
    # Tabla comparativa al inicio
    st.markdown("### Comparativa general")
    comparativa = {
        "Retriever": [
            "Similarity",
            "MultiQuery",
            "FAISS + MMR (proyecto)",
            "BM25",
            "Ensemble",
            "Contextual Compression",
            "WebResearch"
        ],
        "Estrategia": [
            "Vectorial",
            "Vectorial + LLM",
            "Vectorial",
            "Palabras clave",
            "Híbrido",
            "Vectorial + filtrado",
            "Búsqueda en internet"
        ],
        "Velocidad": [
            "Rápido",
            "Lento",
            "Rápido",
            "Muy rápido",
            "Rápido",
            "Lento",
            "Muy lento"
        ],
        "Caso de uso": [
            "Búsqueda semántica simple",
            "Preguntas ambiguas",
            "Resultados variados",
            "Sin costos de API",
            "Máxima precisión",
            "Documentos muy largos",
            "Información actualizada"
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparativa)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("### Describe de cada retriever")
    
    # FAISS + MMR
    with st.expander("FAISS Retriever con MMR", expanded=False):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("""
            **Estrategia:** Vectorial  
            **Dependencias:** Embeddings  
            **Complejidad:** Media
            """)
            st.info("Utilizado en este proyecto")
        
        with col2:
            st.markdown("""
            **MMR** (Maximal Marginal Relevance) busca documentos que sean simultáneamente relevantes a la consulta y distintos entre sí. 
            A diferencia de similarity básico que retorna los K más parecidos (potencialmente redundantes), MMR garantiza diversidad en los resultados.
            """)
        
        st.markdown("**Implementación:**")
        st.code("""retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)""", language="python")
        
        st.markdown("**Cuándo usarlo:** Cuando necesitas variedad en los resultados y evitar información redundante.")
    
    # BM25
    with st.expander("BM25 Retriever", expanded=False):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("""
            **Estrategia:** Palabras clave  
            **Dependencias:** Ninguna (solo texto)  
            **Complejidad:** Baja
            """)
            st.success("Sin costos de API")
        
        with col2:
            st.markdown("""
            Algoritmo clásico de recuperación basado en frecuencia de términos. Es el motor de búsqueda detrás de Elasticsearch. 
            No requiere embeddings vectoriales, funciona únicamente con análisis léxico de los documentos.
            """)
        
        st.markdown("**Implementación:**")
        st.code("""from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(docs)
retriever.k = 3""", language="python")
        
        st.markdown("**Cuándo usarlo:** En prototipado rápido, cuando no tienes API keys, o para searches rápidas sin costos.")
    
    # Ensemble
    with st.expander("Ensemble Retriever", expanded=False):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("""
            **Estrategia:** Híbrida  
            **Dependencias:** 2+ retrievers  
            **Complejidad:** Alta
            """)
            st.warning("Requiere configuración de múltiples fuentes")
        
        with col2:
            st.markdown("""
            Combina dos o más retrievers y fusiona sus resultados usando RRF (Reciprocal Rank Fusion). 
            Ejemplo: combinar BM25 (palabras clave) + FAISS (vectorial) para aprovechar las fortalezas de ambos.
            """)
        
        st.markdown("**Implementación:**")
        st.code("""ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)""", language="python")
        
        st.markdown("**Cuándo usarlo:** Cuando necesitas máxima calidad combinando múltiples estrategias de búsqueda.")
    
    # Contextual Compression
    with st.expander("Contextual Compression Retriever", expanded=False):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("""
            **Estrategia:** Vectorial + filtrado  
            **Dependencias:** LLM  
            **Complejidad:** Alta
            """)
            st.warning("Mayor costo computacional")
        
        with col2:
            st.markdown("""
            Recupera documentos normalmente, luego usa un LLM o compresor para **filtrar y resumir** 
            cada documento, quedándose solo con la sección relevante para la pregunta.
            """)
        
        st.markdown("**Implementación:**")
        st.code("""from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)""", language="python")
        
        st.markdown("**Cuándo usarlo:** Con documentos largos donde solo pequeñas porciones responden la pregunta.")
    
    # WebResearch
    with st.expander("WebResearch Retriever", expanded=False):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("""
            **Estrategia:** Internet en tiempo real  
            **Dependencias:** Google Search API  
            **Complejidad:** Muy alta
            """)
            st.warning("Costos variables + latencia")
        
        with col2:
            st.markdown("""
            Busca en internet usando Google Search API y vectoriza los resultados en tiempo real. 
            Es esencialmente un RAG sobre la web, ideal para información actualizada.
            """)
        
        st.markdown("**Implementación:**")
        st.code("""from langchain_community.retrievers.web_research import WebResearchRetriever

retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=google_search
)""", language="python")
        
        st.markdown("**Cuándo usarlo:** Cuando necesitas información actualizada no disponible en documentos locales.")


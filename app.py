"""
app.py - Interfaz Streamlit para el Recomendador de Películas con IA
Estilo del profe: igual a app.py del proyecto RAG de clase
"""

import streamlit as st
from chain import recomendar
from rag import create_vector_db

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch IA",
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
# INICIALIZACIÓN (crear FAISS si no existe)
# ─────────────────────────────────────────────
if "db_creada" not in st.session_state:
    with st.spinner("Iniciando base de datos de películas..."):
        create_vector_db()
    st.session_state.db_creada = True

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="titulo-principal">🎬 CineMatch IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Dime cómo te sientes y te recomiendo la película perfecta</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT: dos columnas
# ─────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_info:
    st.markdown("### 🔗 Flujo de IA")
    st.markdown("""
    <div class="flujo-box">
        <span class="flujo-step">Runnable 1</span><br>
        Interpreta tu emoción
    </div>
    <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
    <div class="flujo-box">
        <span class="flujo-step">Runnable 2</span><br>
        Traduce a criterios de búsqueda
    </div>
    <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
    <div class="flujo-box">
        <span class="flujo-step">Runnable 3</span><br>
        Busca en RAG con FAISS (MMR)
    </div>
    <div style="text-align:center; color:#F5C518; font-size:1.2rem">↓</div>
    <div class="flujo-box">
        <span class="flujo-step">Runnable 4</span><br>
        Genera tu recomendación
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Ejemplos de mood")
    ejemplos = [
        "Quiero algo para llorar a mares",
        "Necesito adrenalina pura",
        "Algo que me haga pensar mucho",
        "Quiero reírme y sentirme bien",
        "Estoy melancólico y nostálgico",
        "Algo que me impacte fuerte",
        "Quiero una historia de amor bonita",
        "Algo raro y diferente a todo"
    ]
    for e in ejemplos:
        if st.button(e, key=e, use_container_width=True):
            st.session_state.mood_seleccionado = e

with col_chat:
    # Historial de mensajes
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    # Mostrar historial
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["rol"]):
            st.markdown(msg["contenido"])

    # Si se seleccionó un ejemplo, usarlo como input
    mood_inicial = st.session_state.pop("mood_seleccionado", None)

    # Input del usuario
    prompt = st.chat_input("¿Cómo te sientes hoy? ¿Qué quieres ver?")

    # Si hay mood de ejemplo o input directo
    entrada = mood_inicial or prompt

    if entrada:
        # Mostrar mensaje del usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": entrada})
        with st.chat_message("user"):
            st.markdown(entrada)

        # Generar recomendación
        with st.chat_message("assistant"):
            with st.spinner("🎬 Buscando la película perfecta para ti..."):
                try:
                    recomendacion = recomendar(entrada)
                    st.markdown(recomendacion)
                    st.session_state.mensajes.append({
                        "rol": "assistant",
                        "contenido": recomendacion
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

        st.rerun()

"""
app.py - Interfaz Streamlit para el Recomendador de Películas y Series con IA
Estilo del profe: igual a app.py del proyecto RAG de clase
"""

import logging
import os
from datetime import datetime

import streamlit as st
from chain import recomendar


logger = logging.getLogger("cinematch")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
st.markdown('<h1 class="titulo-principal">🎬 CineMatch IA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Dime cómo te sientes y te recomiendo la película o serie perfecta</p>', unsafe_allow_html=True)

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
        "Quiero una serie de misterio para maratonear",
        "Quiero una historia de amor bonita"
    ]
    for e in ejemplos:
        if st.button(e, key=e, use_container_width=True):
            st.session_state.mood_seleccionado = e

with col_chat:
    # Historial de mensajes
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    if "debug_events" not in st.session_state:
        st.session_state.debug_events = []

    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    if "log_counter" not in st.session_state:
        st.session_state.log_counter = 0

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
        logger.info("Usuario envió prompt: %s", entrada)
        st.session_state.debug_events = []

        def emit_log(message: str) -> None:
            st.session_state.log_counter += 1
            entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            st.session_state.debug_events.append(entry)
            logger.info(message)
            live_log_panel.code("\n".join(st.session_state.debug_events[-20:]), language="text")

        st.session_state.debug_events.append(f"[{datetime.now().strftime('%H:%M:%S')}] Prompt recibido: {entrada}")

        # Mostrar mensaje del usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": entrada})
        with st.chat_message("user"):
            st.markdown(entrada)

        # Generar recomendación
        hubo_error = False
        with st.chat_message("assistant"):
            st.markdown("### Diagnóstico en tiempo real")
            live_log_panel = st.empty()
            live_log_panel.code("\n".join(st.session_state.debug_events[-20:]), language="text")
            with st.spinner("🎬 Buscando la mejor opción para ti..."):
                try:
                    emit_log("Iniciando pipeline de recomendación...")
                    recomendacion = recomendar(entrada, emit=emit_log)
                    st.markdown(recomendacion)
                    st.session_state.mensajes.append({
                        "rol": "assistant",
                        "contenido": recomendacion
                    })
                    st.session_state.last_error = None
                    emit_log("Recomendación generada correctamente")
                except Exception as e:
                    hubo_error = True
                    st.session_state.last_error = f"{type(e).__name__}: {e}"
                    st.session_state.debug_events.append(f"[{datetime.now().strftime('%H:%M:%S')}] {st.session_state.last_error}")
                    logger.exception("Error al generar recomendación")
                    st.error("No pude generar la recomendación. Revisa la configuración o los logs.")
                    if os.getenv("STREAMLIT_DEBUG", "0") == "1":
                        st.exception(e)

    with st.expander("Diagnóstico", expanded=False):
        if st.session_state.last_error:
            st.warning(st.session_state.last_error)
        for event in st.session_state.debug_events[-10:]:
            st.caption(event)

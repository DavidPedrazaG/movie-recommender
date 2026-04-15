"""
chain.py - Flujo de IA con 4 Runnables encadenados
Igual a como lo enseñó el profe: runnable1 | runnable2 | runnable3 | runnable4
"""

from dotenv import load_dotenv
load_dotenv()

import re
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import buscar_contenido

# ─────────────────────────────────────────────
# LLM - igual al del profe
# ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# ─────────────────────────────────────────────
# RUNNABLE 1 - Interpreta la emoción del usuario
# Entrada: {"mood": "quiero algo para llorar"}
# Salida:  {"mood": ..., "emocion": "tristeza profunda, melancolía"}
# ─────────────────────────────────────────────
def interpretar_emocion(datos: dict) -> dict:
    template = PromptTemplate(
        template=(
            "El usuario dice: '{mood}'\n\n"
            "Identifica en UNA sola línea las 2-3 emociones principales que busca "
            "en el contenido audiovisual (película o serie) (ej: tristeza, nostalgia, reflexión). "
            "Solo las emociones, sin explicación."
        ),
        input_variables=["mood"]
    )
    chain = template | llm
    resultado = chain.invoke({"mood": datos["mood"]})
    return {**datos, "emocion": resultado.content.strip()}


# ─────────────────────────────────────────────
# RUNNABLE 2 - Traduce emoción a criterios de búsqueda
# Entrada: {"mood": ..., "emocion": "tristeza, nostalgia"}
# Salida:  {"mood": ..., "emocion": ..., "criterios": "drama pausado amor pérdida..."}
# ─────────────────────────────────────────────
def traducir_a_criterios(datos: dict) -> dict:
    objetivo = {
        "pelicula": "películas",
        "serie": "series",
        "ambas": "películas y series"
    }.get(datos.get("tipo", "ambas"), "películas y series")

    template = PromptTemplate(
        template=(
            "Emociones buscadas: {emocion}\n\n"
            "Tipo de contenido objetivo: {objetivo}\n\n"
            "Traduce esto a criterios de búsqueda de contenido audiovisual: géneros, "
            "tipo de ritmo, temas y palabras clave relevantes. "
            "Responde solo con las palabras clave separadas por comas, sin listas ni bullets."
        ),
        input_variables=["emocion", "objetivo"]
    )
    chain = template | llm
    resultado = chain.invoke({"emocion": datos["emocion"], "objetivo": objetivo})
    return {**datos, "criterios": resultado.content.strip()}


# ─────────────────────────────────────────────
# RUNNABLE 3 - Busca en la base de datos RAG (FAISS)
# Entrada: {"mood": ..., "emocion": ..., "criterios": "..."}
# Salida:  {"mood": ..., "emocion": ..., "criterios": ..., "contenido": [...]}
# ─────────────────────────────────────────────
def buscar_en_rag(datos: dict) -> dict:
    contenido_encontrado = buscar_contenido(datos["criterios"], datos.get("tipo", "ambas"))
    return {**datos, "contenido": contenido_encontrado}


# ─────────────────────────────────────────────
# RUNNABLE 4 - Genera la recomendación final
# Entrada: {"mood": ..., "emocion": ..., "criterios": ..., "contenido": [...]}
# Salida:  string con la recomendación
# ─────────────────────────────────────────────
def generar_recomendacion(datos: dict) -> str:
    contexto = "\n\n---\n\n".join(datos["contenido"])
    objetivo = {
        "pelicula": "películas",
        "serie": "series",
        "ambas": "películas o series"
    }.get(datos.get("tipo", "ambas"), "películas o series")

    template = PromptTemplate(
        template=(
            "El usuario quiere: '{mood}'\n"
            "Tipo solicitado: {objetivo}\n"
            "Emociones identificadas: {emocion}\n\n"
            "Basándote SOLO en este contenido del catálogo:\n\n"
            "{contexto}\n\n"
            "Recomienda las 2-3 mejores opciones del tipo solicitado. Para cada una explica:\n"
            "- Por qué encaja con lo que el usuario siente\n"
            "- Qué va a experimentar viéndola\n"
            "- Una frase que la describa perfectamente\n\n"
            "Responde en español, con un tono cálido y personal, "
            "como si fuera un amigo que conoce mucho de cine."
        ),
        input_variables=["mood", "emocion", "contexto", "objetivo"]
    )
    chain = template | llm
    resultado = chain.invoke({
        "mood": datos["mood"],
        "emocion": datos["emocion"],
        "contexto": contexto,
        "objetivo": objetivo
    })
    return resultado.content


# ─────────────────────────────────────────────
# CADENA PRINCIPAL: 4 RUNNABLES ENCADENADOS
# Igual a la estructura que enseñó el profe
# ─────────────────────────────────────────────
runnable1 = RunnableLambda(interpretar_emocion)   # Interpreta emoción
runnable2 = RunnableLambda(traducir_a_criterios)  # Traduce a criterios
runnable3 = RunnableLambda(buscar_en_rag)         # Busca en RAG/FAISS
runnable4 = RunnableLambda(generar_recomendacion) # Genera recomendación

cadena = runnable1 | runnable2 | runnable3 | runnable4


def recomendar(mood: str) -> str:
    """Función principal: recibe el mood y devuelve la recomendación."""
    tipo = inferir_tipo_contenido(mood)
    return cadena.invoke({"mood": mood, "tipo": tipo})


def inferir_tipo_contenido(texto: str) -> str:
    texto_normalizado = texto.lower()

    palabras_series = ["serie", "series", "tv", "temporada", "episodio", "episodios", "miniserie"]
    palabras_peliculas = ["pelicula", "peliculas", "film", "cine", "largometraje"]

    hay_series = any(re.search(rf"\\b{p}\\b", texto_normalizado) for p in palabras_series)
    hay_peliculas = any(re.search(rf"\\b{p}\\b", texto_normalizado) for p in palabras_peliculas)

    if hay_series and not hay_peliculas:
        return "serie"
    if hay_peliculas and not hay_series:
        return "pelicula"
    return "ambas"

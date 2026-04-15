"""
chain.py - Flujo de IA con 4 Runnables encadenados
Igual a como lo enseñó el profe: runnable1 | runnable2 | runnable3 | runnable4
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import buscar_peliculas

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
            "en una película (ej: tristeza, nostalgia, reflexión). "
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
    template = PromptTemplate(
        template=(
            "Emociones buscadas: {emocion}\n\n"
            "Traduce esto a criterios de búsqueda de películas: géneros, "
            "tipo de ritmo, temas y palabras clave relevantes. "
            "Responde solo con las palabras clave separadas por comas, sin listas ni bullets."
        ),
        input_variables=["emocion"]
    )
    chain = template | llm
    resultado = chain.invoke({"emocion": datos["emocion"]})
    return {**datos, "criterios": resultado.content.strip()}


# ─────────────────────────────────────────────
# RUNNABLE 3 - Busca en la base de datos RAG (FAISS)
# Entrada: {"mood": ..., "emocion": ..., "criterios": "..."}
# Salida:  {"mood": ..., "emocion": ..., "criterios": ..., "peliculas": [...]}
# ─────────────────────────────────────────────
def buscar_en_rag(datos: dict) -> dict:
    peliculas_encontradas = buscar_peliculas(datos["criterios"])
    return {**datos, "peliculas": peliculas_encontradas}


# ─────────────────────────────────────────────
# RUNNABLE 4 - Genera la recomendación final
# Entrada: {"mood": ..., "emocion": ..., "criterios": ..., "peliculas": [...]}
# Salida:  string con la recomendación
# ─────────────────────────────────────────────
def generar_recomendacion(datos: dict) -> str:
    contexto = "\n\n---\n\n".join(datos["peliculas"])

    template = PromptTemplate(
        template=(
            "El usuario quiere: '{mood}'\n"
            "Emociones identificadas: {emocion}\n\n"
            "Basándote SOLO en estas películas del catálogo:\n\n"
            "{contexto}\n\n"
            "Recomienda las 2-3 mejores opciones. Para cada una explica:\n"
            "- Por qué encaja con lo que el usuario siente\n"
            "- Qué va a experimentar viéndola\n"
            "- Una frase que la describa perfectamente\n\n"
            "Responde en español, con un tono cálido y personal, "
            "como si fuera un amigo que conoce mucho de cine."
        ),
        input_variables=["mood", "emocion", "contexto"]
    )
    chain = template | llm
    resultado = chain.invoke({
        "mood": datos["mood"],
        "emocion": datos["emocion"],
        "contexto": contexto
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
    return cadena.invoke({"mood": mood})

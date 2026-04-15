# 🔍 Investigación: 5 Retrievers de LangChain

> Punto 2 del proyecto: investigar 5 retrievers diferentes a los vistos en clase.
> El profe usó: `similarity` básico y `MultiQueryRetriever`.

---

## ¿Qué es un Retriever?

Un retriever es el componente del sistema RAG que **recupera documentos relevantes** de una base de datos dado una consulta. Diferentes retrievers usan distintas estrategias para decidir qué documentos son "relevantes".

---

## 1. 🏆 FAISS Retriever con MMR
**`FAISS.as_retriever(search_type="mmr")`**

**¿Cómo funciona?**
MMR = Maximal Marginal Relevance. Busca documentos que sean relevantes a la consulta **Y** distintos entre sí. Evita devolver 3 documentos que digan lo mismo.

**Diferencia con similarity básico:**
- `similarity` devuelve los K más parecidos (pueden repetirse ideas)
- `MMR` devuelve los K más relevantes **y variados**

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

**Cuándo usarlo:** Cuando quieres variedad en los resultados (ej: recomendaciones de películas distintas entre sí).

**Usado en este proyecto ✅**

---

## 2. 🔑 BM25 Retriever
**`BM25Retriever`** — `langchain_community.retrievers`

**¿Cómo funciona?**
Es un algoritmo de búsqueda por **palabras clave** (no vectorial). Analiza la frecuencia de las palabras en los documentos y ranquea por relevancia léxica. Es el algoritmo detrás de motores como Elasticsearch.

**No necesita embeddings ni API** — funciona puro con texto.

```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(docs)
retriever.k = 3
resultados = retriever.invoke("película de terror psicológico")
```

**Cuándo usarlo:** Cuando no tienes API key o quieres búsqueda rápida sin costos de embeddings. Ideal para prototipado.

---

## 3. 🤝 Ensemble Retriever
**`EnsembleRetriever`** — `langchain.retrievers`

**¿Cómo funciona?**
Combina **dos o más retrievers** y fusiona sus resultados usando el algoritmo RRF (Reciprocal Rank Fusion). Ejemplo clásico: combinar BM25 (palabras clave) + FAISS (vectorial) para aprovechar lo mejor de ambos.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]  # 50% cada uno
)
```

**Cuándo usarlo:** Cuando quieres máxima calidad de búsqueda combinando estrategias distintas.

---

## 4. 🗜️ Contextual Compression Retriever
**`ContextualCompressionRetriever`** — `langchain.retrievers`

**¿Cómo funciona?**
Primero recupera documentos normalmente, luego usa un LLM o un **compresor** para **filtrar y comprimir** cada documento, quedándose solo con la parte que realmente responde la pregunta.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

**Cuándo usarlo:** Cuando los documentos son muy largos y solo una pequeña parte responde la pregunta. Reduce el ruido en el contexto.

---

## 5. 🌐 WebResearch Retriever
**`WebResearchRetriever`** — `langchain_community.retrievers`

**¿Cómo funciona?**
Usa un LLM para generar queries de búsqueda, luego **busca en internet** (Google Search API) y vectoriza los resultados para recuperarlos. Básicamente un RAG sobre internet en tiempo real.

```python
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search
)
```

**Cuándo usarlo:** Cuando necesitas información actualizada que no está en tus documentos locales. Por ejemplo, noticias recientes, precios actuales, etc.

---

## 📊 Tabla Comparativa

| Retriever | Tipo | Necesita API | Velocidad | Mejor para |
|---|---|---|---|---|
| Similarity (profe) | Vectorial | ✅ Sí | ⚡ Rápido | Búsqueda semántica básica |
| MultiQuery (profe) | Vectorial + LLM | ✅ Sí | 🐢 Lento | Preguntas ambiguas |
| **FAISS + MMR** ← este proyecto | Vectorial | ✅ Sí | ⚡ Rápido | Resultados variados |
| BM25 | Palabras clave | ❌ No | ⚡⚡ Muy rápido | Sin embeddings |
| Ensemble | Híbrido | Opcional | ⚡ Rápido | Máxima precisión |
| Contextual Compression | Vectorial + compresión | ✅ Sí | 🐢 Lento | Documentos largos |
| WebResearch | Internet | ✅ Sí | 🐢🐢 Muy lento | Datos en tiempo real |

---

*Proyecto: Recomendador de Películas con RAG — LangChain + Gemini*

# paperRag — Instrucciones para Claude

## Qué es este proyecto

Sistema RAG basado en agentes LangGraph para consultas sobre referencias en papers científicos. Es la tesis de grado del usuario. Interfaz en Streamlit, todos los modelos vía OpenRouter (una sola API key).

## Entorno de desarrollo

**SIEMPRE usar el entorno virtual:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para correr la app:
```bash
source .venv/bin/activate
streamlit run app.py
```

## Arquitectura

```
paperRag/
├── app.py                  # Streamlit UI — upload PDF + consultas
├── requirements.txt
├── .env                    # OPENROUTER_API_KEY (única key necesaria)
├── data/papers/            # Papers procesados (auto-creado)
│   └── <sanitized_name>/
│       ├── meta.json           # Metadata mínima del paper
│       ├── chroma_db/          # Vectorstore granular (párrafos)
│       └── faiss_index/        # Vectorstore por sección completa
└── src/
    ├── config.py           # Factory LLMs + embeddings → OpenRouter
    ├── utils.py            # Helpers puros (sanitize, parse, hash, normalize_metadata)
    ├── vectorstore.py      # Load/save Chroma + FAISS
    ├── processor.py        # Pipeline PDF → Markdown → secciones → embeddings
    └── agent.py            # LangGraph 5 nodos (closure-based)
```

## Modelos (todos vía OpenRouter)

| Rol | Modelo |
|-----|--------|
| Procesamiento PDF (título, secciones, refs) | `google/gemini-2.5-flash-lite` |
| Agente RAG (generación de respuestas) | `openai/gpt-4o-mini` (default; configurable) |
| Embeddings | `openai/text-embedding-ada-002` |

## Pipeline de procesamiento (`src/processor.py`)

1. **PDF → Markdown** — `docling` (`DocumentConverter`), sin LLM. Preserva estructura mejor que pymupdf.
2. **Análisis con LLM en paralelo** — 3 calls concurrentes (`ThreadPoolExecutor`):
   - Extraer título (primeros 2 000 chars)
   - Identificar secciones (primeros 40 000 chars)
   - Extraer referencias bibliográficas (últimos 30 000 chars)
3. **Chunking dual**:
   - Chroma: párrafos (granular, con metadata de referencias)
   - FAISS: secciones completas (para resúmenes y vista amplia)
   - Metadata uniforme entre ambos vía `normalize_metadata` (listas/dicts → strings).
4. **Persistencia** — vectorstores + `meta.json` en `data/papers/<sanitized>/`. El `meta.json` tiene solo lo mínimo (título, secciones, hash, modelos, timestamps); las `resolved_references` se reconstruyen desde la metadata del FAISS al cargar.

**Deduplicación por hash**: antes de procesar, se calcula `sha256` del PDF y se chequea si existe un paper con ese hash en `data/papers/`. Si lo hay, se carga directo sin reprocesar.

**Migración**: papers viejos con `processed_data.pkl` se migran lazy a `meta.json` la primera vez que se cargan. El pkl original queda como `.bak`.

## Agente LangGraph (`src/agent.py`)

5 nodos secuenciales, compilados como closures (no usan variables globales):

```
analyze_query → select_vectorstore → extract_filters → execute_search → generate_answer
```

**Tipos de consulta detectados:**
- `reference_sections` → chroma (filtro por ref `[N]`)
- `section_references` → faiss (sección completa)
- `reference_context` → chroma (contexto detallado)
- `section_summary` → faiss (resumen de sección)
- `general_search` → chroma (query corta) / faiss (query larga)

## Session state (`app.py`)

```python
st.session_state["paper"] = {
    "paper_title", "paper_name", "sanitized_name",
    "sections", "resolved_references",
    "chroma_store", "faiss_store",
    "sections_count", "refs_count"
}
st.session_state["query_history"] = []  # lista de resultados
```

El agente compilado **no** vive en session_state: se cachea con `@st.cache_resource` keyed por `(sanitized_name, model_id)`. Embeddings y vectorstores también están cacheados — no se reinstancian entre reruns.

## Variables de entorno

```
OPENROUTER_API_KEY=sk-or-...
```

## Notas macOS

- **OpenMP duplicado (`OMP: Error #15`)**: `faiss-cpu` y docling/numpy cargan `libomp.dylib` dos veces en macOS, causando un `abort`. Fix aplicado: `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` al inicio de `app.py` antes de cualquier import.

## Dependencias clave

- `langchain-openai` — ChatOpenAI compatible con OpenRouter via `base_url`. Embeddings se hacen vía `OpenRouterEmbeddings` (wrapper propio en `config.py`) para no depender del client de OpenAI directo.
- `langchain-chroma` — vector store persistente con filtros de metadata
- `langchain-community` — FAISS integration
- `langgraph` — orquestación del agente
- `docling` — conversión PDF → Markdown estructurado (mejor que pymupdf para papers científicos)
- `chromadb` — backend de Chroma
- `faiss-cpu` — backend de FAISS

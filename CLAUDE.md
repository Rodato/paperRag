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
    ├── agent.py            # Grafo de recuperación (closures) + generación streameable
    └── history.py          # Persistencia del historial de consultas por paper
```

`data/papers/<sanitized>/` también guarda `history.json` (historial de consultas, más reciente primero, tope 100 entradas).

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

**Nombre del paper**: la carpeta y los `chunk_id` se derivan del **nombre real del PDF subido** (`original_filename`, ruteado desde `app.py` → `run_process_pdf` → `process_pdf`), no del path temporal. `sanitize_collection_name` es determinista (fallback con sha1, no `hash()`).

**Secciones persistidas**: `meta.json["sections"]` guarda **solo las secciones realmente indexadas** (`dict.fromkeys` sobre `sections_data`), no la lista cruda del LLM. Los headers padre sin texto propio (su contenido vive en subsecciones) se descartan en `_split_by_sections` y nunca se indexan; persistir la lista cruda haría que la UI ofrezca secciones fantasma y que el agente filtre por un `section_title` ausente → 0 resultados.

**Deduplicación por hash**: antes de procesar, se calcula `sha256` del PDF y se chequea si existe un paper con ese hash en `data/papers/`. Si lo hay, se carga directo sin reprocesar.

**Migración**: papers viejos con `processed_data.pkl` se migran lazy a `meta.json` la primera vez que se cargan. El pkl original queda como `.bak`.

## Agente LangGraph (`src/agent.py`)

**Recuperación y generación están separadas.** `build_agent` compila un grafo LangGraph de **4 nodos** (la fase de recuperación) y devuelve un `CompiledAgent` (NamedTuple) con `retrieval` (grafo), `build_prompt` (closure que arma el prompt desde el estado) y `llm`:

```
analyze_query → select_vectorstore → extract_filters → execute_search   # grafo
build_prompt(state) + llm.invoke / llm.stream                            # generación, fuera del grafo
```

Todos los nodos son closures que capturan stores/secciones/refs (sin globales).

**Por qué separadas**: permite **streamear** la respuesta token a token. La UI usa `run_query_stream(agent, query)` → `(token_iterator, finalize)`: se consume el iterador con `st.write_stream` y luego `finalize(answer)` arma el dict de resultado con su metadata. `run_query` (no-streaming) sigue disponible para tests/uso programático.

**Tipos de consulta detectados:**
- `reference_sections` → chroma (filtro por ref `[N]`)
- `section_references` → faiss (sección completa)
- `reference_context` → chroma (contexto detallado)
- `section_summary` → faiss (resumen de sección)
- `general_search` → chroma (query corta) / faiss (query larga)

La detección es por keywords (frágil — candidato a reemplazar por clasificación con LLM).

**Recuperación con filtro de metadata**: cuando hay `section_title` o `reference_number`, `_search_faiss` lee los docs que matchean **directo del docstore** (`faiss_store.docstore._dict`), no `similarity_search(k)`-y-después-filtrar. Con pocos chunks por paper, la sección buscada podía quedar fuera del top-k y descartarse → 0 resultados. Chroma sí usa filtro nativo (`$eq`).

**Guard anti-alucinación**: si la búsqueda no devuelve resultados, `build_prompt` instruye explícitamente a admitir que no se encontró info y sugerir reformular, en vez de inventar. Ventana de contexto: `CONTEXT_CHARS_PER_RESULT` (1200) × `MAX_CONTEXT_RESULTS` (5).

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

`requirements.txt` está **pinneado** (cota inferior = versión instalada, cota superior en el próximo major) para reproducibilidad — el ecosistema langchain rompe entre majors.

- `langchain-openai` — `ChatOpenAI` compatible con OpenRouter vía `base_url` (LLMs de procesamiento y de consulta).
- `openai` — usado **directamente** en `config.py`: `OpenRouterEmbeddings` instancia `openai.OpenAI(base_url=...)` para los embeddings (langchain-openai no expone bien embeddings sobre OpenRouter). Es dependencia explícita, no solo transitiva.
- `langchain-chroma` — vector store persistente con filtros de metadata
- `langchain-community` — FAISS integration
- `langgraph` — orquestación del grafo de recuperación
- `docling` — conversión PDF → Markdown estructurado (mejor que pymupdf para papers científicos)
- `chromadb` — backend de Chroma
- `faiss-cpu` — backend de FAISS

## UI / diseño (`app.py`)

- Estética editorial: tipografía **serif** (Source Serif 4) para wordmark/títulos/hero, **Inter** para la UI, paleta cálida tipo papel. Todo vía **variables CSS** (`:root`) inyectadas en `_inject_css()`.
- **Cuidado con los selectores de fuente**: NO usar `[class*="st-"]` para `font-family` — pisa la fuente de íconos de Streamlit (Material Symbols) y los íconos salen como texto (expander, uploader). Acotar a `html, body, .stApp` y mantener el guard explícito que preserva la fuente de íconos.
- Respuestas streameadas con `st.write_stream`; tras el stream se persiste en `history.json` y se hace `st.rerun()` (la respuesta queda renderizada desde el historial).

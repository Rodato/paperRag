# paperRag

Sistema multi-agente basado en RAG y LLMs para el **análisis contextual de citas** en documentos científicos.

Dado un artículo en PDF, el sistema permite formular preguntas en lenguaje natural sobre sus
referencias —dónde se cita un trabajo, con qué propósito, qué referencias usa una sección— y
devuelve respuestas fundamentadas en fragmentos recuperados del propio documento.

Código asociado al artículo enviado a la revista **TecnoLógicas** (Universidad del Valle, Cali, Colombia).

---

## Cómo funciona

El pipeline tiene dos mitades: **indexación** (una vez por artículo) y **consulta** (por pregunta).

**Indexación.** El PDF se convierte a Markdown estructurado con `docling`. Tres llamadas
concurrentes a un LLM extraen el título, la lista de secciones y la bibliografía. El texto se
fragmenta con dos granularidades complementarias y se vectoriza:

| Índice | Granularidad | Para qué |
|---|---|---|
| **ChromaDB** | párrafos (200-800 tokens) | consultas específicas con filtrado por metadatos |
| **FAISS** | secciones completas (1000-3000 tokens) | resúmenes y preguntas de alcance amplio |

**Consulta.** Un grafo de estados de LangGraph encadena cinco agentes especializados:

```
analizador → selector → extractor → buscador → generador
  ↓            ↓          ↓           ↓          ↓
tipo de     ¿Chroma    filtros    recuperación  respuesta
consulta    o FAISS?   de sección  con filtros  fundamentada
                       y referencia
```

La recuperación y la generación están separadas: el grafo cubre los cuatro primeros agentes y la
generación queda fuera para poder transmitirse token a token. Si la búsqueda no devuelve
resultados, el generador está instruido para admitirlo explícitamente en lugar de inventar.

## Instalación

Requiere **Python 3.11+** y una clave de [OpenRouter](https://openrouter.ai) (un solo proveedor
para todos los modelos: LLMs de procesamiento, de consulta y embeddings).

```bash
git clone https://github.com/Rodato/paperRag.git
cd paperRag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "OPENROUTER_API_KEY=sk-or-..." > .env
streamlit run app.py
```

## Modelos

| Rol | Modelo por defecto |
|---|---|
| Procesamiento del PDF (título, secciones, referencias) | `google/gemini-2.5-flash-lite` |
| Agente de consulta (generación) | `openai/gpt-4o-mini` (configurable en la interfaz) |
| Embeddings | `openai/text-embedding-ada-002` |

El modelo de consulta se elige desde la interfaz. Los resultados publicados en el artículo se
obtuvieron con GPT-4o Mini, Gemini 2.0 Flash, DeepSeek R1 Distill Qwen 32B, Ministral 8B y
Llama 3.3 70B Instruct.

## Estructura

```
app.py              Interfaz Streamlit: carga de PDF y consultas
src/
  config.py         Factory de LLMs y embeddings sobre OpenRouter
  processor.py      Pipeline PDF → Markdown → secciones → embeddings
  vectorstore.py    Persistencia de Chroma y FAISS
  agent.py          Grafo de recuperación + generación transmitible
  history.py        Historial de consultas por artículo
  utils.py          Helpers puros
data/papers/        Artículos procesados (generado, no versionado)
```

## Notas

- **macOS**: `faiss-cpu` y `docling` cargan `libomp.dylib` dos veces y provocan un `abort`. Se
  mitiga con `KMP_DUPLICATE_LIB_OK=TRUE`, fijado al inicio de `app.py` antes de cualquier import.
- Las dependencias están fijadas con cota inferior y superior por major: el ecosistema de
  LangChain introduce cambios incompatibles entre versiones mayores.
- Los artículos procesados se deduplican por hash SHA-256 del PDF: volver a subir el mismo
  archivo carga el índice existente en lugar de reprocesarlo.

## Cita

Ver [`CITATION.cff`](CITATION.cff). La referencia al artículo se añadirá cuando se publique.

## Licencia

MIT — ver [`LICENSE`](LICENSE).

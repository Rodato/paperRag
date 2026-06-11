"""
Agente LangGraph para consultas sobre papers científicos.

El grafo cubre la fase de *recuperación* (analizar consulta → elegir
vectorstore → extraer filtros → buscar). La *generación* de la respuesta
se separa en un paso aparte para poder streamear los tokens en la UI.

Los nodos son closures que capturan los vectorstores, secciones y
referencias resueltas — sin variables globales.
"""

import re
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, TypedDict

from langgraph.graph import END, StateGraph

from .utils import parse_refs_field

# Cuánto contenido de cada chunk se inyecta al prompt de generación.
CONTEXT_CHARS_PER_RESULT = 1_200
MAX_CONTEXT_RESULTS = 5


# ---------------------------------------------------------------------------
# Estado del agente
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    query_type: str
    vectorstore_choice: str
    search_filters: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    final_answer: str
    reasoning: str
    confidence: float


class CompiledAgent(NamedTuple):
    """Agente compilado: grafo de recuperación + generación streameable."""

    retrieval: Any  # grafo LangGraph compilado (analyze → … → search)
    build_prompt: Callable[[AgentState], str]
    llm: Any


# ---------------------------------------------------------------------------
# Factory del grafo
# ---------------------------------------------------------------------------

def build_agent(
    chroma_store, faiss_store, sections: list, resolved_references: dict, query_llm
) -> CompiledAgent:
    """
    Compila el agente para un paper específico.

    Todos los nodos son closures que capturan los stores y metadatos
    del paper, evitando variables globales.
    """

    # ---- Nodo 1: Analizar consulta ----------------------------------------
    def analyze_query(state: AgentState) -> AgentState:
        query = state["query"].lower()
        has_ref = bool(re.search(r"\[\d+\]", query)) or any(
            w in query for w in ["referencia", "reference"]
        )
        if any(w in query for w in ["secciones", "sections", "sección", "section"]) and has_ref:
            query_type = "reference_sections"
        elif any(w in query for w in ["referencias", "references"]) and \
                any(w in query for w in ["sección", "section"]):
            query_type = "section_references"
        elif "contexto" in query and has_ref:
            query_type = "reference_context"
        elif any(w in query for w in ["trata", "about", "resumen", "summary", "resumí", "resumi"]):
            query_type = "section_summary"
        else:
            query_type = "general_search"
        state["query_type"] = query_type
        return state

    # ---- Nodo 2: Seleccionar vectorstore -----------------------------------
    def select_vectorstore(state: AgentState) -> AgentState:
        qt = state["query_type"]
        query = state["query"].lower()
        if qt == "reference_sections":
            choice, reasoning = "chroma", "Chroma: búsqueda específica de secciones que usan una referencia"
        elif qt == "section_references":
            choice, reasoning = "faiss", "FAISS: obtener todas las referencias de una sección completa"
        elif qt == "reference_context":
            choice, reasoning = "chroma", "Chroma: búsqueda detallada del contexto de uso de referencias"
        elif qt == "section_summary":
            choice, reasoning = "faiss", "FAISS: resumen de sección completa"
        else:
            if len(query.split()) <= 3:
                choice, reasoning = "chroma", "Chroma: búsqueda específica y detallada"
            else:
                choice, reasoning = "faiss", "FAISS: búsqueda semántica amplia"
        state["vectorstore_choice"] = choice
        state["reasoning"] = reasoning
        return state

    # ---- Nodo 3: Extraer filtros ------------------------------------------
    def extract_filters(state: AgentState) -> AgentState:
        query = state["query"]
        query_lower = query.lower()
        filters: Dict[str, Any] = {}

        # Detectar sección: preferir el match más largo (más específico)
        # para no quedarnos con "Results" cuando la query dice "Results and Discussion".
        matched = [s for s in sections if s.lower() in query_lower]
        if matched:
            filters["section_title"] = max(matched, key=len)

        # Detectar referencia(s)
        ref_matches = re.findall(r"\[(\d+)\]", query)
        if ref_matches:
            filters["reference_number"] = ref_matches[0]
            if len(ref_matches) > 1:
                filters["multiple_refs"] = ref_matches

        state["search_filters"] = filters
        return state

    # ---- Búsquedas especializadas (helpers internos) ----------------------
    def _search_chroma(query: str, filters: Dict) -> List[Dict]:
        if "reference_number" in filters:
            ref_num = filters["reference_number"]
            results = chroma_store.similarity_search(f"[{ref_num}]", k=20)
            filtered = [doc for doc in results if ref_num in parse_refs_field(doc.metadata)]
            return [{"content": d.page_content, "metadata": d.metadata} for d in filtered]
        elif "section_title" in filters:
            chroma_filter = {"section_title": {"$eq": filters["section_title"]}}
            results = chroma_store.similarity_search(query, filter=chroma_filter, k=8)
        else:
            results = chroma_store.similarity_search(query, k=8)
        return [{"content": d.page_content, "metadata": d.metadata} for d in results]

    def _search_faiss(query: str, filters: Dict) -> List[Dict]:
        results = faiss_store.similarity_search(query, k=15)
        filtered = []
        for doc in results:
            if "section_title" in filters:
                if doc.metadata.get("section_title") != filters["section_title"]:
                    continue
            if "reference_number" in filters:
                if filters["reference_number"] not in parse_refs_field(doc.metadata):
                    continue
            filtered.append({"content": doc.page_content, "metadata": doc.metadata})
            if len(filtered) >= 5:
                break
        return filtered

    # ---- Nodo 4: Ejecutar búsqueda ----------------------------------------
    def execute_search(state: AgentState) -> AgentState:
        query = state["query"]
        choice = state["vectorstore_choice"]
        filters = state["search_filters"]
        if choice == "chroma":
            search_results = _search_chroma(query, filters)
        else:
            search_results = _search_faiss(query, filters)
        state["search_results"] = search_results
        state["confidence"] = min(len(search_results) / 5.0, 1.0)
        return state

    # ---- Generación (separada del grafo para poder streamear) -------------
    def build_prompt(state: AgentState) -> str:
        query = state["query"]
        qt = state["query_type"]
        results = state["search_results"]
        reasoning = state["reasoning"]
        choice = state["vectorstore_choice"]
        confidence = state["confidence"]
        filters = state["search_filters"]

        context_parts = []
        for i, result in enumerate(results[:MAX_CONTEXT_RESULTS]):
            meta = result["metadata"]
            content = result["content"]
            if len(content) > CONTEXT_CHARS_PER_RESULT:
                content = content[:CONTEXT_CHARS_PER_RESULT] + "…"
            context_parts.append(
                f"RESULTADO {i+1}:\n"
                f"Sección: {meta.get('section_title', 'N/A')}\n"
                f"Referencias mencionadas: {meta.get('references_mentioned', 'Ninguna')}\n"
                f"Contenido: {content}\n"
            )
        context = "\n".join(context_parts) if context_parts else "(sin resultados relevantes)"

        relevant_refs = ""
        if "reference_number" in filters:
            ref_num = filters["reference_number"]
            if ref_num in resolved_references:
                relevant_refs = f"\nREFERENCIA [{ref_num}] COMPLETA:\n{resolved_references[ref_num]}\n"
            else:
                relevant_refs = (
                    f"\nNOTA: La referencia [{ref_num}] no se pudo extraer de la "
                    "bibliografía del paper (puede no existir o no haberse parseado bien). "
                    "Aclará esto en tu respuesta.\n"
                )

        # Sin contexto recuperado: instruir explícitamente para evitar alucinaciones.
        if not results:
            return (
                "Eres un asistente especializado en análisis de papers científicos.\n\n"
                f"CONSULTA: {query}\n\n"
                "No se encontró contenido relevante en el paper para esta consulta. "
                "Respondé honestamente que no encontraste información en el documento "
                "para responder esto, y sugerí reformular la pregunta (por ejemplo, "
                "nombrando una sección o referencia específica). No inventes contenido."
            )

        task_instructions = {
            "reference_sections": "Lista específicamente en qué secciones se menciona la referencia solicitada y proporciona el contexto de cada mención.",
            "section_references": "Lista todas las referencias citadas en la sección especificada, incluyendo sus números y contenido cuando sea posible.",
            "reference_context": "Explica el contexto específico en el que se usa la referencia, incluyendo cómo se relaciona con el argumento o metodología del paper.",
            "section_summary": "Proporciona un resumen comprensivo de lo que trata la sección, incluyendo sus puntos principales.",
        }
        task_instruction = task_instructions.get(
            qt, "Responde la consulta de manera comprehensiva usando el contexto proporcionado."
        )

        return f"""Eres un asistente especializado en análisis de papers científicos.

CONSULTA ORIGINAL: {query}
TIPO DE CONSULTA: {qt}
ESTRATEGIA USADA: {reasoning} (vectorstore: {choice.upper()})
CONFIANZA: {confidence:.1%}

CONTEXTO ENCONTRADO:
{context}
{relevant_refs}
FILTROS APLICADOS: {filters}

INSTRUCCIONES:
{task_instruction}

REGLAS:
1. Sé específico y preciso
2. Si mencionas referencias, usa el formato [número]
3. Incluye números de sección cuando sea relevante
4. Responde ÚNICAMENTE con información presente en el contexto. Si el contexto no
   alcanza para responder, dilo explícitamente en vez de inventar.
5. Estructura tu respuesta de manera clara

RESPUESTA:"""

    # ---- Compilar grafo de recuperación -----------------------------------
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("select", select_vectorstore)
    workflow.add_node("extract", extract_filters)
    workflow.add_node("search", execute_search)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "select")
    workflow.add_edge("select", "extract")
    workflow.add_edge("extract", "search")
    workflow.add_edge("search", END)

    return CompiledAgent(
        retrieval=workflow.compile(),
        build_prompt=build_prompt,
        llm=query_llm,
    )


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def _initial_state(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "query_type": "",
        "vectorstore_choice": "",
        "search_filters": {},
        "search_results": [],
        "final_answer": "",
        "reasoning": "",
        "confidence": 0.0,
    }


def _result_meta(state: Dict[str, Any], answer: str) -> Dict[str, Any]:
    return {
        "pregunta": state["query"],
        "tipo_consulta": state["query_type"],
        "vectorstore_usado": state["vectorstore_choice"],
        "razonamiento": state["reasoning"],
        "filtros_aplicados": state["search_filters"],
        "num_resultados": len(state["search_results"]),
        "confianza": f"{state['confidence']:.1%}",
        "respuesta": answer,
    }


def run_query(agent: CompiledAgent, query: str) -> Dict[str, Any]:
    """Ejecuta recuperación + generación (no streaming) y retorna el resultado."""
    state = agent.retrieval.invoke(_initial_state(query))
    answer = agent.llm.invoke(agent.build_prompt(state)).content
    return _result_meta(state, answer)


def run_query_stream(agent: CompiledAgent, query: str):
    """Versión streaming.

    Retorna `(token_iterator, finalize)`. Consumí `token_iterator` (cada
    elemento es un fragmento de texto) y luego llamá `finalize(answer)` con
    el texto completo para obtener el dict de resultado con su metadata.
    """
    state = agent.retrieval.invoke(_initial_state(query))
    prompt = agent.build_prompt(state)

    def token_iterator() -> Iterator[str]:
        for chunk in agent.llm.stream(prompt):
            text = getattr(chunk, "content", "") or ""
            if text:
                yield text

    def finalize(answer: str) -> Dict[str, Any]:
        return _result_meta(state, answer)

    return token_iterator(), finalize

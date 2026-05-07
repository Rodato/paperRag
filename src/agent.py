"""
Agente LangGraph para consultas sobre papers científicos.

Los 5 nodos son funciones closure que capturan los vectorstores,
secciones y referencias resueltas — sin variables globales.
"""

import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .utils import parse_refs_field


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


# ---------------------------------------------------------------------------
# Factory del grafo
# ---------------------------------------------------------------------------

def build_agent(chroma_store, faiss_store, sections: list, resolved_references: dict, query_llm):
    """
    Compila el grafo LangGraph para un paper específico.

    Todos los nodos son closures que capturan los stores y metadatos
    del paper, evitando variables globales.
    """

    # ---- Nodo 1: Analizar consulta ----------------------------------------
    def analyze_query(state: AgentState) -> AgentState:
        query = state["query"].lower()
        if any(w in query for w in ["secciones", "sections", "sección", "section"]) and \
           any(w in query for w in ["referencia", "reference", "["]):
            query_type = "reference_sections"
        elif any(w in query for w in ["referencias", "references"]) and \
             any(w in query for w in ["sección", "section"]):
            query_type = "section_references"
        elif "contexto" in query and any(w in query for w in ["referencia", "reference", "["]):
            query_type = "reference_context"
        elif any(w in query for w in ["trata", "about", "resumen", "summary"]):
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

        # Detectar sección
        for section in sections:
            if section.lower() in query_lower:
                filters["section_title"] = section
                break

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

    # ---- Nodo 5: Generar respuesta ----------------------------------------
    def generate_answer(state: AgentState) -> AgentState:
        query = state["query"]
        qt = state["query_type"]
        results = state["search_results"]
        reasoning = state["reasoning"]
        choice = state["vectorstore_choice"]
        confidence = state["confidence"]
        filters = state["search_filters"]

        context_parts = []
        for i, result in enumerate(results[:4]):
            meta = result["metadata"]
            context_parts.append(
                f"RESULTADO {i+1}:\n"
                f"Sección: {meta.get('section_title', 'N/A')}\n"
                f"Referencias mencionadas: {meta.get('references_mentioned', 'Ninguna')}\n"
                f"Contenido: {result['content'][:400]}...\n"
            )
        context = "\n".join(context_parts)

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

        task_instructions = {
            "reference_sections": "Lista específicamente en qué secciones se menciona la referencia solicitada y proporciona el contexto de cada mención.",
            "section_references": "Lista todas las referencias citadas en la sección especificada, incluyendo sus números y contenido cuando sea posible.",
            "reference_context": "Explica el contexto específico en el que se usa la referencia, incluyendo cómo se relaciona con el argumento o metodología del paper.",
            "section_summary": "Proporciona un resumen comprensivo de lo que trata la sección, incluyendo sus puntos principales.",
        }
        task_instruction = task_instructions.get(qt, "Responde la consulta de manera comprehensiva usando el contexto proporcionado.")

        prompt = f"""Eres un asistente especializado en análisis de papers científicos.

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
4. Si la información es limitada, sé honesto al respecto
5. Estructura tu respuesta de manera clara

RESPUESTA:"""

        response = query_llm.invoke(prompt)
        state["final_answer"] = response.content
        return state

    # ---- Compilar grafo ---------------------------------------------------
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("select", select_vectorstore)
    workflow.add_node("extract", extract_filters)
    workflow.add_node("search", execute_search)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "select")
    workflow.add_edge("select", "extract")
    workflow.add_edge("extract", "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def run_query(app, query: str) -> Dict[str, Any]:
    """Ejecuta una consulta contra el grafo compilado y retorna el resultado completo."""
    result = app.invoke(
        {
            "query": query,
            "query_type": "",
            "vectorstore_choice": "",
            "search_filters": {},
            "search_results": [],
            "final_answer": "",
            "reasoning": "",
            "confidence": 0.0,
        }
    )
    return {
        "pregunta": query,
        "tipo_consulta": result["query_type"],
        "vectorstore_usado": result["vectorstore_choice"],
        "razonamiento": result["reasoning"],
        "filtros_aplicados": result["search_filters"],
        "num_resultados": len(result["search_results"]),
        "confianza": f"{result['confidence']:.1%}",
        "respuesta": result["final_answer"],
    }

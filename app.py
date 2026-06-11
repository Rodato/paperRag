"""paperRag — Streamlit App.

RAG con agentes para consultas sobre referencias en papers científicos.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src import agent as agent_module
from src import config
from src.history import append_to_history, clear_history, load_history
from src.processor import (
    extract_resolved_references,
    find_paper_by_hash,
    load_paper_meta,
    run_process_pdf,
)
from src.utils import compute_pdf_hash
from src.vectorstore import load_paper_stores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="paperRag",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&display=swap');

        :root {
            --pr-bg: #FFFFFF;
            --pr-surface: #FAFAF7;
            --pr-surface-2: #F3F2EC;
            --pr-border: #E8E6DE;
            --pr-border-strong: #D8D6CB;
            --pr-ink: #1B1B22;
            --pr-ink-soft: #44444F;
            --pr-muted: #8A8A93;
            --pr-accent: #4F46E5;
            --pr-accent-ink: #3730A3;
            --pr-accent-soft: #EEF0FF;
            --pr-radius: 14px;
            --pr-radius-sm: 10px;
            --pr-shadow-sm: 0 1px 2px rgba(20,20,40,0.04);
            --pr-shadow-md: 0 8px 28px rgba(28,26,40,0.08);
            --pr-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            --pr-serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
        }

        /* Tipografía base — NO usar [class*="st-"] para no pisar la fuente
           de íconos (Material Symbols) de Streamlit. */
        html, body, .stApp { font-family: var(--pr-sans); }
        .stApp { background: var(--pr-bg); color: var(--pr-ink); }
        /* Preservar la fuente de íconos pase lo que pase */
        [data-testid="stIconMaterial"],
        span[data-testid="stIconMaterial"],
        .material-icons, .material-symbols-rounded, .material-symbols-outlined {
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        }

        /* Header transparente, sin barra gris */
        [data-testid="stHeader"] { background: transparent; }

        /* Layout general */
        .main .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1000px; }
        h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--pr-ink); }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: var(--pr-surface);
            border-right: 1px solid var(--pr-border);
        }
        [data-testid="stSidebar"] .stButton > button {
            text-align: left;
            font-weight: 500;
        }
        .pr-wordmark {
            display: flex; align-items: baseline; gap: 2px;
            font-family: var(--pr-serif);
            font-size: 1.5rem; font-weight: 600;
            color: var(--pr-ink); letter-spacing: -0.02em;
            margin: 0.2rem 0 0.1rem;
        }
        .pr-wordmark .pr-wm-accent { color: var(--pr-accent); }
        .pr-wordmark-sub {
            font-size: 0.78rem; color: var(--pr-muted);
            margin-bottom: 0.4rem; letter-spacing: 0.01em;
        }

        /* ---------- Empty state / hero ---------- */
        .pr-hero { padding: 2.5rem 0 1.25rem; max-width: 680px; }
        .pr-eyebrow {
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--pr-accent);
            margin-bottom: 0.7rem;
        }
        .pr-hero-title {
            font-family: var(--pr-serif);
            font-size: 2.9rem; font-weight: 600; line-height: 1.05;
            letter-spacing: -0.025em; color: var(--pr-ink);
            margin-bottom: 0.7rem;
        }
        .pr-hero-sub {
            font-size: 1.08rem; color: var(--pr-ink-soft);
            max-width: 560px; line-height: 1.6;
        }

        /* ---------- Pills / badges ---------- */
        .pr-pill {
            display: inline-flex; align-items: center;
            font-size: 0.74rem; font-weight: 500;
            background: var(--pr-accent-soft); color: var(--pr-accent-ink);
            padding: 4px 11px; border-radius: 999px; margin-right: 6px;
            border: 1px solid rgba(79,70,229,0.12);
        }
        .pr-pill-muted {
            background: var(--pr-surface-2); color: var(--pr-muted);
            border-color: var(--pr-border);
        }

        /* ---------- Cards (st.container border=True) ---------- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--pr-radius) !important;
            border-color: var(--pr-border) !important;
            box-shadow: var(--pr-shadow-sm);
            transition: border-color 0.18s, box-shadow 0.18s, transform 0.18s;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: var(--pr-border-strong) !important;
            box-shadow: var(--pr-shadow-md);
            transform: translateY(-2px);
        }

        /* ---------- Paper activo (sidebar) ---------- */
        .pr-active-paper {
            background: var(--pr-bg);
            border: 1px solid var(--pr-border);
            border-left: 3px solid var(--pr-accent);
            border-radius: var(--pr-radius-sm);
            padding: 13px 15px; margin-bottom: 12px;
            box-shadow: var(--pr-shadow-sm);
        }
        .pr-active-paper-label {
            font-size: 0.66rem; text-transform: uppercase;
            letter-spacing: 0.08em; color: var(--pr-accent);
            font-weight: 600;
        }
        .pr-active-paper-title {
            font-family: var(--pr-serif);
            font-size: 1rem; font-weight: 600; color: var(--pr-ink);
            line-height: 1.32; margin-top: 4px;
        }

        /* ---------- Paper view header ---------- */
        .pr-doc-eyebrow {
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--pr-muted);
            margin-bottom: 0.4rem;
        }
        .pr-doc-title {
            font-family: var(--pr-serif);
            font-size: 2rem; font-weight: 600; line-height: 1.15;
            letter-spacing: -0.02em; color: var(--pr-ink);
            margin-bottom: 0.8rem;
        }

        /* ---------- Library card title ---------- */
        .pr-card-title {
            font-family: var(--pr-serif);
            font-size: 1.08rem; font-weight: 600; line-height: 1.3;
            color: var(--pr-ink); margin-bottom: 2px;
        }

        /* ---------- Section label ---------- */
        .pr-section-label {
            font-size: 0.74rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--pr-muted);
        }

        /* ---------- Inputs ---------- */
        [data-testid="stTextInput"] input {
            border-radius: var(--pr-radius-sm) !important;
            border-color: var(--pr-border-strong) !important;
            font-size: 0.98rem !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--pr-accent) !important;
            box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
        }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            border-radius: var(--pr-radius-sm) !important;
            font-weight: 500 !important;
            transition: transform 0.12s, box-shadow 0.12s, background 0.12s;
        }
        .stButton > button[kind="primary"],
        [data-testid="stBaseButton-primary"],
        [data-testid="stFormSubmitButton"] button {
            background: var(--pr-accent) !important;
            border-color: var(--pr-accent) !important;
            box-shadow: 0 1px 2px rgba(79,70,229,0.25);
        }
        .stButton > button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: var(--pr-accent-ink) !important;
            border-color: var(--pr-accent-ink) !important;
            box-shadow: 0 4px 14px rgba(79,70,229,0.28);
            transform: translateY(-1px);
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] details {
            border-radius: var(--pr-radius-sm) !important;
            border-color: var(--pr-border) !important;
        }
        [data-testid="stExpander"] summary { font-weight: 500; }

        /* ---------- Metric ---------- */
        [data-testid="stMetric"] {
            background: var(--pr-surface);
            border: 1px solid var(--pr-border);
            border-radius: var(--pr-radius-sm);
            padding: 12px 14px;
        }

        /* Form sin borde */
        [data-testid="stForm"] { border: none; padding: 0; }

        /* Divider más suave */
        hr { border-color: var(--pr-border) !important; }

        /* ---------- Footer ---------- */
        .pr-footer {
            margin-top: 3.5rem; padding-top: 1.1rem;
            border-top: 1px solid var(--pr-border);
            color: var(--pr-muted); font-size: 0.8rem;
            display: flex; align-items: center; gap: 6px;
        }
        .pr-footer .pr-dot { color: var(--pr-border-strong); }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("paper", None)
st.session_state.setdefault("query_history", [])
st.session_state.setdefault("selected_model", next(iter(config.QUERY_MODELS.keys())))
st.session_state.setdefault("query_text", "")

# Si el run anterior pidió limpiar el input post-submit, hacelo antes de instanciar el widget.
if st.session_state.pop("clear_query_after_submit", False):
    st.session_state["query_text"] = ""


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _cached_embeddings():
    return config.get_embeddings()


@st.cache_resource(show_spinner=False)
def _cached_paper_stores(paper_dir_str: str, collection_name: str):
    return load_paper_stores(Path(paper_dir_str), collection_name, _cached_embeddings())


@st.cache_resource(show_spinner=False)
def _cached_agent(
    sanitized_name: str,
    model_id: str,
    _chroma_store,
    _faiss_store,
    _sections: tuple,
    _resolved_references: tuple,
):
    """Cachea el grafo compilado por (paper, modelo). Args con `_` no entran al hash."""
    return agent_module.build_agent(
        chroma_store=_chroma_store,
        faiss_store=_faiss_store,
        sections=list(_sections),
        resolved_references=dict(_resolved_references),
        query_llm=config.get_query_llm(model_id),
    )


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _load_paper_into_session(paper_dir: Path) -> None:
    meta = load_paper_meta(paper_dir)
    chroma_store, faiss_store = _cached_paper_stores(str(paper_dir), meta["collection_name"])
    resolved_references = extract_resolved_references(faiss_store)
    st.session_state["paper"] = {
        "paper_title": meta["paper_title"],
        "paper_name": meta["paper_name"],
        "sanitized_name": meta["sanitized_name"],
        "sections": meta["sections"],
        "resolved_references": resolved_references,
        "chroma_store": chroma_store,
        "faiss_store": faiss_store,
        "sections_count": len(meta["sections"]),
        "refs_count": len(resolved_references),
        "paper_dir": str(paper_dir),
        "processed_at": meta.get("processed_at", ""),
    }
    st.session_state["query_history"] = load_history(paper_dir)


def _delete_paper(paper_dir: Path) -> None:
    sanitized = paper_dir.name
    shutil.rmtree(paper_dir, ignore_errors=True)
    paper = st.session_state.get("paper")
    if paper is not None and paper["sanitized_name"] == sanitized:
        st.session_state["paper"] = None
        st.session_state["query_history"] = []
    _cached_paper_stores.clear()


def _get_or_build_agent():
    paper = st.session_state["paper"]
    model_id = config.QUERY_MODELS[st.session_state["selected_model"]]
    return _cached_agent(
        paper["sanitized_name"],
        model_id,
        _chroma_store=paper["chroma_store"],
        _faiss_store=paper["faiss_store"],
        _sections=tuple(paper["sections"]),
        _resolved_references=tuple(paper["resolved_references"].items()),
    )


def _list_papers() -> list[dict]:
    if not config.DATA_DIR.exists():
        return []
    out = []
    for d in sorted(config.DATA_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        if not ((d / "meta.json").exists() or (d / "processed_data.pkl").exists()):
            continue
        try:
            meta = load_paper_meta(d)
            out.append({"paper_dir": d, "meta": meta, "corrupted": False})
        except Exception:
            logger.exception("Paper inválido en %s", d)
            out.append({"paper_dir": d, "meta": None, "corrupted": True})
    out.sort(key=lambda p: (p["meta"] or {}).get("processed_at", ""), reverse=True)
    return out


def _format_processed_at(iso: str) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y")
    except Exception:
        return iso


def _build_query_suggestions(sections, resolved_references) -> list[str]:
    suggestions: list[str] = []
    sec_list = list(sections) if sections else []
    if sec_list:
        candidates = [s for s in sec_list if any(k in s.lower() for k in ("método", "method", "metodolog"))]
        section_a = candidates[0] if candidates else sec_list[0]
        suggestions.append(f"Resumí la sección {section_a}")
        if len(sec_list) > 1:
            section_b = next(
                (s for s in sec_list[::-1] if "introduc" not in s.lower()),
                sec_list[-1],
            )
            if section_b != section_a:
                suggestions.append(f"¿Qué referencias se citan en {section_b}?")
    if resolved_references:
        first_ref = next(iter(resolved_references.keys()))
        suggestions.append(f"¿En qué contexto se usa la referencia [{first_ref}]?")
        if len(resolved_references) > 1:
            suggestions.append(f"¿En qué secciones se menciona la referencia [{first_ref}]?")
    return suggestions[:4]


_REF_PATTERN = re.compile(r"\[(\d+)\]")


def _extract_cited_refs(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _REF_PATTERN.finditer(text or ""):
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _build_export_md(entry: dict, paper: dict) -> str:
    lines = [
        f"# Consulta sobre _{paper['paper_title']}_",
        "",
        f"**Pregunta:** {entry.get('pregunta', '')}",
        "",
        f"_Modelo: {entry.get('modelo', st.session_state['selected_model'])} · "
        f"Tipo: {entry.get('tipo_consulta', '')} · "
        f"Vectorstore: {entry.get('vectorstore_usado', '')} · "
        f"Confianza: {entry.get('confianza', '')}_",
        "",
        "## Respuesta",
        "",
        entry.get("respuesta", ""),
        "",
    ]
    cited = _extract_cited_refs(entry.get("respuesta", ""))
    if cited:
        lines.append("## Referencias citadas")
        lines.append("")
        for n in cited:
            text = paper["resolved_references"].get(n)
            if text:
                lines.append(f"- **[{n}]** {text}")
            else:
                lines.append(f"- **[{n}]** _(no extraída de la bibliografía)_")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def _process_uploaded(uploaded_file) -> None:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    pdf_bytes = uploaded_file.read()
    pdf_hash = compute_pdf_hash(pdf_bytes)
    existing = find_paper_by_hash(config.DATA_DIR, pdf_hash)

    if existing is not None:
        status_text.info("Este PDF ya estaba procesado. Cargándolo…")
        progress_bar.progress(1.0)
        _load_paper_into_session(existing)
        st.rerun()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        def on_progress(step: str, pct: float) -> None:
            progress_bar.progress(pct)
            status_text.text(step)

        result = run_process_pdf(
            pdf_path=tmp_path,
            output_dir=config.DATA_DIR,
            processing_llm=config.get_processing_llm(),
            embeddings=_cached_embeddings(),
            pdf_hash=pdf_hash,
            processing_model=config.PROCESSING_MODEL,
            embedding_model=config.EMBEDDING_MODEL,
            on_progress=on_progress,
            original_filename=uploaded_file.name,
        )
        if result:
            paper_dir = config.DATA_DIR / result["sanitized_name"]
            _load_paper_into_session(paper_dir)
            st.rerun()
        else:
            status_text.error("No pudimos procesar este PDF.")
    except Exception as e:
        logger.exception("Falló el procesamiento del PDF %s", uploaded_file.name)
        status_text.error(
            "No pudimos procesar este PDF. Verificá que tenga texto seleccionable "
            "y estructura de secciones (Introducción, Métodos, etc.)."
        )
        with st.expander("Detalle técnico"):
            st.code(str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Confirm-delete dialog
# ---------------------------------------------------------------------------

@st.dialog("Borrar paper")
def _confirm_delete_dialog(paper_dir: Path, title: str) -> None:
    st.warning(f"Esto borra permanentemente «{title}» y su historial.")
    st.caption(f"Carpeta: `{paper_dir}`")
    col1, col2 = st.columns(2)
    if col1.button("Cancelar", use_container_width=True):
        st.rerun()
    if col2.button("Borrar definitivamente", type="primary", use_container_width=True):
        _delete_paper(paper_dir)
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div class='pr-wordmark'>paper<span class='pr-wm-accent'>Rag</span></div>"
        "<div class='pr-wordmark-sub'>Análisis de referencias con agentes</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not config.OPENROUTER_KEY:
        st.error("Falta `OPENROUTER_API_KEY` en el archivo .env")
        st.stop()

    paper = st.session_state.get("paper")

    if paper:
        st.markdown(
            f"<div class='pr-active-paper'>"
            f"<div class='pr-active-paper-label'>Paper activo</div>"
            f"<div class='pr-active-paper-title'>{paper['paper_title']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        col_a.caption(f"📂 {paper['sections_count']} secciones")
        col_b.caption(f"🔖 {paper['refs_count']} referencias")
        if st.button("Cerrar paper", use_container_width=True):
            st.session_state["paper"] = None
            st.session_state["query_history"] = []
            st.rerun()
        st.divider()

    st.markdown("**Subir paper**")
    uploaded_file = st.file_uploader(
        "PDF",
        type="pdf",
        label_visibility="collapsed",
        key="pdf_uploader",
    )
    if uploaded_file is not None:
        if st.button("Procesar paper", type="primary", use_container_width=True):
            _process_uploaded(uploaded_file)

    st.divider()

    st.markdown("**Modelo de consulta**")
    model_names = list(config.QUERY_MODELS.keys())
    selected_name = st.selectbox(
        "Modelo",
        options=model_names,
        index=model_names.index(st.session_state["selected_model"]),
        label_visibility="collapsed",
    )
    if selected_name != st.session_state["selected_model"]:
        st.session_state["selected_model"] = selected_name

    st.divider()
    st.caption("Modelos vía OpenRouter")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

paper = st.session_state.get("paper")
papers_in_disk = _list_papers()

if paper is None:
    # ---------- Empty state / biblioteca ----------
    st.markdown(
        """
        <div class='pr-hero'>
            <div class='pr-eyebrow'>RAG con agentes · LangGraph</div>
            <div class='pr-hero-title'>Conversá con la bibliografía de tus papers</div>
            <div class='pr-hero-sub'>
                Subí un PDF y consultá sobre sus referencias, secciones y argumentos.
                paperRag detecta automáticamente los papers que ya procesaste.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not papers_in_disk:
        st.info("Empezá subiendo tu primer PDF en el panel lateral. ↖")
        st.markdown("##### Tipos de consultas que vas a poder hacer")
        st.markdown(
            """
            - *¿En qué secciones se menciona la referencia [3]?*
            - *¿Qué referencias se citan en la sección de Metodología?*
            - *¿En qué contexto se usa la referencia [5]?*
            - *Resumí la sección de Discusión.*
            - Cualquier consulta libre sobre el contenido del paper.
            """
        )
    else:
        st.markdown(
            f"<div class='pr-section-label' style='margin-bottom:0.9rem'>"
            f"Tu biblioteca · {len(papers_in_disk)} "
            f"{'paper' if len(papers_in_disk) == 1 else 'papers'}</div>",
            unsafe_allow_html=True,
        )

        cols_per_row = 2
        for row_start in range(0, len(papers_in_disk), cols_per_row):
            row = papers_in_disk[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, item in zip(cols, row):
                with col:
                    with st.container(border=True):
                        if item["corrupted"]:
                            st.markdown(f"**⚠️ {item['paper_dir'].name}**")
                            st.caption("Paper corrupto — no se pudo leer la metadata")
                            if st.button(
                                "Borrar",
                                key=f"del_corrupt_{item['paper_dir'].name}",
                                use_container_width=True,
                            ):
                                _confirm_delete_dialog(item["paper_dir"], item["paper_dir"].name)
                            continue
                        meta = item["meta"]
                        title = meta.get("paper_title", item["paper_dir"].name)
                        short_title = title if len(title) <= 80 else title[:77] + "…"
                        st.markdown(
                            f"<div class='pr-card-title'>{short_title}</div>",
                            unsafe_allow_html=True,
                        )
                        date_str = _format_processed_at(meta.get("processed_at", ""))
                        meta_line = f"📂 {len(meta.get('sections', []))} secciones · 🔖 {meta.get('faiss_chunks_count', '?')} secciones indexadas"
                        if date_str:
                            meta_line += f" · 🕒 {date_str}"
                        st.caption(meta_line)

                        c_open, c_del = st.columns([3, 1])
                        if c_open.button("Abrir", key=f"open_{item['paper_dir'].name}", type="primary", use_container_width=True):
                            with st.spinner("Cargando paper…"):
                                _load_paper_into_session(item["paper_dir"])
                            st.rerun()
                        if c_del.button("🗑", key=f"del_{item['paper_dir'].name}", help="Borrar paper", use_container_width=True):
                            _confirm_delete_dialog(item["paper_dir"], title)

else:
    # ---------- Paper view ----------
    st.markdown(
        f"<div class='pr-doc-eyebrow'>Paper activo</div>"
        f"<div class='pr-doc-title'>{paper['paper_title']}</div>",
        unsafe_allow_html=True,
    )
    date_str = _format_processed_at(paper.get("processed_at", ""))
    pills = [
        f"<span class='pr-pill'>{paper['sections_count']} secciones</span>",
        f"<span class='pr-pill'>{paper['refs_count']} referencias</span>",
    ]
    if date_str:
        pills.append(f"<span class='pr-pill pr-pill-muted'>Procesado: {date_str}</span>")
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    # Bibliografía completa (collapsible)
    if paper["resolved_references"]:
        with st.expander(f"📚 Bibliografía completa ({paper['refs_count']} referencias)"):
            sorted_refs = sorted(
                paper["resolved_references"].items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else 9_999,
            )
            for ref_num, text in sorted_refs:
                st.markdown(f"**[{ref_num}]** {text}")

    st.divider()

    # ---------- Sugerencias ----------
    suggestions = _build_query_suggestions(paper["sections"], paper["resolved_references"])
    if suggestions:
        st.markdown(
            "<div class='pr-section-label' style='margin-bottom:0.5rem'>Sugerencias</div>",
            unsafe_allow_html=True,
        )
        sug_cols = st.columns(len(suggestions))
        for i, (col, suggestion) in enumerate(zip(sug_cols, suggestions)):
            if col.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state["query_text"] = suggestion
                st.rerun()

    # ---------- Input ----------
    with st.form("query_form", clear_on_submit=False):
        st.text_input(
            "Pregunta",
            key="query_text",
            placeholder="Ej: ¿En qué secciones se menciona la referencia [2]?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Consultar", type="primary", use_container_width=True)

    query = st.session_state["query_text"]
    if submitted and query.strip():
        try:
            app = _get_or_build_agent()
            with st.spinner("Buscando en el paper…"):
                token_iter, finalize = agent_module.run_query_stream(app, query.strip())
            st.markdown("### Respuesta")
            answer = st.write_stream(token_iter)
            result = finalize(answer)
            paper_dir = Path(paper["paper_dir"])
            enriched = append_to_history(
                paper_dir,
                result,
                model_name=st.session_state["selected_model"],
            )
            st.session_state["query_history"].insert(0, enriched)
            st.session_state["clear_query_after_submit"] = True
            st.rerun()
        except Exception as e:
            logger.exception("Falló run_query con query=%r", query)
            st.error("No pudimos ejecutar la consulta. Probá de nuevo en un momento.")
            with st.expander("Detalle técnico"):
                st.code(str(e))

    # ---------- Resultado más reciente ----------
    history_list = st.session_state["query_history"]
    if history_list:
        latest = history_list[0]

        st.markdown("### Respuesta")
        st.markdown(latest["respuesta"])

        # Citaciones con popover
        cited = _extract_cited_refs(latest["respuesta"])
        if cited:
            st.markdown(
                "<div class='pr-section-label' style='margin:0.6rem 0 0.4rem'>"
                "Citas en esta respuesta</div>",
                unsafe_allow_html=True,
            )
            cite_cols = st.columns(min(len(cited), 6))
            for i, ref_num in enumerate(cited):
                with cite_cols[i % len(cite_cols)]:
                    with st.popover(f"[{ref_num}]", use_container_width=True):
                        resolved = paper["resolved_references"].get(ref_num)
                        if resolved:
                            st.markdown(f"**Referencia [{ref_num}]**")
                            st.markdown(resolved)
                        else:
                            st.caption(
                                f"La referencia [{ref_num}] no fue extraída de la bibliografía del paper."
                            )

        # Export
        with st.expander("Copiar / descargar respuesta"):
            export_md = _build_export_md(latest, paper)
            st.code(export_md, language="markdown")
            st.download_button(
                "Descargar como .md",
                data=export_md.encode("utf-8"),
                file_name=f"{paper['sanitized_name']}_respuesta.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Detalles
        with st.expander("Detalles de la consulta"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Tipo", latest["tipo_consulta"])
            col_b.metric("Vectorstore", latest["vectorstore_usado"].upper())
            col_c.metric("Confianza", latest["confianza"])
            st.markdown(f"**Razonamiento:** {latest['razonamiento']}")
            st.markdown(f"**Filtros aplicados:** `{latest['filtros_aplicados']}`")
            st.markdown(f"**Resultados recuperados:** {latest['num_resultados']}")
            if latest.get("modelo"):
                st.markdown(f"**Modelo usado:** {latest['modelo']}")
            if latest.get("timestamp"):
                st.markdown(f"**Cuando:** {latest['timestamp']}")

        # Historial
        if len(history_list) > 1:
            with st.expander(f"Historial ({len(history_list) - 1} consultas anteriores)"):
                col_left, col_right = st.columns([4, 1])
                col_left.caption("El historial se guarda automáticamente con cada paper.")
                if col_right.button("Limpiar", use_container_width=True):
                    clear_history(Path(paper["paper_dir"]))
                    st.session_state["query_history"] = []
                    st.rerun()
                for past in history_list[1:]:
                    st.markdown(f"**Q:** {past['pregunta']}")
                    st.markdown(past["respuesta"])
                    extras = []
                    if past.get("tipo_consulta"):
                        extras.append(f"Tipo: {past['tipo_consulta']}")
                    if past.get("vectorstore_usado"):
                        extras.append(f"VS: {past['vectorstore_usado']}")
                    if past.get("confianza"):
                        extras.append(f"Confianza: {past['confianza']}")
                    if past.get("modelo"):
                        extras.append(f"Modelo: {past['modelo']}")
                    if past.get("timestamp"):
                        extras.append(f"{past['timestamp']}")
                    if extras:
                        st.caption(" · ".join(extras))
                    st.divider()

st.markdown(
    "<div class='pr-footer'>"
    "<span style='font-family:var(--pr-serif);font-weight:600'>paperRag</span>"
    "<span class='pr-dot'>·</span> RAG con agentes LangGraph "
    "<span class='pr-dot'>·</span> Modelos vía OpenRouter"
    "</div>",
    unsafe_allow_html=True,
)

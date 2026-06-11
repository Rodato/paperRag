"""
Pipeline de procesamiento de PDFs.

Flujo:
  1. PDF → Markdown  (docling, sin LLM)
  2. Extraer título, secciones y referencias bibliográficas  (3 LLM calls en paralelo)
  3. Chunking dual (párrafos → Chroma, secciones → FAISS)
  4. Guardar vectorstores + meta.json
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from docling.document_converter import DocumentConverter

from .utils import (
    extract_references,
    normalize_metadata,
    parse_gemini_sections,
    parse_references_to_dict,
    sanitize_collection_name,
)
from .vectorstore import save_paper_stores

logger = logging.getLogger(__name__)

TITLE_SAMPLE_CHARS = 2_000
SECTIONS_SAMPLE_CHARS = 40_000
BIBLIOGRAPHY_SAMPLE_CHARS = 30_000
MIN_PARAGRAPH_LENGTH = 50


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_call(llm, prompt: str) -> str:
    """Invoca el LLM y retorna el texto de la respuesta."""
    content = llm.invoke(prompt).content
    if content is None:
        return ""
    if isinstance(content, list):
        # Gemini puede devolver lista de bloques de contenido
        parts = [b["text"] if isinstance(b, dict) else str(b) for b in content]
        return "".join(parts)
    return str(content)


def _extract_title(md_text: str, llm) -> str:
    first_part = md_text[:TITLE_SAMPLE_CHARS]
    prompt = f"""Analiza el siguiente texto del inicio de un artículo científico y extrae únicamente el TÍTULO principal del paper.

Texto:
{first_part}

Instrucciones:
- Identifica y extrae solo el título principal del artículo
- NO incluyas nombres de autores, afiliaciones, resumen, o cualquier otro contenido
- Devuelve únicamente el título, sin comillas ni prefijos
- Si hay múltiples líneas que parecen título, combínalas en una sola línea
- Si no puedes identificar un título claro, devuelve "Título no identificado"

TÍTULO:"""
    title = _llm_call(llm, prompt).strip()
    title = re.sub(r"^(TÍTULO|Title|TITLE):\s*", "", title, flags=re.IGNORECASE)
    title = title.strip('"').strip("'").strip()
    return title if len(title) >= 5 else "Título no identificado"


def _identify_sections(md_text: str, llm) -> List[str]:
    text_sample = md_text[:SECTIONS_SAMPLE_CHARS]
    prompt = f"""Analiza el siguiente texto de un artículo científico:

{text_sample}

Identifica y lista todos los encabezados de sección principales (como Introduction, Methods, Results, Discussion, Conclusion, etc.) en el orden que aparecen.

Formato de respuesta:
## SECCIONES IDENTIFICADAS:
- [Lista de secciones en orden]

Solo los nombres de las secciones, sin explicaciones adicionales."""
    response = _llm_call(llm, prompt)
    return parse_gemini_sections(response)


def _extract_bibliography(md_text: str, llm) -> Dict[str, str]:
    text_sample = md_text[-BIBLIOGRAPHY_SAMPLE_CHARS:]
    prompt = f"""Analiza el siguiente texto de un artículo científico:

{text_sample}

Identifica y extrae todas las referencias bibliográficas del artículo.

Formato de respuesta:
## REFERENCIAS BIBLIOGRÁFICAS:
- [Número] Referencia completa"""
    response = _llm_call(llm, prompt)
    return parse_references_to_dict(response)


def _llm_analysis_parallel(md_text: str, llm):
    """Lanza título, secciones y bibliografía en paralelo (3 LLM calls independientes)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_title = pool.submit(_extract_title, md_text, llm)
        f_sections = pool.submit(_identify_sections, md_text, llm)
        f_refs = pool.submit(_extract_bibliography, md_text, llm)
        return f_title.result(), f_sections.result(), f_refs.result()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_by_sections(
    md_text: str, section_titles: List[str], resolved_references: Dict[str, str]
) -> List[Dict]:
    """Divide el texto por secciones y asocia las referencias resueltas."""
    if not section_titles:
        return []
    sections_data = []
    text_with_end = md_text + "\n## END_OF_DOCUMENT##"
    escaped = [re.escape(t) for t in section_titles]
    # Match section headers with optional markdown # prefix (e.g. "## Introduction")
    section_pattern = "|".join([f"^(?:#{{1,6}}\\s*)?{t}\\s*$" for t in escaped])
    end_pattern = r"^## END_OF_DOCUMENT##"
    combined = f"(?m)(?:{section_pattern})|{end_pattern}"

    all_matches = list(re.finditer(combined, text_with_end))
    if not all_matches or all_matches[0].group(0).strip() == "## END_OF_DOCUMENT##":
        return []

    for i in range(len(all_matches) - 1):
        start_m = all_matches[i]
        end_m = all_matches[i + 1]
        title = start_m.group(0).strip()
        # Strip markdown header prefix for comparison and storage
        plain_title = re.sub(r"^#+\s*", "", title).strip()
        original_title = next(
            (t for t in section_titles if t.strip().lower() == plain_title.lower()), plain_title
        )
        section_text = text_with_end[start_m.end() : end_m.start()].strip()
        if not section_text:
            # Header padre sin contenido propio (sus subsecciones llevan el texto):
            # no lo indexamos para no contaminar el RAG con placeholders.
            continue
        cited_refs = extract_references(section_text)
        section_resolved = {
            ref_id: resolved_references[ref_id]
            for ref_id in cited_refs
            if ref_id in resolved_references
        }
        sections_data.append(
            {
                "title": original_title,
                "text": section_text,
                "refs": cited_refs,
                "resolved_refs": section_resolved,
            }
        )
    return sections_data


def _chunks_for_chroma(
    sections_data: List[Dict], paper_name: str, paper_title: str
) -> List[Dict]:
    chunks = []
    for section in sections_data:
        for i, paragraph in enumerate(section["text"].split("\n\n")):
            if len(paragraph.strip()) > MIN_PARAGRAPH_LENGTH:
                chunk_refs = extract_references(paragraph)
                chunk_resolved = {
                    ref_id: section["resolved_refs"][ref_id]
                    for ref_id in chunk_refs
                    if ref_id in section["resolved_refs"]
                }
                chunks.append(
                    {
                        "text": paragraph.strip(),
                        "metadata": {
                            "paper_name": paper_name,
                            "paper_title": paper_title,
                            "section_title": section["title"],
                            "chunk_id": f"{paper_name}_{section['title']}_{i}",
                            "references_mentioned": chunk_refs,
                            "resolved_references": chunk_resolved,
                            "chunk_type": "paragraph",
                        },
                    }
                )
    return chunks


def _chunks_for_faiss(
    sections_data: List[Dict], paper_name: str, paper_title: str
) -> List[Dict]:
    return [
        {
            "text": section["text"],
            "metadata": {
                "paper_name": paper_name,
                "paper_title": paper_title,
                "section_title": section["title"],
                "chunk_id": f"{paper_name}_{section['title']}",
                "references_mentioned": section["refs"],
                "resolved_references": section["resolved_refs"],
                "chunk_type": "full_section",
            },
        }
        for section in sections_data
    ]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    processing_llm,
    embeddings,
    pdf_hash: str,
    processing_model: str,
    embedding_model: str,
    original_filename: Optional[str] = None,
) -> Generator[tuple, None, Dict[str, Any]]:
    """
    Generador que procesa un PDF y emite (step_name, progress_fraction).
    Al terminar retorna el dict de meta via StopIteration.value.

    `original_filename` es el nombre real del PDF subido por el usuario; se
    usa para nombrar la carpeta y los chunks. Si no se pasa, cae al stem del
    path (que puede ser un archivo temporal `tmpXXXX`).
    """
    source_name = original_filename or pdf_path.name
    paper_name = Path(source_name).stem
    sanitized_name = sanitize_collection_name(paper_name)
    collection_name = f"paper_{sanitized_name}_paragraphs"
    paper_dir = output_dir / sanitized_name

    # Paso 1: PDF → Markdown
    yield ("Convirtiendo PDF a Markdown… (~3 s)", 0.05)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()
    total_chars = len(md_text)

    # Paso 2: análisis con LLM (título, secciones y bibliografía en paralelo)
    yield ("Analizando título, secciones y bibliografía con LLM… (~10 s)", 0.25)
    paper_title, sections, resolved_references = _llm_analysis_parallel(md_text, processing_llm)

    # Paso 3: Chunking
    yield ("Calculando embeddings de párrafos y secciones… (~5 s)", 0.75)
    if not sections:
        raise ValueError(
            "El LLM no identificó secciones en el documento. "
            "Verifica que el PDF tiene estructura de secciones clara."
        )
    sections_data = _split_by_sections(md_text, sections, resolved_references)

    if not sections_data:
        raise ValueError(
            "No se encontraron secciones en el documento. "
            "Verifica que el PDF contiene texto legible y estructura de secciones reconocible."
        )

    chroma_chunks = _chunks_for_chroma(sections_data, paper_name, paper_title)
    faiss_chunks = _chunks_for_faiss(sections_data, paper_name, paper_title)

    if not faiss_chunks:
        raise ValueError("No se generaron secciones para indexar.")
    if not chroma_chunks:
        raise ValueError("No se generaron párrafos para indexar (todos los párrafos son muy cortos).")

    chroma_texts = [c["text"] for c in chroma_chunks]
    chroma_metas = [normalize_metadata(c["metadata"]) for c in chroma_chunks]
    faiss_texts = [c["text"] for c in faiss_chunks]
    faiss_metas = [normalize_metadata(c["metadata"]) for c in faiss_chunks]

    # Paso 6: Guardar vectorstores + meta.json
    yield ("Guardando vectorstores en disco… (~1 s)", 0.90)
    save_paper_stores(
        paper_dir,
        chroma_texts,
        chroma_metas,
        faiss_texts,
        faiss_metas,
        collection_name,
        embeddings,
    )

    meta = {
        "paper_name": paper_name,
        "paper_title": paper_title,
        "sanitized_name": sanitized_name,
        "collection_name": collection_name,
        "sections": sections,
        "pdf_hash": pdf_hash,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "processing_model": processing_model,
        "embedding_model": embedding_model,
        "total_chars": total_chars,
        "filename": source_name,
        "chroma_chunks_count": len(chroma_chunks),
        "faiss_chunks_count": len(faiss_chunks),
    }
    _write_meta(paper_dir, meta)

    yield ("¡Listo!", 1.0)
    return meta


def run_process_pdf(
    pdf_path: Path,
    output_dir: Path,
    processing_llm,
    embeddings,
    pdf_hash: str,
    processing_model: str,
    embedding_model: str,
    on_progress=None,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrapper síncrono sobre el generador process_pdf."""
    result: Dict[str, Any] = {}
    gen = process_pdf(
        pdf_path, output_dir, processing_llm, embeddings,
        pdf_hash, processing_model, embedding_model,
        original_filename=original_filename,
    )
    try:
        while True:
            step, pct = next(gen)
            if on_progress:
                on_progress(step, pct)
    except StopIteration as e:
        result = e.value if e.value else {}
    return result


# ---------------------------------------------------------------------------
# Persistencia: meta.json (con migración silenciosa desde processed_data.pkl)
# ---------------------------------------------------------------------------

META_FILENAME = "meta.json"
LEGACY_PKL_FILENAME = "processed_data.pkl"


def _write_meta(paper_dir: Path, meta: Dict[str, Any]) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _migrate_pkl_to_json(paper_dir: Path) -> Optional[Dict[str, Any]]:
    """Si existe processed_data.pkl y no hay meta.json, lo lee y escribe meta.json.

    El pkl se renombra a .bak. Retorna el meta migrado, o None si no había pkl.
    """
    pkl_path = paper_dir / LEGACY_PKL_FILENAME
    if not pkl_path.exists():
        return None
    import pickle  # solo necesario para leer el legacy
    with open(pkl_path, "rb") as f:
        legacy = pickle.load(f)
    legacy_metadata = legacy.get("metadata", {}) or {}
    meta = {
        "paper_name": legacy.get("paper_name", paper_dir.name),
        "paper_title": legacy.get("paper_title", paper_dir.name),
        "sanitized_name": legacy.get("sanitized_name", paper_dir.name),
        "collection_name": legacy.get(
            "collection_name", f"paper_{paper_dir.name}_paragraphs"
        ),
        "sections": legacy.get("sections", []),
        "pdf_hash": legacy.get("pdf_hash", ""),
        "processed_at": legacy.get("processed_at", ""),
        "processing_model": legacy.get("processing_model", ""),
        "embedding_model": legacy.get("embedding_model", ""),
        "total_chars": legacy_metadata.get("total_chars", 0),
        "filename": Path(legacy_metadata.get("filename", "")).name,
        "chroma_chunks_count": legacy.get("chroma_chunks_count", 0),
        "faiss_chunks_count": legacy.get("faiss_chunks_count", 0),
    }
    _write_meta(paper_dir, meta)
    pkl_path.rename(pkl_path.with_suffix(pkl_path.suffix + ".bak"))
    logger.info("Migrado pkl → meta.json en %s", paper_dir)
    return meta


def load_paper_meta(paper_dir: Path) -> Dict[str, Any]:
    """Carga meta.json. Si solo existe el pkl legacy, lo migra primero."""
    meta_path = paper_dir / META_FILENAME
    if not meta_path.exists():
        migrated = _migrate_pkl_to_json(paper_dir)
        if migrated is not None:
            return migrated
        raise FileNotFoundError(f"No se encontró meta.json ni pkl en {paper_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_resolved_references(faiss_store) -> Dict[str, str]:
    """Reconstruye el dict {ref_id: cita} desde la metadata del FAISS store.

    Cada doc FAISS tiene `resolved_references` serializado como JSON string.
    """
    refs: Dict[str, str] = {}
    docstore = getattr(faiss_store, "docstore", None)
    raw_docs = getattr(docstore, "_dict", {}) if docstore else {}
    for doc in raw_docs.values():
        raw = doc.metadata.get("resolved_references", "")
        if not raw:
            continue
        if isinstance(raw, dict):
            for ref_id, citation in raw.items():
                refs.setdefault(str(ref_id), str(citation))
        else:
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                for ref_id, citation in parsed.items():
                    refs.setdefault(str(ref_id), str(citation))
    return refs


def find_paper_by_hash(data_dir: Path, pdf_hash: str) -> Optional[Path]:
    """Busca en data_dir un paper ya procesado con el mismo hash. None si no hay."""
    if not data_dir.exists():
        return None
    for paper_dir in data_dir.iterdir():
        if not paper_dir.is_dir():
            continue
        meta_path = paper_dir / META_FILENAME
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("pdf_hash") == pdf_hash:
            return paper_dir
    return None

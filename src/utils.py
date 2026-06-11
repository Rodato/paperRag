import hashlib
import json
import re
from typing import Any, Dict, List


def sanitize_collection_name(name: str) -> str:
    """Sanitiza nombres para compatibilidad con Chroma.

    Determinista: el fallback para nombres muy cortos usa un hash estable
    (sha1) en vez de `hash()`, que cambia entre procesos por PYTHONHASHSEED.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_.-")
    if len(sanitized) < 3:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        sanitized = f"paper_{digest}"
    if len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip("_.-")
    return sanitized


def extract_references(text: str) -> List[str]:
    """Extrae números de referencias citadas del texto (e.g. [3], [1,2], [4–6])."""
    raw_refs = re.findall(r"\[([^\[\]]+?)\]", text)
    final_refs: set = set()
    for group in raw_refs:
        parts = [p.strip() for p in group.split(",")]
        for part in parts:
            if "–" in part or "-" in part:
                sep = "–" if "–" in part else "-"
                bounds = part.split(sep)
                if len(bounds) != 2:
                    continue
                try:
                    start, end = int(bounds[0]), int(bounds[1])
                except ValueError:
                    continue
                if 0 <= start <= end and end - start <= 500:
                    final_refs.update(str(i) for i in range(start, end + 1))
            elif part.isdigit():
                final_refs.add(part)
    return sorted(final_refs, key=int)


def parse_gemini_sections(response_text: str) -> List[str]:
    """Extrae lista de secciones de la respuesta del LLM."""
    sections = []
    if "## SECCIONES IDENTIFICADAS:" in response_text:
        sections_text = response_text.split("## SECCIONES IDENTIFICADAS:")[1]
        if "## REFERENCIAS BIBLIOGRÁFICAS:" in sections_text:
            sections_text = sections_text.split("## REFERENCIAS BIBLIOGRÁFICAS:")[0]
        for line in sections_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                section = line[2:].strip()
                if section:
                    sections.append(section)
    return sections


def parse_references_to_dict(text: str) -> Dict[str, str]:
    """Convierte el bloque de referencias en dict {numero: texto_completo}."""
    matches = list(
        re.finditer(r"- \[(\d+)\] (.+?)(?=(?:- \[\d+\])|\Z)", text, re.DOTALL)
    )
    return {
        m.group(1).strip(): m.group(2).strip().replace("\n", " ").replace("  ", " ")
        for m in matches
    }


def normalize_metadata(metadata: Dict) -> Dict:
    """Normaliza metadata para que sea consistente entre Chroma y FAISS.

    Listas → strings CSV, dicts → JSON. Así `references_mentioned` y
    `resolved_references` tienen el mismo tipo en ambos vectorstores
    y el agente no necesita branchear por tipo al leerlos.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            cleaned[key] = ", ".join(map(str, value)) if value else ""
        elif isinstance(value, dict):
            cleaned[key] = json.dumps(value) if value else "{}"
        else:
            cleaned[key] = str(value)
    return cleaned


def parse_refs_field(metadata: Dict) -> List[str]:
    """Lee `references_mentioned` de metadata y devuelve lista de números como strings."""
    raw = metadata.get("references_mentioned", "")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(r).strip() for r in raw if str(r).strip()]
    return [r.strip() for r in str(raw).split(",") if r.strip()]


def compute_pdf_hash(pdf_bytes: bytes) -> str:
    """SHA-256 del contenido del PDF — sirve para detectar papers ya procesados."""
    return hashlib.sha256(pdf_bytes).hexdigest()

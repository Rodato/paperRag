"""Persistencia del historial de consultas por paper.

Cada paper tiene su propio `history.json` en `data/papers/<sanitized>/history.json`
con la lista de queries y respuestas en orden cronológico inverso (más reciente primero).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "history.json"
MAX_HISTORY_ENTRIES = 100


def load_history(paper_dir: Path) -> List[Dict[str, Any]]:
    path = paper_dir / HISTORY_FILENAME
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        logger.exception("No se pudo leer history.json en %s", paper_dir)
        return []


def append_to_history(
    paper_dir: Path,
    entry: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Agrega una entrada al historial y la persiste. Retorna la entrada con timestamp/modelo."""
    enriched = {
        **entry,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelo": model_name,
    }
    history = load_history(paper_dir)
    history.insert(0, enriched)
    history = history[:MAX_HISTORY_ENTRIES]
    paper_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_dir / HISTORY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return enriched


def clear_history(paper_dir: Path) -> None:
    path = paper_dir / HISTORY_FILENAME
    if path.exists():
        path.unlink()

from pathlib import Path
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def save_paper_stores(
    paper_dir: Path,
    chroma_texts: List[str],
    chroma_metadatas: List[Dict],
    faiss_texts: List[str],
    faiss_metadatas: List[Dict],
    collection_name: str,
    embeddings: Embeddings,
) -> None:
    """Crea y persiste los dos vectorstores (Chroma + FAISS) en disco."""
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Chroma — se guarda con persist_directory
    chroma_path = str(paper_dir / "chroma_db")
    chroma_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )
    chroma_store.add_texts(texts=chroma_texts, metadatas=chroma_metadatas)

    # FAISS
    faiss_store = FAISS.from_texts(
        texts=faiss_texts,
        embedding=embeddings,
        metadatas=faiss_metadatas,
    )
    faiss_store.save_local(str(paper_dir / "faiss_index"))


def load_paper_stores(
    paper_dir: Path,
    collection_name: str,
    embeddings: Embeddings,
):
    """Carga Chroma y FAISS desde disco. Retorna (chroma_store, faiss_store)."""
    chroma_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(paper_dir / "chroma_db"),
    )
    faiss_store = FAISS.load_local(
        str(paper_dir / "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return chroma_store, faiss_store
